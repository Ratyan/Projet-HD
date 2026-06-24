#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Nanopore (Linux / macOS) : QC -> trimming -> filtrage -> alignement.

Etapes :
  1. NanoPlot   -> QC des reads bruts
  2. Porechop   -> retrait des adaptateurs
  3. NanoFilt   -> filtrage longueur (>= -l) et qualite (>= -q, definition ONT)
  4. NanoPlot   -> QC des reads filtres
  5. minimap2   -> alignement (preset map-ont) sur la reference
  6. samtools   -> tri, index, flagstat, view (apercu)
  7. samtools consensus -> sequence consensus FASTA (consensus.fasta)

Les outils manquants sont installes automatiquement (sans demander) :
  - NanoPlot / NanoFilt / Porechop : pip (Porechop : fallback source GitHub)
  - minimap2 / samtools           : conda/mamba -> brew (mac) -> apt (linux) ->
                                     binaire minimap2 (linux) -> build source

Usage :
  python3 nanopore_vntr_pipeline.py reads1.fastq.gz [reads2.fastq.gz ...] ref.fa [-o OUT] [-t N] [-l 1000] [-q 15]
  (la reference est toujours le DERNIER argument positionnel)
"""

import argparse
import gzip
import math
import os
import platform
import shutil
import site
import subprocess
import sys
import tempfile
import urllib.request

# ----------------------------------------------------------------------------- utils

def log(msg):
    print(f"\n\033[1;36m==> {msg}\033[0m", flush=True)

def warn(msg):
    print(f"\033[1;33m[!] {msg}\033[0m", flush=True)

def die(msg, code=1):
    print(f"\033[1;31m[ERREUR] {msg}\033[0m", file=sys.stderr, flush=True)
    sys.exit(code)

def have(tool):
    return shutil.which(tool) is not None

def run(cmd, shell=False, check=True):
    """Execute une commande en affichant ce qui est lance."""
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\033[0;90m$ {printable}\033[0m", flush=True)
    return subprocess.run(cmd, shell=shell, check=check,
                          executable="/bin/bash" if shell else None)

def add_user_bins_to_path():
    """Ajoute les dossiers de scripts pip --user au PATH du process."""
    extra = [os.path.expanduser("~/.local/bin"),
             os.path.join(site.getuserbase(), "bin")]
    for d in extra:
        if d and d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

def python_can_import(mod):
    """Teste si `mod` s'importe SANS erreur dans le meme Python que ce script
    (celui qu'utilisent NanoPlot/NanoFilt). Renvoie (ok: bool, stderr: str)."""
    r = subprocess.run([sys.executable, "-c", f"import {mod}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return r.returncode == 0, r.stderr.decode("utf-8", "replace")

# table de conversion score Phred (0..93) -> probabilite d'erreur, pour le filtre Python
_EP = [10 ** (-(i) / 10.0) for i in range(94)]

def count_reads(fq):
    """Compte les reads d'un FASTQ (.gz ou non)."""
    op = gzip.open if fq.endswith(".gz") else open
    n = 0
    with op(fq, "rt") as f:
        for i, _ in enumerate(f):
            if i % 4 == 1:
                n += 1
    return n

def merge_fastqs(paths, outpath):
    """Fusionne plusieurs FASTQ en un seul .fastq.gz.
    - si tous les fichiers sont en .gz : concatenation binaire (les membres gzip
      se concatenent et restent lisibles par tous les outils) -> rapide.
    - sinon : decompression/recompression uniforme en .fastq.gz."""
    all_gz = all(p.endswith(".gz") for p in paths)
    if all_gz:
        with open(outpath, "wb") as fo:
            for p in paths:
                with open(p, "rb") as fi:
                    shutil.copyfileobj(fi, fo)
    else:
        with gzip.open(outpath, "wt", compresslevel=4) as fo:
            for p in paths:
                iop = gzip.open if p.endswith(".gz") else open
                with iop(p, "rt") as fi:
                    shutil.copyfileobj(fi, fo)
    return outpath

def python_filter(infq, outfq, minlen, minq):
    """Filtre longueur/qualite en Python pur (repli si NanoFilt indisponible/casse).
    Qualite = definition ONT : -10*log10(probabilite d'erreur moyenne)."""
    iop = gzip.open if infq.endswith(".gz") else open
    kept = total = 0
    with iop(infq, "rt") as fi, gzip.open(outfq, "wt", compresslevel=4) as fo:
        while True:
            h = fi.readline()
            if not h:
                break
            s = fi.readline(); p = fi.readline(); q = fi.readline()
            total += 1
            seq = s.strip(); qual = q.strip()
            if len(seq) < minlen:
                continue
            if qual:
                b = qual.encode("ascii")
                mean_ep = sum(_EP[x - 33] for x in b) / len(b)
                meanq = -10.0 * math.log10(mean_ep) if mean_ep > 0 else 99.0
            else:
                meanq = 0.0
            if meanq < minq:
                continue
            fo.write(h); fo.write(s); fo.write(p); fo.write(q)
            kept += 1
    return kept, total

# ----------------------------------------------------------------------------- install

def pip_install(pkgs):
    base = [sys.executable, "-m", "pip", "install", "--user", "--upgrade"]
    try:
        run(base + list(pkgs))
    except subprocess.CalledProcessError:
        # Python "externally-managed" (PEP 668) : frequent sur PC recents/ecole
        warn("pip --user refuse (PEP 668), nouvelle tentative avec --break-system-packages")
        run(base + ["--break-system-packages", *pkgs])
    add_user_bins_to_path()

def ensure_pip_tool(tool, pip_name=None, git_url=None):
    if have(tool):
        return
    pip_name = pip_name or tool
    log(f"Installation de {tool} (pip)")
    try:
        pip_install([pip_name])
    except subprocess.CalledProcessError:
        if git_url:
            warn(f"pip {pip_name} a echoue, tentative depuis la source : {git_url}")
            pip_install([f"git+{git_url}"])
        else:
            raise
    if not have(tool):
        die(f"{tool} introuvable apres installation.")

def ensure_pandas_importable(allow_install=True):
    """NanoPlot et NanoFilt importent pandas des leur demarrage. Si pandas casse a
    cause d'un conflit d'ABI numpy/pandas ('numpy.dtype size changed, may indicate
    binary incompatibility'), on reinstalle un couple coherent (numpy<2) en espace
    utilisateur pour masquer les versions systeme incompatibles."""
    ok, err = python_can_import("pandas")
    if ok:
        return
    last = err.strip().splitlines()[-1] if err.strip() else "(pas de detail)"
    abi = ("numpy.dtype size changed" in err) or ("binary incompatibility" in err)
    if abi:
        warn(f"Conflit d'ABI numpy/pandas detecte : {last}")
    else:
        warn(f"pandas ne s'importe pas correctement : {last}")
    if not allow_install:
        die("NanoPlot/NanoFilt ont besoin de pandas, mais son import echoue.\n"
            "Corrige-le manuellement puis relance (ou retire --skip-install) :\n"
            f"  {sys.executable} -m pip install --user --force-reinstall "
            "--no-cache-dir 'numpy<2' 'pandas<2.2'")
    log("Reinstallation d'un couple numpy/pandas compatible (numpy<2)")
    pip_install(["--force-reinstall", "--no-cache-dir", "numpy<2", "pandas<2.2"])
    ok, err = python_can_import("pandas")
    if not ok:
        die("Le conflit numpy/pandas persiste apres reinstallation :\n" + err.strip() +
            "\n\nPiste : utilise un environnement isole (conda/venv) pour eviter le "
            "melange paquets systeme (/usr/local/lib) et pip --user (~/.local).")
    log("numpy/pandas OK desormais.")

def _try(cmd):
    try:
        run(cmd, check=True)
        return True
    except Exception:
        return False

def sudo_noninteractive_ok():
    """True seulement si sudo marche SANS mot de passe (jamais bloquant/interactif)."""
    if not have("sudo"):
        return False
    try:
        return subprocess.run(["sudo", "-n", "true"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False

def find_conda():
    """Cherche conda/mamba dans le PATH ou dans les emplacements user habituels."""
    for exe in ("mamba", "conda"):
        if have(exe):
            return exe
    for cand in ("~/miniconda3/bin/conda", "~/miniforge3/bin/conda",
                 "~/anaconda3/bin/conda", "~/mambaforge/bin/conda"):
        p = os.path.expanduser(cand)
        if os.path.isfile(p):
            os.environ["PATH"] = os.path.dirname(p) + os.pathsep + os.environ["PATH"]
            return p
    return None

def install_miniconda():
    """Installe Miniconda dans ~/miniconda3 SANS sudo (espace utilisateur). Renvoie le chemin de conda."""
    sysname = platform.system()           # Linux / Darwin
    mach = platform.machine()             # x86_64 / aarch64 / arm64
    osm = {"Linux": "Linux", "Darwin": "MacOSX"}.get(sysname)
    if osm is None:
        return None
    archm = {"x86_64": "x86_64", "amd64": "x86_64",
             "aarch64": "aarch64", "arm64": "arm64"}.get(mach.lower(), mach)
    url = f"https://repo.anaconda.com/miniconda/Miniconda3-latest-{osm}-{archm}.sh"
    prefix = os.path.expanduser("~/miniconda3")
    log(f"Installation de Miniconda (espace utilisateur, sans sudo) : {url}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            inst = os.path.join(tmp, "miniconda.sh")
            urllib.request.urlretrieve(url, inst)
            run(["bash", inst, "-b", "-p", prefix])   # -b = batch, pas de question
        conda = os.path.join(prefix, "bin", "conda")
        if os.path.isfile(conda):
            os.environ["PATH"] = os.path.dirname(conda) + os.pathsep + os.environ["PATH"]
            return conda
    except Exception as e:
        warn(f"Installation Miniconda echouee : {e}")
    return None

def conda_install(conda, pkgs):
    return _try([conda, "install", "-y", "-c", "bioconda", "-c", "conda-forge", *pkgs])

def install_minimap2_binary_linux(install_dir):
    """Telecharge le binaire precompile minimap2 (Linux x86_64) sans sudo."""
    ver = "2.28"
    url = (f"https://github.com/lh3/minimap2/releases/download/v{ver}/"
           f"minimap2-{ver}_x64-linux.tar.bz2")
    log(f"Telechargement du binaire minimap2 {ver}")
    os.makedirs(install_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar = os.path.join(tmp, "mm2.tar.bz2")
        urllib.request.urlretrieve(url, tar)
        run(["tar", "xjf", tar, "-C", tmp])
        src = os.path.join(tmp, f"minimap2-{ver}_x64-linux", "minimap2")
        dst = os.path.join(install_dir, "minimap2")
        shutil.copy(src, dst)
        os.chmod(dst, 0o755)
    os.environ["PATH"] = install_dir + os.pathsep + os.environ["PATH"]

def build_from_source(tool, install_dir):
    """Dernier recours : compile minimap2 / samtools depuis la source (make requis)."""
    if not have("make") or not (have("gcc") or have("cc")):
        return False
    os.makedirs(install_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            if tool == "minimap2":
                url = "https://github.com/lh3/minimap2/releases/download/v2.28/minimap2-2.28.tar.bz2"
                tar = os.path.join(tmp, "s.tar.bz2"); urllib.request.urlretrieve(url, tar)
                run(["tar", "xjf", tar, "-C", tmp])
                d = os.path.join(tmp, "minimap2-2.28")
                run(["make", "-C", d])
                shutil.copy(os.path.join(d, "minimap2"), os.path.join(install_dir, "minimap2"))
            elif tool == "samtools":
                url = "https://github.com/samtools/samtools/releases/download/1.21/samtools-1.21.tar.bz2"
                tar = os.path.join(tmp, "s.tar.bz2"); urllib.request.urlretrieve(url, tar)
                run(["tar", "xjf", tar, "-C", tmp])
                d = os.path.join(tmp, "samtools-1.21")
                run(["bash", "-c", f"cd '{d}' && ./configure --prefix='{os.path.dirname(install_dir)}' && make && make install"])
            os.environ["PATH"] = install_dir + os.pathsep + os.environ["PATH"]
            return have(tool)
        except Exception as e:
            warn(f"Build source de {tool} echoue : {e}")
            return False

def ensure_binaries(tools):
    """Installe minimap2/samtools par la meilleure methode dispo, en privilegiant
    les voies SANS sudo (utilisable sur PC verrouilles type ecole)."""
    missing = [t for t in tools if not have(t)]
    if not missing:
        return
    log(f"Installation de : {', '.join(missing)}")
    local_bin = os.path.expanduser("~/.local/bin")
    os.makedirs(local_bin, exist_ok=True)
    os.environ["PATH"] = local_bin + os.pathsep + os.environ["PATH"]

    def still():
        return [t for t in tools if not have(t)]

    # 1) conda / mamba deja present (bioconda, sans sudo, Linux + macOS)
    conda = find_conda()
    if conda:
        conda_install(conda, missing)
        if not still():
            return

    # 2) Homebrew (macOS, sans sudo)
    if sys.platform == "darwin" and have("brew"):
        _try(["brew", "install", *still()])
        if not still():
            return

    # 3) apt (Linux) UNIQUEMENT si sudo non interactif dispo -> jamais sur PC ecole.
    #    Sinon on saute proprement (pas de "permission denied").
    if sys.platform.startswith("linux") and have("apt-get"):
        if sudo_noninteractive_ok():
            _try(["sudo", "-n", "apt-get", "update"])
            _try(["sudo", "-n", "apt-get", "install", "-y", *still()])
            if not still():
                return
        else:
            warn("Pas de droits sudo : apt ignore (normal sur un PC d'ecole).")

    # 4) binaire minimap2 precompile (Linux x86_64, sans sudo)
    if "minimap2" in still() and sys.platform.startswith("linux") \
            and platform.machine().lower() in ("x86_64", "amd64"):
        try:
            install_minimap2_binary_linux(local_bin)
        except Exception as e:
            warn(f"Binaire minimap2 indisponible : {e}")

    # 5) Miniconda installe dans le HOME (sans sudo) : recupere minimap2 ET samtools
    if still():
        conda = find_conda() or install_miniconda()
        if conda:
            conda_install(conda, still())
            if not still():
                return

    # 6) build source en dernier recours (necessite make/gcc + libs)
    for t in still():
        build_from_source(t, local_bin)

    missing = still()
    if missing:
        die("Impossible d'installer sans privileges : " + ", ".join(missing) +
            ".\nSur un PC verrouille, le plus simple est d'installer Miniconda dans ton home :\n"
            "  bash <(curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-"
            f"{platform.system().replace('Darwin','MacOSX')}-{platform.machine()}.sh) -b -p ~/miniconda3\n"
            "  ~/miniconda3/bin/conda install -y -c bioconda -c conda-forge " + " ".join(missing))

# ----------------------------------------------------------------------------- pipeline

def main():
    ap = argparse.ArgumentParser(description="Pipeline Nanopore QC/trim/filter/align")
    ap.add_argument("fastq", nargs="+",
                    help="reads Nanopore (.fastq/.fastq.gz) — un ou plusieurs fichiers, "
                         "fusionnes automatiquement")
    ap.add_argument("reference", help="plasmide de reference (.fa / .fasta) — DOIT etre le dernier argument")
    ap.add_argument("-o", "--outdir", default="nanopore_out", help="dossier de sortie")
    ap.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("-l", "--min-length", type=int, default=1000, help="longueur min (NanoFilt)")
    ap.add_argument("-q", "--min-quality", type=float, default=15, help="qualite min (NanoFilt)")
    ap.add_argument("--skip-install", action="store_true", help="ne pas installer les outils")
    ap.add_argument("--from", "--start-step", dest="start_step", type=int, default=0,
                    choices=range(0, 8), metavar="N",
                    help="reprendre le pipeline a partir de l'etape N : 0=fusion, 1=QC brut, "
                         "2=Porechop, 3=filtrage, 4=QC filtre, 5=alignement, 6=stats, "
                         "7=consensus FASTA. Les sorties des etapes precedentes doivent deja "
                         "exister dans -o (defaut: 0, soit tout depuis le debut).")
    ap.add_argument("--consensus-mode", choices=["simple", "bayesian"], default="simple",
                    help="algorithme de samtools consensus (etape 7). 'simple' (defaut) = "
                         "comptage de frequences, bien plus rapide a haute profondeur ; "
                         "'bayesian' = defaut de samtools, plus lent.")
    ap.add_argument("--consensus-subsample", type=float, default=None, metavar="FRAC",
                    help="ne garder qu'une fraction des alignements pour le consensus "
                         "(ex. 0.05 = 5%%), utile quand la profondeur est enorme. Defaut: tous.")
    args = ap.parse_args()

    if os.name == "nt":
        die("Ce script est prevu pour Linux/macOS (sous Windows, lance-le via WSL).")
    for f in (*args.fastq, args.reference):
        if not os.path.isfile(f):
            die(f"Fichier introuvable : {f}")

    fastqs = [os.path.abspath(f) for f in args.fastq]
    ref = os.path.abspath(args.reference)
    out = os.path.abspath(args.outdir)
    qc = os.path.join(out, "qc")
    os.makedirs(qc, exist_ok=True)
    add_user_bins_to_path()

    # Etape de depart pour la reprise : do(N) == True si l'etape N doit etre executee
    start = args.start_step
    def do(step):
        return step >= start

    # --- outils (on installe/exige seulement ceux utiles aux etapes restantes) ---
    need_nanoplot = do(1) or do(4)   # NanoPlot sert aux etapes 1 et 4
    need_porechop = do(2)
    need_nanofilt = do(3)            # filtrage (repli Python si absent)
    need_minimap2 = do(5)
    need_samtools = do(5) or do(6) or do(7)   # tri/index/stats + consensus

    if not args.skip_install:
        if need_nanoplot:
            ensure_pip_tool("NanoPlot")
        if need_nanofilt:
            ensure_pip_tool("NanoFilt")
        if need_porechop:
            ensure_pip_tool("porechop", git_url="https://github.com/rrwick/Porechop.git")
        bins = (["minimap2"] if need_minimap2 else []) + (["samtools"] if need_samtools else [])
        if bins:
            ensure_binaries(bins)

    required = []
    if need_nanoplot: required.append("NanoPlot")
    if need_porechop: required.append("porechop")
    if need_nanofilt: required.append("NanoFilt")
    if need_minimap2: required.append("minimap2")
    if need_samtools: required.append("samtools")
    for t in required:
        if not have(t):
            die(f"Outil requis absent : {t} (relance sans --skip-install).")

    # Verifie/repare l'ABI numpy/pandas : NanoPlot et NanoFilt importent pandas des
    # le demarrage, et l'installation pip ci-dessus peut tirer un numpy 2.x
    # incompatible avec le pandas systeme. A faire APRES tous les pip install.
    if need_nanoplot or need_nanofilt:
        ensure_pandas_importable(allow_install=not args.skip_install)

    trimmed = os.path.join(out, "trimmed.fastq.gz")
    filtered = os.path.join(out, "filtered.fastq.gz")
    bam = os.path.join(out, "aligned.sorted.bam")
    consensus = os.path.join(out, "consensus.fasta")
    th = str(args.threads)

    # 0. Fusion des fichiers de reads d'entree (si plusieurs)
    if start > 0:
        log(f"Reprise du pipeline a l'etape {start}/7 — les sorties des etapes "
            f"0..{start - 1} sont reutilisees depuis {out}")

    if len(fastqs) == 1:
        fastq = fastqs[0]
    else:
        fastq = os.path.join(out, "merged.fastq.gz")
        if do(0):
            log(f"0/7 Fusion de {len(fastqs)} fichiers de reads -> merged.fastq.gz")
            for p in fastqs:
                print(f"    + {p}")
            merge_fastqs(fastqs, fastq)
            if not os.path.exists(fastq) or count_reads(fastq) == 0:
                die("La fusion n'a produit aucun read. Verifie les FASTQ d'entree.")
            print(f"reads totaux apres fusion : {count_reads(fastq)}")
        elif not os.path.exists(fastq):
            die(f"Reprise a l'etape {start} demandee mais le fichier fusionne est absent :\n"
                f"  {fastq}\nRelance avec --from 0 pour le regenerer.")
        else:
            log(f"0/7 Fusion ignoree (reprise) — reutilisation de {fastq}")

    # Verifie que les entrees produites par les etapes sautees existent bien.
    def _need(path, label, producing_step):
        if not os.path.exists(path):
            die(f"Reprise a l'etape {start} impossible : '{label}' manquant.\n"
                f"  Attendu : {path}\n"
                f"  Cette entree est produite par l'etape {producing_step}. "
                f"Relance avec --from {producing_step} (ou --from 0) pour la (re)generer.")
    if start == 3:
        _need(trimmed, "reads trimmes (Porechop)", 2)
    if start in (4, 5):
        _need(filtered, "reads filtres", 3)
    if start in (6, 7):
        _need(bam, "alignement BAM trie", 5)

    # 1. NanoPlot brut
    if do(1):
        log("1/7 NanoPlot — QC des reads bruts")
        run(["NanoPlot", "--fastq", fastq, "-o", os.path.join(qc, "nanoplot_raw"),
             "-t", th, "-p", "raw_", "--N50", "--loglength"])

    # 2. Porechop
    if do(2):
        log("2/7 Porechop — retrait des adaptateurs")
        run(["porechop", "-i", fastq, "-o", trimmed, "-t", th, "--discard_middle"])
        if not os.path.exists(trimmed) or count_reads(trimmed) == 0:
            die("Porechop n'a produit aucun read (trimmed.fastq.gz vide). Verifie le FASTQ d'entree.")

    # 3. Filtrage longueur/qualite : NanoFilt si possible, sinon repli Python integre
    if do(3):
        log(f"3/7 Filtrage (longueur >= {args.min_length}, Q >= {args.min_quality})")
        ok = False
        # NanoFilt n'accepte qu'un ENTIER pour -q : on convertit 15.0 -> 15. Si le seuil
        # demande est decimal (ex. 12.5), NanoFilt ne sait pas le gerer -> filtre Python.
        q_entier = float(args.min_quality).is_integer()
        if have("NanoFilt") and q_entier:
            qval = int(args.min_quality)
            # set -o pipefail : si NanoFilt plante, le pipe renvoie une erreur (au lieu d'un .gz vide silencieux)
            r = run(f"set -o pipefail; gzip -dc '{trimmed}' | "
                    f"NanoFilt -l {args.min_length} -q {qval} | "
                    f"gzip > '{filtered}'", shell=True, check=False)
            ok = (r.returncode == 0 and os.path.exists(filtered)
                  and os.path.getsize(filtered) > 0 and count_reads(filtered) > 0)
            if not ok:
                warn("NanoFilt a echoue ou n'a produit aucun read "
                     "(souvent incompatible avec Python recent) -> repli sur le filtre Python integre.")
        elif have("NanoFilt") and not q_entier:
            warn(f"NanoFilt n'accepte qu'une qualite entiere ; seuil -q {args.min_quality} "
                 "decimal -> filtre Python integre.")
        else:
            warn("NanoFilt absent -> filtre Python integre.")
        if not ok:
            kept, tot = python_filter(trimmed, filtered, args.min_length, args.min_quality)
            print(f"filtre Python : {kept}/{tot} reads conserves")
        n_filt = count_reads(filtered)
        if n_filt == 0:
            die(f"Aucun read apres filtrage (seuils -l {args.min_length} / -q {args.min_quality} "
                f"trop stricts, ou trimmed vide).")
        print(f"reads apres filtrage : {n_filt}")

    # 4. NanoPlot filtre
    if do(4):
        log("4/7 NanoPlot — QC des reads filtres")
        run(["NanoPlot", "--fastq", filtered, "-o", os.path.join(qc, "nanoplot_filtered"),
             "-t", th, "-p", "filt_", "--N50", "--loglength"])

    # 5. minimap2 + tri samtools
    if do(5):
        log("5/7 minimap2 — alignement (map-ont) + tri samtools")
        run(f"minimap2 -ax map-ont -t {th} --secondary=no '{ref}' '{filtered}' "
            f"| samtools sort -@ {th} -o '{bam}' -", shell=True)
        run(["samtools", "index", bam])

    # 6. samtools view / flagstat
    if do(6):
        log("6/7 samtools — statistiques et apercu")
        run(["samtools", "flagstat", bam])
        print("\n--- apercu des 3 premiers alignements (samtools view) ---", flush=True)
        run(f"samtools view '{bam}' | head -3 | cut -c1-120", shell=True, check=False)

    # 7. samtools consensus -> sequence consensus FASTA
    if do(7):
        mode = args.consensus_mode
        log(f"7/7 samtools consensus (mode {mode}) — sequence consensus FASTA")
        # 'samtools consensus' existe depuis samtools >= 1.16. On verifie sa presence
        # via la liste des sous-commandes : le code retour de '... --help' n'est pas
        # fiable (plusieurs sous-commandes samtools renvoient un code != 0 sur --help).
        h = subprocess.run(["samtools", "--help"], capture_output=True, text=True)
        if "consensus" not in (h.stdout + h.stderr):
            die("Ta version de samtools ne propose pas la sous-commande 'consensus' "
                "(necessite samtools >= 1.16). Mets samtools a jour, puis relance avec --from 7.")
        cons = f"samtools consensus -@ {th} -m {mode} -f fasta -o '{consensus}'"
        sub = args.consensus_subsample
        if sub is not None:
            if not (0.0 < sub < 1.0):
                die("--consensus-subsample doit etre strictement entre 0 et 1 (ex. 0.05 pour 5%).")
            log(f"  sous-echantillonnage a {sub:.1%} des alignements avant consensus "
                "(profondeur reduite -> plus rapide)")
            run(f"samtools view -b -@ {th} -s {sub} '{bam}' | {cons} -", shell=True)
        else:
            run(f"{cons} '{bam}'", shell=True)
        if not os.path.exists(consensus) or os.path.getsize(consensus) == 0:
            die("samtools consensus n'a produit aucune sequence (consensus.fasta vide). "
                "Verifie que le BAM contient des alignements (etape 6 / flagstat).")
        print(f"consensus ecrit : {consensus}")

    log("Termine.")
    print(f"""
Resultats dans : {out}
  - QC brut      : {os.path.join(qc, 'nanoplot_raw', 'raw_NanoPlot-report.html')}
  - QC filtre    : {os.path.join(qc, 'nanoplot_filtered', 'filt_NanoPlot-report.html')}
  - reads trimmes: {trimmed}
  - reads filtres: {filtered}
  - alignement   : {bam} (+ .bai)
  - consensus    : {consensus}
""")

if __name__ == "__main__":
    main()


