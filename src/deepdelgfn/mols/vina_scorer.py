#!/usr/bin/env python3
from __future__ import annotations
import shutil, subprocess, sys, tempfile, os, uuid
from typing import Tuple, Optional
from pathlib import Path
from typing import Tuple

# Docking boxes for the receptors shipped under data/targets/.
# Each entry is (cx, cy, cz, sx, sy, sz). Centers and sizes are sourced from the
# RGFN docking proxy:
#   RGFN/rgfn/gfns/reaction_gfn/proxies/docking_proxy/docking_proxy.py
TARGET_BOXES = {
    "Mpro": (-20.458, 18.109, -26.914, 18.0, 18.0, 18.0),
    "TBLR1": (-1.014, 42.097, 39.750, 18.0, 18.0, 18.0),
    "ClpP": (-38.127, 45.671, -20.898, 17.0, 17.0, 17.0),
    "sEH": (-13.4, 26.3, -13.3, 20.013, 16.3, 18.5),
}


def box_for_target(target: str) -> Tuple[float, float, float, float, float, float]:
    """Return the Vina box (cx, cy, cz, sx, sy, sz) for a named target."""
    if target not in TARGET_BOXES:
        raise ValueError(
            f"Unknown target {target!r}. Valid targets: {sorted(TARGET_BOXES)}"
        )
    return TARGET_BOXES[target]


# Backward-compatible alias: the sEH/4JNC box.
DEFAULT_4JNC_BOX: Tuple[float, float, float, float, float, float] = TARGET_BOXES["sEH"]

class DockingScorer:
    """
    Minimal, robust wrapper around AutoDock Vina/QuickVina for scoring a SMILES.
    - SMILES → 3D SDF + PDB (RDKit)
    - PDB/SDF → PDBQT (ADFR prepare_ligand → Meeko CLI → Open Babel)
    - Dock with vina/qvina02 and parse 'REMARK VINA RESULT'
    """

    def __init__(
        self,
        receptor_pdbqt: str | Path,
        engine_path: str | Path = "vina",
        center_size: Tuple[float, float, float, float, float, float] = DEFAULT_4JNC_BOX,
        exhaustiveness: int = 8,
        seed: int = 42,
        cpu: int = 1,
        tmp_dir: str | Path | None = None,
    ) -> None:
        self.receptor_pdbqt = receptor_pdbqt
        self.receptor = Path(receptor_pdbqt).resolve()
        if not self.receptor.exists():
            raise FileNotFoundError(f"Receptor PDBQT not found: {self.receptor}")
        self.engine_path = str(engine_path)
        self.center_size = center_size
        self.exhaustiveness = int(exhaustiveness)
        self.seed = int(seed)
        self.cpu = cpu
        # --- tmp_dir handling (NEW) ---
        # Normalize and create project-local tmp dir if provided; otherwise None
        if tmp_dir is None:
            self.tmp_dir = None
        else:
            from pathlib import Path as _Path
            self.tmp_dir = _Path(tmp_dir).resolve()
            self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # sanity for engine presence (string path OR on PATH)
        if "/" in self.engine_path:
            if not Path(self.engine_path).exists():
                raise FileNotFoundError(f"Docking engine not found: {self.engine_path}")
        else:
            if shutil.which(self.engine_path) is None:
                raise FileNotFoundError(f"Docking engine '{self.engine_path}' not found on PATH")

    # ---------------- internal helpers (chemistry lives here) ----------------

    @staticmethod
    def _smiles_to_3d_files(smiles: str, out_sdf: Path, out_pdb: Path) -> None:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        out_sdf.parent.mkdir(parents=True, exist_ok=True)
        out_pdb.parent.mkdir(parents=True, exist_ok=True)

        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise RuntimeError("RDKit failed to parse SMILES.")
        m = Chem.AddHs(m)

        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(m, p) != 0:
            p.useRandomCoords = True
            if AllChem.EmbedMolecule(m, p) != 0:
                raise RuntimeError("3D embedding failed.")
        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass

        Chem.MolToMolFile(m, str(out_sdf))
        Chem.MolToPDBFile(m, str(out_pdb))

    @staticmethod
    def _make_ligand_pdbqt(in_pdb: Path, in_sdf: Path, out_pdbqt: Path, ph: float = 7.4) -> None:
        out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        in_pdb = in_pdb.resolve(); in_sdf = in_sdf.resolve(); out_pdbqt = out_pdbqt.resolve()

        # 1) ADFR 'prepare_ligand' (PDB)
        if shutil.which("prepare_ligand"):
            try:
                subprocess.run(
                    ["prepare_ligand", "-l", in_pdb.name, "-o", out_pdbqt.name],
                    check=True, capture_output=True, text=True, cwd=in_pdb.parent
                )
                produced = in_pdb.parent / out_pdbqt.name
                if produced.exists():
                    produced.replace(out_pdbqt)
                if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
                    return
            except subprocess.CalledProcessError as e:
                print("[prepare_ligand failed]\n", e.stdout or "", e.stderr or "")

        # 2) Meeko CLI (SDF)
        if shutil.which("mk_prepare_ligand.py"):
            try:
                subprocess.run(
                    ["mk_prepare_ligand.py", "-i", str(in_sdf), "-o", str(out_pdbqt)],
                    check=True, capture_output=True, text=True
                )
                if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
                    return
            except subprocess.CalledProcessError as e:
                print("[mk_prepare_ligand.py failed]\n", e.stdout or "", e.stderr or "")

        # 3) Open Babel fallback (SDF→PDBQT, Gasteiger charges)
        if shutil.which("obabel") is None:
            raise RuntimeError("Open Babel not found. Install via conda-forge: conda install -c conda-forge openbabel")
        try:
            subprocess.run(
                ["obabel", "-isdf", str(in_sdf), "-opdbqt", "-O", str(out_pdbqt),
                 "-p", str(ph), "-h", "--partialcharge", "gasteiger"],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print("[obabel failed]\n", e.stdout or "", e.stderr or "")
            raise
        if not (out_pdbqt.exists() and out_pdbqt.stat().st_size > 0):
            raise RuntimeError("Open Babel did not produce a PDBQT.")

    def _run_vina(self, ligand_pdbqt: Path, out_pdb: Path, log_path: Optional[Path] = None) -> str:
        """
        Run vina (or qvina) on ligand_pdbqt, write out_pdb, and optionally save stdout/stderr
        to log_path for debugging. Returns stdout (string) on success, raises on failure.
        """
        cx, cy, cz, sx, sy, sz = self.center_size
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        is_gpu = 'gpu' in self.engine_path.lower()
        cmd = [
            self.engine_path,
            "--receptor", str(self.receptor),
            "--ligand",   str(ligand_pdbqt),
            "--center_x", f"{cx}", "--center_y", f"{cy}", "--center_z", f"{cz}",
            "--size_x",   f"{sx}", "--size_y",   f"{sy}", "--size_z",   f"{sz}",
            "--exhaustiveness", str(self.exhaustiveness),
            "--seed", str(self.seed),
        ]
        if not is_gpu:
            cmd.extend(["--cpu", str(self.cpu)])
        cmd.extend(["--out", str(out_pdb)])
        print(f"[debug] CMD: {' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # write per-ligand log if requested
            if log_path is not None:
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_text = "=== STDOUT ===\n" + (p.stdout or "") + "\n\n=== STDERR ===\n" + (p.stderr or "")
                    log_path.write_text(log_text)
                except Exception as e:
                    # If logging fails, print a warning but don't crash the run
                    print(f"[warn] could not write vina log {log_path}: {e}", file=sys.stderr)
            return p.stdout
        except subprocess.CalledProcessError as e:
            # write failing stdout/stderr if possible
            if log_path is not None:
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_text = "=== STDOUT ===\n" + (e.stdout or "") + "\n\n=== STDERR ===\n" + (e.stderr or "")
                    log_path.write_text(log_text)
                except Exception:
                    pass
            # print failure diagnostics and re-raise so caller logs the failure
            print("\n[Vina failed]\nCMD:\n", " ".join(cmd),
                "\n--- STDOUT ---\n", e.stdout or "", "\n--- STDERR ---\n", e.stderr or "", sep="")
            raise

    @staticmethod
    def _parse_score_from_outfile(outfile: Path) -> float:
        lines = outfile.read_text().splitlines()
        if len(lines) >= 2 and lines[1].startswith("REMARK VINA RESULT"):
            return float(lines[1].split()[3])
        for ln in lines:
            if ln.startswith("REMARK VINA RESULT"):
                return float(ln.split()[3])
        raise RuntimeError("Couldn't find 'REMARK VINA RESULT' in Vina output.")

    # ------------------------------ public API ------------------------------

    def score_smiles(self, smiles: str, *, cleanup: bool = True) -> float:
        """
        Score a single SMILES string with AutoDock Vina.

        A unique temporary directory is created per call (under ``tmp_dir`` if
        configured, otherwise the system temp directory) to hold the intermediate
        ligand files (SDF, PDB, PDBQT, docked PDB, and a Vina log).  By default
        the temporary directory is *removed immediately after the score is
        extracted* to keep scratch inode usage under control; pass
        ``cleanup=False`` for debugging.

        Parameters
        ----------
        smiles : str
            SMILES string to score.
        cleanup : bool
            If True (default), delete the per-molecule work directory after
            extracting the score.  Set to False to preserve intermediate
            ligand/dock files for manual inspection.
        """
        if self.tmp_dir is None:
            td = Path(tempfile.mkdtemp())
        else:
            unique = f"score_{os.getpid()}_{uuid.uuid4().hex[:12]}"
            td = Path(self.tmp_dir) / unique
            td.mkdir(parents=True, exist_ok=True)

        sdf = td / "ligand.sdf"
        pdb = td / "ligand.pdb"
        pdbqt = td / "ligand.pdbqt"
        out_pdb = td / "docked.pdb"
        log_path = td / "vina.log"

        try:
            self._smiles_to_3d_files(smiles, sdf, pdb)
            self._make_ligand_pdbqt(pdb, sdf, pdbqt)
            self._run_vina(pdbqt, out_pdb, log_path=log_path)
            score = self._parse_score_from_outfile(out_pdb)
        finally:
            if cleanup:
                shutil.rmtree(td, ignore_errors=True)

        return score


# ---------------------------------------------------------------------------
# DOCK3 scorer — uses ligbuild + dock64 instead of Vina
# ---------------------------------------------------------------------------

class Dock3Scorer:
    """
    Score a single SMILES using DOCK3 (ligbuild → dock64) against a prepared receptor.

    Pipeline per call to score_smiles():
      1. Write a one-line .smi file.
      2. Run ``ligbuild`` (sourcing dockenv.sh) to produce a .db2 bundle (.tgz).
      3. Unpack the .tgz to obtain the .db2 file.
      4. Patch the INDOCK template (version header + ligand filename) and run dock64.
      5. Parse the DOCK3 total score from OUTDOCK.

    Both Dock3Scorer and DockingScorer expose the same ``score_smiles(smiles) -> float``
    interface so they are drop-in substitutes in score_library.py.
    """

    _INDOCK_VER_OLD = "DOCK 3.7 parameter"
    _INDOCK_VER_NEW = "DOCK 3.8 parameter"
    _INDOCK_LIGAND_PLACEHOLDER = "split_database_index"

    def __init__(
        self,
        indock_template: str | Path,
        dockfiles_dir: str | Path,
        dockenv_sh: str | Path = "/project/rrg-mailhoto/share/dockingpackages/dockenv.sh",
        dock64_exe: str | Path = "/project/rrg-mailhoto/share/dock64",
        ligbuild_exe: str = "ligbuild",
        tmp_dir: str | Path | None = None,
        strain_weight: float = 0.0,
        timeout: int = 300,
        ligbuild_timeout: int = 600,
    ) -> None:
        self.indock_template = Path(indock_template).resolve()
        self.dockfiles_dir = Path(dockfiles_dir).resolve()
        self.dockenv_sh = str(dockenv_sh)
        self.dock64_exe = str(dock64_exe)
        self.ligbuild_exe = ligbuild_exe
        self.strain_weight = float(strain_weight)
        self.timeout = int(timeout)
        self.ligbuild_timeout = int(ligbuild_timeout)
        self.last_failure_reason: Optional[str] = None

        if not self.indock_template.exists():
            raise FileNotFoundError(f"INDOCK template not found: {self.indock_template}")
        if not self.dockfiles_dir.is_dir():
            raise FileNotFoundError(
                f"dockfiles_dir not found or not a directory: {self.dockfiles_dir}"
            )

        if tmp_dir is None:
            self.tmp_dir = None
        else:
            self.tmp_dir = Path(tmp_dir).resolve()
            self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Warm up the docking env once, single-threaded, BEFORE any parallel
        # ligbuild call.  ``dockenv.sh`` bootstraps a Python venv (build_3d_dock_py
        # + openeye + Pyro4 + ...) on first source on a fresh compute node, and
        # if N>1 worker threads source it concurrently they race on `pip install`
        # and corrupt the venv (e.g. "No such file or directory: .../Pyro4/constants.py",
        # then "ModuleNotFoundError: No module named 'openeye'" for every ligbuild).
        # Calling the warmup here is safe because score_library.py constructs
        # Dock3Scorer once on the main thread before launching the ThreadPoolExecutor.
        self._warmup_dockenv()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _classify_failure(stage: str, message: str) -> str:
        """Return a compact, summary-CSV-friendly failure label."""
        text = message.lower()
        if stage == "ligbuild":
            if "timeout reached for" in text and ".db2" in text:
                return "ligbuild_db2_timeout"
            if "timed out" in text or "timeoutexpired" in text:
                return "ligbuild_subprocess_timeout"
            if "error in build_db2" in text and "list index out of range" in text:
                return "ligbuild_build_db2_index_error"
            if "protomer builds failed" in text:
                return "ligbuild_protomer_build_failed"
            if "produced no .tgz" in text:
                return "ligbuild_no_tgz"
            return "ligbuild_failed"
        if stage == "db2_extract":
            return "db2_extract_failed"
        if stage == "dock64":
            if "produced no outdock" in text:
                return "dock64_no_outdock"
            if "timed out" in text or "timeoutexpired" in text:
                return "dock64_timeout"
            return "dock64_failed"
        if stage == "outdock_parse":
            return "dock64_no_pose_or_score"
        return f"{stage}_failed"

    def _warmup_dockenv(self, warmup_timeout: int = 600) -> None:
        """Source dockenv.sh once and probe that ligbuild + openeye are importable.

        On a fresh compute node the first ``source dockenv.sh`` triggers a
        ``pip install`` of build_3d_dock_py and its OpenEye/Pyro4 dependencies
        into a node-local venv. Running this serially before any parallel
        ligbuild call eliminates the concurrent-pip-install race that otherwise
        corrupts the venv when ``--jobs > 1``.

        This is best-effort: failures are logged as warnings, not raised, so
        nodes that already have a healthy env pre-built still proceed (and the
        per-call ligbuild will surface the real error if any).
        """
        probe_cmd = (
            f'source "{self.dockenv_sh}" && '
            f'command -v "{self.ligbuild_exe}" >/dev/null && '
            f'python -c "import openeye, Pyro4" 2>&1'
        )
        try:
            result = subprocess.run(
                ["bash", "-lc", probe_cmd],
                capture_output=True, text=True, timeout=warmup_timeout,
            )
        except subprocess.TimeoutExpired:
            print(
                f"[dock3][warn] dockenv warmup timed out after {warmup_timeout}s "
                f"(dockenv_sh={self.dockenv_sh}). Continuing; per-call ligbuild "
                f"may fail if the venv is not provisioned.",
                file=sys.stderr,
            )
            return
        except Exception as e:
            print(f"[dock3][warn] dockenv warmup raised {type(e).__name__}: {e}", file=sys.stderr)
            return

        if result.returncode == 0:
            print(f"[dock3] dockenv warmup OK (dockenv_sh={self.dockenv_sh})")
        else:
            tail = (result.stdout or "") + (result.stderr or "")
            tail = tail[-1500:] if len(tail) > 1500 else tail
            print(
                f"[dock3][warn] dockenv warmup probe exited rc={result.returncode}. "
                f"Continuing — per-call ligbuild will retry. Last output:\n{tail}",
                file=sys.stderr,
            )


    @staticmethod
    def _short_workdir_base() -> Path:
        """Return a short (~12 char) directory alias to use as the workdir base.

        AMSOL (called by ligbuild) has a fixed-width Fortran path buffer (~80
        chars). Once the per-protomer / per-tautomer / per-conformer subdirs
        are appended underneath the workdir, the workdir prefix must stay
        short (~25 chars or less) or AMSOL silently corrupts builds — leaving
        a degraded ``.db2`` and a much weaker DOCK3 score (e.g. -16 instead
        of -71 for the same molecule). When the workdir lives under
        ``$SLURM_TMPDIR`` (e.g. ``/localscratch/<user>.<JOBID>.0/`` ≈ 30
        chars) the threshold is already breached for many ligands.

        To keep the fast local NVMe scratch *and* a short path, we expose
        ``$SLURM_TMPDIR`` (or ``/tmp`` on login nodes) via a short symlink at
        ``/tmp/d.<JOB_TAG>`` and place all per-call workdirs under that
        alias. The alias is per-Slurm-job (or ``cli`` outside Slurm) so
        concurrent jobs on the same node never collide on the symlink target.
        """
        job_tag = os.environ.get("SLURM_JOB_ID") or os.environ.get("RUN_ID") or "cli"
        alias = Path(f"/tmp/d.{job_tag}")
        slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
        if slurm_tmpdir:
            target = Path(slurm_tmpdir)
            try:
                if alias.is_symlink():
                    if alias.resolve() != target.resolve():
                        alias.unlink()
                        alias.symlink_to(target, target_is_directory=True)
                elif alias.exists():
                    # Pre-existing real dir at the alias path: keep using it.
                    pass
                else:
                    alias.symlink_to(target, target_is_directory=True)
            except OSError:
                # Symlink path failed (e.g. /tmp not writable): fall back to
                # a plain /tmp directory; AMSOL still gets a short prefix.
                alias.mkdir(parents=True, exist_ok=True)
        else:
            alias.mkdir(parents=True, exist_ok=True)
        return alias

    def _make_workdir(self, smiles: str) -> Path:
        """Return a unique per-molecule work directory.

        Uses :meth:`_short_workdir_base` so the workdir prefix stays short
        enough for AMSOL's fixed-width Fortran path buffer (see that method's
        docstring for the full rationale).

        When ``self.tmp_dir`` is None (the default), the workdir is an
        ephemeral ``/tmp/d.<JOBTAG>/dXXXXXX`` directory.  When the user passes
        an explicit ``tmp_dir`` we respect it (for debugging) and warn if its
        prefix is long enough to risk AMSOL corruption.
        """
        if self.tmp_dir is None:
            short_base = self._short_workdir_base()
            return Path(tempfile.mkdtemp(prefix="d", dir=str(short_base)))
        # User-supplied tmp_dir: warn if its prefix is long enough to break
        # AMSOL once the per-protomer subdirs are appended.
        if len(str(self.tmp_dir)) > 30:
            print(
                f"[dock3][warn] tmp_dir='{self.tmp_dir}' is {len(str(self.tmp_dir))} "
                "chars; ligbuild/AMSOL may silently corrupt builds when total "
                "paths exceed ~80 chars. Prefer a tmp_dir under /tmp/ "
                "(<=25 chars) for reliable docking scores.",
                file=sys.stderr,
            )
        unique = f"dock3_{os.getpid()}_{abs(hash(smiles)) % (10 ** 8)}"
        td = self.tmp_dir / unique
        td.mkdir(parents=True, exist_ok=True)
        return td

    def _run_ligbuild(self, smi_file: Path, out_dir: Path) -> Path:
        """
        Run ligbuild on *smi_file* and return the path of the produced .tgz bundle.
        Raises RuntimeError if no .tgz is produced.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write a custom_parms override with verbose=1 so that build_db2 prints
        # error messages instead of silently returning None (default verbose=0).
        import json as _json
        custom_parms_path = out_dir.parent / "custom_parms.json"
        custom_parms_path.write_text(_json.dumps({"verbose": 1, "timeout": self.ligbuild_timeout}))

        cmd = (
            f'source "{self.dockenv_sh}" && '
            f'{self.ligbuild_exe} "{smi_file}" "{out_dir}" "{custom_parms_path}"'
        )
        # Run ligbuild with cwd=out_dir.parent (= unique td/) so that:
        #   - db2_outputs/ and db2_archives/ are isolated per call (no race conditions)
        #   - the final `cp db2_archives/bundle.tgz <output_dir>/` succeeds because
        #     output_dir (out_dir) differs from cwd, so the tgz ends up in out_dir
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=self.timeout,
            cwd=str(out_dir.parent),
        )
        # Do NOT raise on non-zero rc here: ligbuild commonly exits rc=1 because
        # shutil.rmtree("db2_outputs") fails (cleanup race in parallel jobs) AFTER
        # the .tgz has already been produced and copied to out_dir.  Instead,
        # search for the tgz and only error if it is truly absent.
        #
        # Search priority:
        #   1. out_dir/*.tgz          — normal cp destination
        #   2. out_dir/db2_archives/  — fallback if cp failed
        #   3. cwd (td) /db2_archives/ — if CWD == out_dir was the case previously
        tgz_files = (
            list(out_dir.glob("*.tgz"))
            or list((out_dir / "db2_archives").glob("*.tgz"))
            or list((out_dir.parent / "db2_archives").glob("*.tgz"))
        )
        if not tgz_files:
            # Also do a broad recursive search as last resort
            tgz_files = list(out_dir.parent.rglob("*.tgz"))

        if not tgz_files:
            raise RuntimeError(
                f"ligbuild produced no .tgz under {out_dir.parent} "
                f"(rc={result.returncode}, ligbuild_timeout={self.ligbuild_timeout}s, "
                f"subprocess_timeout={self.timeout}s).\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        return tgz_files[0]

    def _extract_db2(self, tgz_path: Path, extract_dir: Path) -> Path:
        """Extract the .tgz bundle and return the path of the first .db2 file."""
        import tarfile
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        # The tgz contains a sub-directory (e.g. bundle_lig_000/lig.db2),
        # so use rglob to find the .db2 regardless of depth.
        db2_files = list(extract_dir.rglob("*.db2"))
        if not db2_files:
            raise RuntimeError(f"No .db2 file found after extracting {tgz_path}")
        return db2_files[0]

    def _build_indock(self, db2_path: Path, dest_dir: Path) -> Path:
        """
        Read the INDOCK template, patch the version header and ligand filename,
        write the patched file into dest_dir, and return its path.

        db2_path is absolute; we use its path relative to dest_dir (which is
        also the dock64 CWD), e.g. 'bundle_lig_000/lig.db2'.
        """
        text = self.indock_template.read_text()
        # dock64 at /project/rrg-mailhoto/share/dock64 is DOCK 3.8 and requires
        # the "DOCK 3.8 parameter" header (the INDOCK template is written as 3.7).
        text = text.replace(self._INDOCK_VER_OLD, self._INDOCK_VER_NEW)
        # Compute the path dock64 will use to find the .db2 (relative to its CWD=dest_dir)
        try:
            db2_rel = db2_path.relative_to(dest_dir)
        except ValueError:
            # db2 is not under dest_dir — fall back to absolute path
            db2_rel = db2_path
        text = text.replace(self._INDOCK_LIGAND_PLACEHOLDER, str(db2_rel))
        out = dest_dir / "INDOCK_run"
        out.write_text(text)
        return out

    def _run_dock64(self, work_dir: Path, indock_path: Path) -> None:
        """
        Copy dock64 into work_dir and run it there so that relative paths inside
        INDOCK (../dockfiles/…) resolve against the correct parent directory.

        IMPORTANT: dockfiles must be a REAL COPY (not a symlink) of the receptor
        grid directory placed at work_dir.parent/dockfiles.  The large Fortran
        binary grid files (.phi, .bmp, .vdw, .desolv) read incorrectly through
        symlinks on networked/Lustre filesystems, causing silent mis-scoring.
        This matches exactly what omltk.docking.run_docking_from_tgz does:
            shutil.copytree(dockfiles, os.path.join(base_dir, 'dockfiles'))
        """
        dock64_dest = work_dir / "dock64"
        if not dock64_dest.exists():
            shutil.copy2(self.dock64_exe, str(dock64_dest))
            dock64_dest.chmod(dock64_dest.stat().st_mode | 0o111)

        # dockfiles must be a sibling directory of work_dir called 'dockfiles'
        # so that '../dockfiles/...' in INDOCK resolves correctly.
        # Use a REAL COPY (not a symlink) to avoid Fortran binary I/O issues
        # with networked filesystems (Lustre, GPFS, etc.).
        dockfiles_copy = work_dir.parent / "dockfiles"
        if not dockfiles_copy.exists():
            shutil.copytree(str(self.dockfiles_dir), str(dockfiles_copy))

        result = subprocess.run(
            ["./dock64", indock_path.name],
            capture_output=True, text=True, cwd=str(work_dir),
            timeout=self.timeout,
        )
        # dock64 exits with non-zero even on success (ieee_inexact signal); check OUTDOCK instead.
        outdock = work_dir / "OUTDOCK"
        if not outdock.exists():
            raise RuntimeError(
                f"dock64 produced no OUTDOCK.\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

    @staticmethod
    def _parse_outdock_score(outdock_path: Path) -> Optional[float]:
        """
        Return the best 'total' score from the raw OUTDOCK file, or None if no
        pose was placed.

        Column layout (0-indexed from start of split line):
          0=mol#  1=id_num  2=flexiblecode  3=matched  4=nscored  5=time
          6=hac  7=setnum  8=matnum  9=rank  10=charge  11=elect  12=gist
          13=vdW  14=psol  15=asol  16=tStrain  17=mStrain  18=rec_d
          19=r_hyd  20=Total

        Succeeded lines have exactly 21 fields; the Total is parts[20].

        DOCK3 Fortran field-width overflow bug
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        When the Total value is too large for its fixed-width Fortran field,
        dock64 writes '**********' instead of the number. Because there is no
        space separator between r_hyd and Total in the format string, the two
        fields merge into one token, e.g. '0.00**********'. This gives 20
        tokens instead of 21 and makes float(parts[-1]) fail.

        Workaround: when len(parts)==20 and the last token contains '*',
        compute Total from the individual energy components:
            Total = elect + gist + vdW + psol + asol + rec_d + r_hyd
        (charge, tStrain, and mStrain are intentionally excluded — confirmed
        by user cross-checking: for aspirin the components sum to -17.26.)

        Format signals:
          - 'close the file:' / 'open the file:' → skip next line (filename).
          - 'we reached the end of the' → stop parsing.
          - '  mol#           id_num' → header; start collecting pose lines.
        """
        best: Optional[float] = None
        try:
            lines = outdock_path.read_text().splitlines()
        except Exception:
            return None

        found_start = False
        skip_next = False
        for line in lines:
            if skip_next:
                skip_next = False
                continue
            if line.startswith("  we reached the end of the"):
                break
            if line.startswith(" close the file:") or line.startswith(" open the file:"):
                skip_next = True
                continue
            if line.startswith("  mol#           id_num"):
                found_start = True
                continue
            if not found_start:
                continue

            parts = line.split()
            total: Optional[float] = None

            if len(parts) == 21:
                # Normal case: Total printed correctly as last field.
                try:
                    total = float(parts[20])
                except ValueError:
                    pass

            elif len(parts) == 20 and "*" in parts[-1]:
                # Overflow bug: r_hyd and Total merged as e.g. '0.00**********'.
                # Recompute Total from energy components (charge/tStrain/mStrain excluded).
                try:
                    elect = float(parts[11])
                    gist  = float(parts[12])
                    vdw   = float(parts[13])
                    psol  = float(parts[14])
                    asol  = float(parts[15])
                    rec_d = float(parts[18])
                    r_hyd = float(parts[19].rstrip("*"))
                    total = elect + gist + vdw + psol + asol + rec_d + r_hyd
                except (ValueError, IndexError):
                    pass

            if total is not None:
                if best is None or total < best:
                    best = total

        return best

    # ------------------------------------------------------------------ public

    def score_smiles(self, smiles: str, name: str = "lig") -> float:
        """
        Dock *smiles* and return the DOCK3 total score (kcal/mol).

        Returns 0.0 if ligbuild cannot build the molecule or if docking fails to
        place any pose (the same convention used for un-dockable molecules in this
        project).
        """
        self.last_failure_reason = None
        td = self._make_workdir(smiles)
        try:
            # 1. Write SMILES file
            smi_file = td / f"{name}.smi"
            smi_file.write_text(f"{smiles} {name}\n")

            # 2. Build db2 bundle with ligbuild
            lb_out = td / "ligbuild_out"
            try:
                tgz_path = self._run_ligbuild(smi_file, lb_out)
            except Exception as e:
                self.last_failure_reason = self._classify_failure("ligbuild", str(e))
                print(
                    f"[dock3] ligbuild failed for '{name}' "
                    f"(workdir={td}, smiles={smiles}): {e}",
                    file=sys.stderr,
                )
                return 0.0

            # 3. Extract .db2
            try:
                db2_path = self._extract_db2(tgz_path, td)
            except Exception as e:
                self.last_failure_reason = self._classify_failure("db2_extract", str(e))
                print(f"[dock3] db2 extraction failed for '{name}' (workdir={td}): {e}", file=sys.stderr)
                return 0.0

            # 4. Patch INDOCK into run_dir; dock64 runs from run_dir so that
            #    '../dockfiles/...' resolves to td/dockfiles (private per call).
            run_dir = td / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            indock_path = self._build_indock(db2_path, run_dir)

            # 5. Run dock64
            try:
                self._run_dock64(run_dir, indock_path)
            except Exception as e:
                self.last_failure_reason = self._classify_failure("dock64", str(e))
                print(f"[dock3] dock64 failed for '{name}' (workdir={td}): {e}", file=sys.stderr)
                return 0.0

            # 6. Parse score from OUTDOCK (written into run_dir by dock64)
            outdock = run_dir / "OUTDOCK"
            score = self._parse_outdock_score(outdock)
            if score is None:
                self.last_failure_reason = self._classify_failure("outdock_parse", "no pose or score")
                return 0.0
            return score

        finally:
            shutil.rmtree(td, ignore_errors=True)
