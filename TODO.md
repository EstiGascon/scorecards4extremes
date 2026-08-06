# TODO — scorecards4extremes

_Daily task list. Newest/active tasks at the top of "Pending". Move finished items to "Done" with a date._

---

## Pending

### Migrate to a single Python 3.13 venv (so `vtb` works in-process)
**Why:** currently the pipeline runs in `.venv` (Python 3.12) but `vtb` only exists in the
ECMWF `python3` module (3.13), so STVL obs retrieval runs via a subprocess fallback. A single
3.13 venv makes `import vtb` work in-process — cleaner, no indirection.
**Do it carefully (reversible, no downtime):**
- [ ] Wait until all running jobs finish (they depend on the current 3.12 `.venv`).
- [ ] Build a NEW venv alongside the old one — don't delete `.venv` yet:
      `module load python3 && python3 -m venv --system-site-packages .venv313 && .venv313/bin/pip install -r requirements.txt`
- [ ] Validate on `.venv313`: `import vtb` in-process; run a small `mars`+`stvl` config through
      `run.py` (retrieve → extract → score → plot); re-run one completed config and confirm scores match 3.12.
- [ ] Watch for eccodes/metview ABI issues from `--system-site-packages` vs pinned pip versions.
- [ ] Only once green: point `submit_job.sh` at `.venv313`, then retire `.venv`.
- [ ] Keep the subprocess fallback in `mars_retrieve.py` regardless (harmless; keeps the tool portable).

---

## Done

- (add finished tasks here with a date, e.g. `2026-07-27 — ...`)
