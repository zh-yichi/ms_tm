#!/bin/bash
# Re-run the (system, method) pairs listed in <results>/problems.txt with more
# SCF iterations and more stability-following rounds.
#
#   ./rerun_problems.sh [RESULTS_DIR] [RERUN_DIR]
#
# Output goes to RERUN_DIR (default results_rerun) so that the original logs
# and chkfiles are not overwritten.  Copy the new <stem>/* over the old ones
# once you are happy with them.  Thread count comes from OMP_NUM_THREADS.
cd "$(dirname "$0")"
RES=${1:-results}
OUT=${2:-results_rerun}
XYZDIR=../xyzfiles

MAX_CYCLE=1000        # DIIS iterations
NEWTON_MAX_CYCLE=200  # Newton iterations in the fallback
MAX_STAB=15           # stability-following rounds
STAB_TOL=1e-4         # |Hessian eigenvalue| below this is a zero mode, not an instability

[ -f "$RES/problems.txt" ] || python collect_summary.py "$RES" > /dev/null
mkdir -p "$OUT"

# problems.txt lines look like:  <stem>  <method>  charge=.. mult=..  converged=..  stable=..  <note>
grep -v '^#' "$RES/problems.txt" | awk '{print $1, $2}' | sort -u \
| awk '{m[$1] = (m[$1] ? m[$1]","$2 : $2)} END {for (s in m) print s, m[s]}' \
| while read -r stem methods; do
    xyz=$(ls "$XYZDIR"/*/"$stem".xyz | head -1)
    echo "== $stem  methods=$methods"
    python run_scf.py --outdir "$OUT" --force \
        --methods "$methods" \
        --max-cycle $MAX_CYCLE --newton-max-cycle $NEWTON_MAX_CYCLE --max-stab $MAX_STAB --stab-tol $STAB_TOL \
        "$xyz"
done

python collect_summary.py "$OUT"
