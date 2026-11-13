#!/bin/bash
# Run all systems with NW workers x NT threads each, then collect the summary.
# usage: ./run_all.sh [NW] [NT]
cd "$(dirname "$0")"
NW=${1:-4}
NT=${2:-4}
mkdir -p results_minao logs
mapfile -t FILES < <(ls ../xyzfiles/*/*.xyz | awk -F/ '{print $NF, $0}' | sort -u -k1,1 | awk '{print $2}')
echo "${#FILES[@]} unique systems, $NW workers x $NT threads"
for ((w=0; w<NW; w++)); do
  sub=()
  for ((i=w; i<${#FILES[@]}; i+=NW)); do sub+=("${FILES[$i]}"); done
  OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT \
    nohup python run_scf.py --outdir results_minao "${sub[@]}" > logs/worker$w.out 2>&1 &
done
wait
python collect_summary.py results_minao
