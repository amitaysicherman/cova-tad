#!/bin/bash
#SBATCH --job-name=covatad_iclr27_audit
#SBATCH --partition=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/amitay.s/covatad_iclr2027_cpu_audit/logs/%x_%A_%a.out
#SBATCH --error=/home/amitay.s/covatad_iclr2027_cpu_audit/logs/%x_%A_%a.err

set -euo pipefail

project_dir=/home/amitay.s/covatad_iclr2027_cpu_audit
python_bin=/home/amitay.s/miniconda3/envs/zik/bin/python
manifest="$project_dir/scripts/iclr2027_cpu_manifest.csv"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")
if [[ -z "$line" ]]; then
  echo "No audit task for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=, read -r mode dataset <<< "$line"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
cd "$project_dir"
output="$project_dir/results/iclr2027_cpu_audit/${mode}_${dataset}.csv"
"$python_bin" -u scripts/run_iclr2027_cpu_audit.py \
  --mode "$mode" \
  --dataset "$dataset" \
  --data-dir "$project_dir/data" \
  --output "$output" \
  --threads "$SLURM_CPUS_PER_TASK"
