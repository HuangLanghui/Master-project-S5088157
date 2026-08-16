set -e
python scripts/run_experiment.py --modes none graded resample generative
python scripts/run_experiment.py --variants all --relight 0.0
python scripts/analyse.py > experiments/analysis.txt 2>&1
echo TR_DONE
