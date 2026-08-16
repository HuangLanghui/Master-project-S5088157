set -e
python scripts/run_experiment.py --variants all --relight 0.0 --seeds 42 43 44
python scripts/run_experiment.py
python scripts/analyse.py > experiments/analysis.txt 2>&1
echo "ALL DONE"
