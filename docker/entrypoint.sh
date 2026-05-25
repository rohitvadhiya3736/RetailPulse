#!/bin/bash
set -e
cd /app
export PYTHONPATH=/app
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export LOKY_MAX_CPU_COUNT=4

case "${1:-dashboard}" in
  api)
    exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    ;;
  dashboard)
    exec streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0
    ;;
  train)
    exec python scripts/train_all.py
    ;;
  pipeline)
    exec python scripts/train_all.py
    ;;
  *)
    exec "$@"
    ;;
esac
