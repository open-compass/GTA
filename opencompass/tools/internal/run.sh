CONFIG=$1
WORK_DIR=$2
PY_ARGS=${@:3}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python run.py ${CONFIG} -w ${WORK_DIR} ${PY_ARGS}
