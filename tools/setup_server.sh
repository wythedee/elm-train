#!/usr/bin/env bash
# Prepare isolated environments and public assets; does not start model jobs.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: ELM_WORK=/your/authorized/workspace bash tools/setup_server.sh'
  echo 'Optional: ELM_DOWNLOAD_WORKERS=4 ELM_WAIT_DATA_SECONDS=21600 ELM_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple'
  exit 0
fi
[[ "$(uname -s)" == Linux ]] || { echo 'Run on the authorized Linux training server.' >&2; exit 1; }
: "${ELM_WORK:?Set ELM_WORK to an authorized private workspace on the training server}"
ELM_PYTHON_TRAIN="${ELM_PYTHON_TRAIN:-python3.10}"
ELM_PYTHON_AR="${ELM_PYTHON_AR:-python3.11}"
command -v "$ELM_PYTHON_TRAIN" >/dev/null || { echo 'Python 3.10 is required; set ELM_PYTHON_TRAIN to its executable.' >&2; exit 1; }
command -v "$ELM_PYTHON_AR" >/dev/null || { echo 'Python 3.11 is required; set ELM_PYTHON_AR to its executable.' >&2; exit 1; }
command -v flock >/dev/null || { echo 'Linux flock is required to prevent concurrent setup.' >&2; exit 1; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ELM_WORK=$("$ELM_PYTHON_AR" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$ELM_WORK")
[[ "$ELM_WORK" != / && "$ELM_WORK" != "$REPO" ]] || { echo 'Choose a dedicated workspace.' >&2; exit 1; }
umask 077
mkdir -p "$ELM_WORK/setup" "$ELM_WORK/envs" "$ELM_WORK/tmp" "$ELM_WORK/cache/pip"
"$ELM_PYTHON_AR" -c 'import shutil,sys; free=shutil.disk_usage(sys.argv[1]).free; print("Available GiB:", round(free/2**30,1)); sys.exit(0 if free >= 120*2**30 else "Need at least 120 GiB free for initial assets/environments; 200+ GiB recommended")' "$ELM_WORK"
exec 9>"$ELM_WORK/setup/setup.lock"
flock -n 9 || { echo 'Another setup owns this workspace.' >&2; exit 1; }
export TMPDIR="$ELM_WORK/tmp"
export PIP_CACHE_DIR="$ELM_WORK/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="${ELM_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
export PYTHONDONTWRITEBYTECODE=1

make_env() {
  local name="$1" interpreter="$2" envdir="$ELM_WORK/envs/$1"
  if [[ -e "$envdir" ]]; then
    [[ -f "$envdir/.elm_setup_owned" && -x "$envdir/bin/python" ]] || {
      echo "Refusing to modify an existing unmarked environment: $envdir" >&2; return 1;
    }
  else
    "$interpreter" -m venv "$envdir"
    printf '%s\n' 'Created by elm-train tools/setup_server.sh' > "$envdir/.elm_setup_owned"
  fi
}

install_train() {
  make_env train "$ELM_PYTHON_TRAIN"
  local py="$ELM_WORK/envs/train/bin/python"
  "$py" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  "$py" -m pip install -r "$REPO/requirements-train.txt"
  "$py" -m pip check
}
install_ar() {
  make_env ar-v1 "$ELM_PYTHON_AR"
  local py="$ELM_WORK/envs/ar-v1/bin/python"
  "$py" -m pip install --extra-index-url https://download.pytorch.org/whl/cu129 -r "$REPO/requirements-ar-v1.txt"
  "$py" -m pip check
}
install_download() {
  make_env download "$ELM_PYTHON_AR"
  "$ELM_WORK/envs/download/bin/python" -m pip install -r "$REPO/requirements-download.txt"
  "$ELM_WORK/envs/download/bin/python" -m pip check
}

# Independent environments install concurrently. Each job keeps its own log.
install_train >"$ELM_WORK/setup/train.log" 2>&1 & train_pid=$!
install_ar >"$ELM_WORK/setup/ar-v1.log" 2>&1 & ar_pid=$!
install_download >"$ELM_WORK/setup/download.log" 2>&1 & download_pid=$!
failed=0
wait "$download_pid" || failed=1
if [[ "$failed" == 0 ]]; then
  "$ELM_WORK/envs/download/bin/python" -B "$REPO/tools/prepare_server_assets.py" \
    --work "$ELM_WORK" --workers "${ELM_DOWNLOAD_WORKERS:-4}" \
    --wait-data-seconds "${ELM_WAIT_DATA_SECONDS:-21600}" \
    >"$ELM_WORK/setup/assets.log" 2>&1 & assets_pid=$!
else
  assets_pid=''
fi
wait "$train_pid" || failed=1
wait "$ar_pid" || failed=1
if [[ -n "$assets_pid" ]]; then wait "$assets_pid" || failed=1; fi
if [[ "$failed" != 0 ]]; then
  echo "Setup stopped. Inspect $ELM_WORK/setup/{train,ar-v1,download,assets}.log; preserve completed downloads." >&2
  exit 1
fi
echo "Environments and assets prepared: $ELM_WORK/setup/assets_ready.json"
echo 'Next: generate a run configuration with tools/make_server_config.py; see README.'
