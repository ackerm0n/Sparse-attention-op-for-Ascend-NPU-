#!/bin/bash
#
# Copyright 2026 TriangleMix contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir"

: "${VLLM_ASCEND_SRC:?Set VLLM_ASCEND_SRC to a vLLM-Ascend source checkout}"

python_bin=${PYTHON:-python}
strip_bin=${STRIP:-strip}
command -v "$python_bin" >/dev/null
command -v "$strip_bin" >/dev/null

build_root=$(mktemp -d "${TMPDIR:-/tmp}/trianglemix-adapter.XXXXXX")
cleanup() {
    rm -rf -- "$build_root"
}
trap cleanup EXIT

# Stabilise compiler/tool output that is otherwise derived from process state.
export LC_ALL=C
export TZ=UTC
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-0}

"$python_bin" setup.py build_ext \
    --inplace \
    --force \
    --build-temp "$build_root/temp" \
    --build-lib "$build_root/lib"

extension_suffix=$(
    "$python_bin" -c \
        'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'
)
adapter="$script_dir/triangle_paged_attention_torch${extension_suffix}"
if [[ ! -f "$adapter" ]]; then
    echo "adapter build did not produce $adapter" >&2
    exit 1
fi

"$strip_bin" --strip-unneeded "$adapter"
"$python_bin" "$script_dir/../tools/check_release_artifact.py" \
    --adapter "$adapter"

printf 'release adapter: %s\n' "$adapter"
