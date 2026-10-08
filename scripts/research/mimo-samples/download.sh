#!/usr/bin/env bash
# Download the real HuggingFace parquet shards of XiaomiMiMo/MiMo-V2.6-RL-oss
# into this directory, then validate them with:
#
#   cargo run -p dsec-agentenv --features parquet --example real_dataset <this-dir>
#
# ~20 MB total. The general split lives at general/train.parquet upstream and
# is saved here as general_train.parquet (the name the example probes).
set -euo pipefail

BASE_URL="https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/resolve/main"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

for f in code cyber music webdev; do
    echo "-> $f.parquet"
    curl -fSL "${BASE_URL}/${f}.parquet" -o "${f}.parquet"
done

echo "-> general_train.parquet (from general/train.parquet)"
curl -fSL "${BASE_URL}/general/train.parquet" -o general_train.parquet

echo
ls -la
echo
echo "done. validate with:"
echo "  cargo run -p dsec-agentenv --features parquet --example real_dataset $DIR"
