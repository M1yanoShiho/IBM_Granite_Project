#!/bin/bash
#SBATCH --job-name=exp05-g1-data
#SBATCH --account=coms039904
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/user/work/fl25387/experiment05_goal1/logs/%x-%j.out

set -euo pipefail

root=/user/work/fl25387/experiment05_goal1
cache="$root/source_cache"
mkdir -p "$cache" "$root/logs"

download() {
  local url="$1"
  local destination="$2"
  curl --fail --location --retry 12 --retry-delay 10 --continue-at - \
    --output "$destination" "$url"
}

download \
  https://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json \
  "$cache/kilt_knowledgesource.json"
download \
  https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz \
  "$cache/psgs_w100.tsv.gz"
download \
  https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz \
  "$cache/triviaqa-unfiltered.tar.gz"

sha256sum \
  "$cache/kilt_knowledgesource.json" \
  "$cache/psgs_w100.tsv.gz" \
  "$cache/triviaqa-unfiltered.tar.gz" \
  > "$cache/full_corpora.sha256"
