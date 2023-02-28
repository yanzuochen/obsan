#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

logdir=../results/bae
mkdir -p "$logdir"

mode=$1
eps=$2
nqueries=$3
seed=${4:-}

seed_suffix=${seed:+-}$seed
logfile="$logdir/$mode-${eps}-${nqueries}$seed_suffix.log"

seed_arg=${seed:+--random-seed}
# shellcheck disable=SC2086
./attack_obsan.py $mode --eps $eps --nqueries $nqueries $seed_arg $seed | tee "$logfile"
./stats.py "$logfile"
