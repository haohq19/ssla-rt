#!/usr/bin/env bash
# Scaling sweep: throughput as a function of thread count, both modes.
# Run from deploy/.

set -euo pipefail

WEIGHTS=${WEIGHTS:-/tmp/ssla_s_random}
EVENTS_N=${EVENTS_N:-1100000}
WARMUP=${WARMUP:-10000}
H=${H:-240}
W=${W:-304}
THREADS=${THREADS:-"1 2 4 8 16 32 64"}

BIN=$(dirname "$0")/../build/multicore_bench

run_one() {
  local mode=$1
  local n=$2
  "$BIN" --weights "$WEIGHTS" --threads "$n" --warmup "$WARMUP" \
         --random-n "$EVENTS_N" --random-hw "$H" "$W" --mode "$mode" \
    | awk '
        /^  throughput/      { thru=$3 }
        /^  ns\/event/        { ns=$3 }
        /^  wall \(slowest\)/ { wall=$4 }
        END { printf "%-12s\t%.2f\t%.1f\t%.2f\n", mode, thru, ns, wall }
      ' mode=$mode
}

printf "mode       \tThr(M/s)\tns/ev\twall_ms\tthreads=%s\n" "$THREADS"
for n in $THREADS; do
  printf "  N=%-2d  " "$n"
  run_one strip     "$n"
done
echo
for n in $THREADS; do
  printf "  N=%-2d  " "$n"
  run_one replicate "$n"
done
