#!/usr/bin/env bash
set -euo pipefail

BIN="./token_lookup_cpu"   # adjust if binary is elsewhere
REPEATS=7
OUT="cpu_sweep.csv"
SIZES=(1000000 5000000 20000000 50000000)

echo "n,repeats,median_ms,throughput_mops,effective_bw_gbps,checksum" > "$OUT"

for n in "${SIZES[@]}"; do
  echo "Running n=$n ..."
  # program writes CSV of a single-row summary when --csv is used
  TMP="tmp_${n}.csv"
  "$BIN" "$n" --repeats "$REPEATS" --csv "$TMP"
  # skip header, append line to OUT
  tail -n +2 "$TMP" >> "$OUT"
  rm -f "$TMP"
done

echo "Sweep complete — results in $OUT"
cat "$OUT"