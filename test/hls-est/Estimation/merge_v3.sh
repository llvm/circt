#!/usr/bin/env bash
# Merge results.csv BRAM predictions into v2.tsv -> v3.tsv.
# Only the last column ("BRAM (prediction, shim)") is updated, and only for
# benchmarks that ran OK (numeric total_bram). No rows are added or removed.
set -u
cd /mnt/d/HLS/circt/test/hls-est/Estimation || exit 1

CSV=results.csv
IN=v2.tsv
OUT=v3.tsv

# dir (results.csv) -> dataset (v2.tsv)
declare -A DS=(
  [chstone_xilinx]=chstone_xilinx
  [flowgnn_xilinx]=flowgnn_xilinx
  [forgebench]=forgebench_xilinx
  [machsuite]=machsuite_xilinx
  [polybench]=polybench_xilinx
  [spectorhls]=spectorhls_xilinx
)

# Build map: "<dataset>\t<benchmark>" -> total_bram, for OK rows only.
# OK rows have an empty error field, so simple comma splitting is safe.
MAP=$(mktemp)
while IFS=, read -r test _tf _tr status _ec bram _rest; do
  [[ "$status" == OK ]] || continue
  dir=${test%%/*}
  base=${test##*/}; stem=${base%.mlir}
  ds=${DS[$dir]:-}
  [[ -n "$ds" ]] || continue          # skip datasets not present in v2.tsv
  [[ -n "$bram" ]] || continue        # need a numeric value
  printf '%s\t%s\t%s\n' "$ds" "$stem" "$bram" >>"$MAP"
done < <(tail -n +2 "$CSV")

awk -F'\t' -v OFS='\t' '
  NR==FNR { key=$1 SUBSEP $2; bram[key]=$3; next }
  { sub(/\r$/, "") }                            # normalize CRLF -> LF
  FNR==1 { print; next }                        # header row unchanged
  {
    k=$1 SUBSEP $2                              # dataset, benchmark
    if (k in bram) { $NF=bram[k]; upd++ }
    print
  }
  END { print "updated " upd " rows" > "/dev/stderr" }
' "$MAP" "$IN" >"$OUT"

rm -f "$MAP"
echo "wrote $OUT ($(wc -l <"$OUT") lines; source $IN has $(wc -l <"$IN"))"
