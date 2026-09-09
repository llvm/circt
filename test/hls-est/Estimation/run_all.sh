#!/usr/bin/env bash
# Batch BRAM estimation over every *.mlir test under the Estimation test root.
# Reproducible driver — see CLAUDE.md for the spec.
set -u

TOOL=/mnt/d/HLS/circt/build-clion-v2/bin/hls-est
ROOT=/mnt/d/HLS/circt/test/hls-est/Estimation
LOGDIR="$ROOT/logs"
CSV="$ROOT/results.csv"
TIMEOUT=300

cd "$ROOT" || exit 1
mkdir -p "$LOGDIR"

# RFC 4180 quoting: wrap in double quotes iff the field contains a comma,
# quote, CR or LF; double any embedded quotes.
csv_field() {
  local s=$1
  if [[ "$s" == *[,\"$'\n'$'\r']* ]]; then
    s=${s//\"/\"\"}
    printf '"%s"' "$s"
  else
    printf '%s' "$s"
  fi
}

# Determine top function and which rule fired.
# Priority: hls.allocation attribute > name (top/kernel) > single func > last func.
determine_top() {
  local f=$1
  local names alloc name rule=""

  # Rule 0: a func.func carrying the hls.allocation attribute.
  local alloc_line
  alloc_line=$(grep -nE 'func\.func @[A-Za-z0-9_$.]+.*hls\.allocation' "$f" | head -1)
  if [[ -n "$alloc_line" ]]; then
    name=$(grep -oE 'func\.func @[A-Za-z0-9_$.]+' <<<"$alloc_line" | head -1 | sed 's/func\.func @//')
    printf '%s\thls.allocation' "$name"
    return
  fi

  # Collect func names in file order.
  mapfile -t names < <(grep -oE 'func\.func @[A-Za-z0-9_$.]+' "$f" | sed 's/func\.func @//')

  # Rule 1: name encodes top/kernel.
  for n in "${names[@]}"; do
    if [[ "$n" =~ ^_Z[0-9]+top || "$n" =~ ^_Z[0-9]+kernel || "$n" == "top" || "$n" == "kernel" ]]; then
      printf '%s\tname' "$n"
      return
    fi
  done

  # Rule 2: exactly one func.
  if [[ ${#names[@]} -eq 1 ]]; then
    printf '%s\tsingle' "${names[0]}"
    return
  fi

  # Rule 3: last func (cgeist emits callees before callers).
  if [[ ${#names[@]} -ge 1 ]]; then
    printf '%s\tlast' "${names[-1]}"
    return
  fi

  printf '\tnone'
}

printf 'test,top_func,top_rule,status,exit_code,total_bram,num_memrefs,ram_kinds,elapsed_s,error,log\n' >"$CSV"

n_total=0; n_ok=0; n_nototal=0; n_fail=0; n_timeout=0
declare -a bad_lines=()

while IFS= read -r f; do
  rel=${f#./}
  n_total=$((n_total+1))

  logname="logs/$(printf '%s' "$rel" | sed 's#/#__#g').log"
  logpath="$ROOT/$logname"

  # top function + rule
  IFS=$'\t' read -r top_func top_rule < <(determine_top "$rel")

  # Run the tool, combined stdout+stderr to the log; keep separate copies for parsing.
  outfile=$(mktemp); errfile=$(mktemp)
  start=$(date +%s.%N)
  timeout "$TIMEOUT" "$TOOL" "$rel" \
    --affine-loop-normalize \
    --memory-banking-bram \
    --convert-affine-to-loopschedule \
    --symbol-privatize=exclude="$top_func" \
    --symbol-dce \
    --bram-analysis >"$outfile" 2>"$errfile"
  rc=$?
  end=$(date +%s.%N)
  elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.1f", e-s}')

  # Combined log (stdout then stderr).
  { cat "$outfile"; cat "$errfile"; } >"$logpath"

  # First stderr line containing error:
  error=$(grep -m1 'error:' "$errfile" | head -1)
  error=${error%$'\r'}

  # total_bram + analysis-derived fields (analysis is printed before the IR dump).
  total_bram=""; num_memrefs=0; ram_kinds=""
  if grep -q '^Total BRAM:' "$outfile"; then
    total_bram=$(grep -m1 '^Total BRAM:' "$outfile" | sed -E 's/^Total BRAM:[[:space:]]*//' | tr -d '\r')
    analysis=$(awk '/^Total BRAM:/{exit} {print}' "$outfile")
    num_memrefs=$(grep -c '^  kind=' <<<"$analysis")
    ram_kinds=$(grep -oE 'kind=[A-Za-z0-9_]+' <<<"$analysis" | sed 's/kind=//' | sort -u | paste -sd';' -)
  fi

  # Status.
  if [[ $rc -eq 124 ]]; then
    status=TIMEOUT; n_timeout=$((n_timeout+1))
  elif [[ $rc -ne 0 ]]; then
    status=FAIL; n_fail=$((n_fail+1))
  elif [[ -n "$total_bram" ]]; then
    status=OK; n_ok=$((n_ok+1))
  else
    status=NO_TOTAL; n_nototal=$((n_nototal+1))
  fi

  if [[ "$status" != OK ]]; then
    bad_lines+=("$status  $rel  ${error:-<no error: line>}")
  fi

  {
    csv_field "$rel";       printf ','
    csv_field "$top_func";  printf ','
    csv_field "$top_rule";  printf ','
    csv_field "$status";    printf ','
    csv_field "$rc";        printf ','
    csv_field "$total_bram";printf ','
    csv_field "$num_memrefs";printf ','
    csv_field "$ram_kinds"; printf ','
    csv_field "$elapsed";   printf ','
    csv_field "$error";     printf ','
    csv_field "$logname";   printf '\n'
  } >>"$CSV"

  rm -f "$outfile" "$errfile"
done < <(find . -name '*.mlir' | sort)

echo "==================== SUMMARY ===================="
echo "total tests: $n_total"
echo "  OK:       $n_ok"
echo "  NO_TOTAL: $n_nototal"
echo "  FAIL:     $n_fail"
echo "  TIMEOUT:  $n_timeout"
if [[ ${#bad_lines[@]} -gt 0 ]]; then
  echo "---- non-OK tests ----"
  for l in "${bad_lines[@]}"; do echo "  $l"; done
fi
echo "CSV: $CSV"
