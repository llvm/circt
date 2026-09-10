# hls-est batch estimation

## Task

Enumerate every test case under the Estimation test directory, run the BRAM
estimation pipeline on each one, and dump the results into a CSV.

## Paths

- Tool binary: `/mnt/d/HLS/circt/build-clion-v2/bin/hls-est`
- Test root: `/mnt/d/HLS/circt/test/hls-est/Estimation/`
- Test cases: every `*.mlir` file under the test root, **recursively**
  (there are subdirectories, e.g. `forgebench/activation_op1.mlir`).
- Output: `/mnt/d/HLS/circt/test/hls-est/Estimation/results.csv`
- Raw logs: `/mnt/d/HLS/circt/test/hls-est/Estimation/logs/<test>.log`
  (mirror the relative path of the test, replacing `/` with `__`)

## Command to run per test

```
/mnt/d/HLS/circt/build-clion-v2/bin/hls-est <test>.mlir \
  --affine-loop-normalize \
  --memory-banking-bram \
  --convert-affine-to-loopschedule \
  --symbol-privatize=exclude=<top-func> \
  --symbol-dce \
  --bram-analysis
```

`<test>` is the path of the `.mlir` file. `<top-func>` is the name of the
top-level function **without** the leading `@`.

## Determining `<top-func>`

The IR is Polygeist/cgeist output, so function names are Itanium-mangled.
Pick the top function with this priority order, and record which rule fired:

1. A function whose mangled name encodes `top` or `kernel`, i.e. matches
   `@_Z[0-9]+top` or `@_Z[0-9]+kernel`, or is literally `@top` / `@kernel`.
2If the file has exactly one `func.func`, that one.
3Otherwise the **last** `func.func` in the file (cgeist emits callees
   before callers), and flag the row as `heuristic=last`.

Extract names with `grep -oE 'func\.func @[A-Za-z0-9_$.]+' <file>` and strip
the `func.func @` prefix.

## Running

- Do this with a single bash script (write it to
  `/mnt/d/HLS/circt/test/hls-est/Estimation/run_all.sh`) rather than
  invoking the tool by hand per test, so the run is reproducible.
- Run each test with a timeout of 300 s (`timeout 300 ...`). Treat a timeout
  as a failure with status `TIMEOUT`.
- Capture stdout and stderr together into the per-test log.
- Never stop on the first failure; every test must produce a CSV row.
- Do not modify any test file or the tool.

## Parsing the output

The tool prints one block per memref followed by a summary line:

```
memref.global @ loc("...":3:3)
  kind=RAM_1P  bram=49
    50176x16b -> RAM_1P (x32/block) = 49 BRAM_18K
...
Total BRAM: 196
```

From each run, extract:

- `total_bram`: the integer after `Total BRAM:` (empty if absent)
- `error`: first line of stderr containing `error:` (empty if none)

## CSV format

Header row, then one row per test, sorted by `test`. Fields containing commas,
quotes, or newlines must be quoted per RFC 4180.

```
test,top_func,top_rule,status,exit_code,total_bram,num_memrefs,ram_kinds,elapsed_s,error,log
```

- `test`: path relative to the test root (e.g. `forgebench/activation_op1.mlir`)
- `top_rule`: `hls.allocation` | `name` | `single` | `last`
- `status`: `OK` if exit code 0 and `Total BRAM:` was found;
  `NO_TOTAL` if exit 0 but no summary line; `FAIL` on non-zero exit;
  `TIMEOUT` on timeout
- `elapsed_s`: wall-clock seconds, one decimal place
- `log`: path to the per-test log

## When done

Print a short summary to the console: number of tests, count per status,
and the list of any `FAIL` / `TIMEOUT` / `NO_TOTAL` tests with the first
error line for each. Do not paste the full CSV into the chat.