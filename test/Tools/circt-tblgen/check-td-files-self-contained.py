# !/usr/bin/env python3
# Smoke test: verify the TableGen files are self-contained.
#
# Recursively walks selected directories and runs circt-tblgen --print-records
# on every .td file found. Each file must parse without errors when processed
# in isolation, which means no missing includes and no unresolved references.
#
# File checks are executed concurrently since each invocation is independent.
#
# Usage:
#   python3 check-td-files-self-contained.py --circt_root <path/to/circt>
#     --circt_tblgen <path/to/circt-tblgen>               # optional
#     --mlir_include <path/to/llvm-project/mlir/include>  # optional
#     --jobs <number of jobs>                             # optional

import argparse
import concurrent.futures
import os
import subprocess
import sys


def find_mlir_include(circt_root):
  """Locate MLIR TableGen headers under the CIRCT-internal llvm/ submodule."""
  candidate = os.path.join(circt_root, "llvm", "mlir", "include")
  if os.path.isfile(os.path.join(candidate, "mlir", "IR", "OpBase.td")):
    return candidate
  return None


def collect_td_files(root_dirs):
  """Recursively collect all .td files under the given root directories."""
  files = []
  for root in root_dirs:
    for dirpath, dirnames, filenames in os.walk(root):
      dirnames[:] = [
          d for d in dirnames if not d.startswith(".") and d != "llvm"
      ]
      for f in sorted(filenames):
        if f.endswith(".td"):
          files.append(os.path.join(dirpath, f))
  return files


def check_one(circt_tblgen, include_flags, circt_root, full_path):
  """Run circt-tblgen on a single .td file, return (rel_path, ok, stderr)."""
  rel_path = os.path.relpath(full_path, circt_root)
  cmd = [circt_tblgen, "--print-records"] + include_flags + [full_path]
  result = subprocess.run(cmd, capture_output=True, text=True)
  return rel_path, result.returncode == 0, result.stderr


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--circt_tblgen",
                      default="circt-tblgen",
                      help="Path to circt-tblgen binary")
  parser.add_argument("--circt_root",
                      required=True,
                      help="CIRCT repository root (e.g. /path/to/circt)")
  parser.add_argument("--mlir_include",
                      nargs="?",
                      help="MLIR TableGen include directory. "
                      "Defaults to circt_root/llvm/mlir/include")
  parser.add_argument("--jobs",
                      type=int,
                      default=0,
                      help="Number of parallel workers (0 = cpu count)")
  args = parser.parse_args()

  circt_root = os.path.normpath(args.circt_root)
  circt_include = os.path.join(circt_root, "include")

  if not os.path.isdir(circt_include):
    print(f"error: CIRCT include directory not found: {circt_include}",
          file=sys.stderr)
    sys.exit(1)

  if args.mlir_include:
    mlir_include = os.path.normpath(args.mlir_include)
  else:
    mlir_include = find_mlir_include(circt_root)
    if not mlir_include:
      print(
          "error: cannot locate MLIR TableGen includes "
          "(mlir/IR/OpBase.td). "
          "Expected at circt/llvm/mlir/include. "
          "Set up a symlink: ln -s /path/to/llvm-project circt/llvm",
          file=sys.stderr)
      sys.exit(1)

    td_files = collect_td_files([
        circt_include,
        os.path.join(circt_root, "lib", "Dialect"),
        os.path.join(circt_root, "lib", "Bindings", "Python", "dialects"),
    ])

  include_flags = ["-I", circt_include, "-I", mlir_include]
  max_workers = args.jobs if args.jobs > 0 else os.cpu_count()

  failed = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
    futs = [
        pool.submit(check_one, args.circt_tblgen, include_flags, circt_root, p)
        for p in td_files
    ]
    for f in concurrent.futures.as_completed(futs):
      rel_path, ok, stderr = f.result()
      if not ok:
        print(f"FAIL: {rel_path}", file=sys.stderr)
        print(stderr, file=sys.stderr)
        failed.append(rel_path)
      else:
        print(f"PASS: {rel_path}")

  if failed:
    print(f"\n{len(failed)} file(s) failed self-contained check:",
          file=sys.stderr)
    for f in failed:
      print(f"  {f}", file=sys.stderr)
    sys.exit(1)
  print("\nAll files passed.")


if __name__ == "__main__":
  main()
