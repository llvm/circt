#!/usr/bin/env python3
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


class Config:
  """Configuration for the self-contained check."""

  def __init__(self, args):
    self.circt_tblgen = args.circt_tblgen
    self.circt_root = os.path.normpath(args.circt_root)
    # If not using the circt-internal llvm/ submodule (e.g. a separate
    # llvm-project checkout), use --mlir_include to point to the MLIR
    # include directory; otherwise auto-detected from circt_root/llvm/.
    self.mlir_include = self._resolve_mlir_include(args.mlir_include)
    self.jobs = self._clamp_jobs(args.jobs)

  def _resolve_mlir_include(self, user_path):
    if user_path:
      return os.path.normpath(user_path)
    return os.path.join(self.circt_root, "llvm", "mlir", "include")

  @staticmethod
  def _clamp_jobs(jobs):
    cpu_count = os.cpu_count() or 1
    if jobs <= 0:
      return cpu_count
    return min(jobs, cpu_count)

  def check_valid(self):
    try:
      subprocess.run([self.circt_tblgen, "--version"],
                     capture_output=True,
                     check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
      raise ValueError(
          f"circt-tblgen not found or not executable: {self.circt_tblgen}")

    dialect_root = os.path.join(self.circt_root, "include", "circt", "Dialect")
    if not os.path.isdir(dialect_root):
      raise ValueError(
          f"CIRCT include/circt/Dialect/ directory not found in {self.circt_root}"
      )

    opbase = os.path.join(self.mlir_include, "mlir", "IR", "OpBase.td")
    if not self.mlir_include or not os.path.isfile(opbase):
      raise ValueError(
          f"cannot locate MLIR TableGen includes (e.g., '{opbase}')")

  @property
  def circt_include(self):
    return os.path.join(self.circt_root, "include")

  @property
  def include_flags(self):
    return ["-I", self.circt_include, "-I", self.mlir_include]


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
                      help="MLIR TableGen include directory. "
                      "Defaults to circt_root/llvm/mlir/include")
  parser.add_argument("--jobs",
                      type=int,
                      default=0,
                      help="Number of parallel workers (0 = cpu count)")
  config = Config(parser.parse_args())
  config.check_valid()

  td_files = collect_td_files([
      config.circt_include,
      os.path.join(config.circt_root, "lib", "Dialect"),
      os.path.join(config.circt_root, "lib", "Bindings", "Python", "dialects"),
  ])

  failed = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=config.jobs) as pool:
    futs = [
        pool.submit(check_one, config.circt_tblgen, config.include_flags,
                    config.circt_root, p) for p in td_files
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
  try:
    main()
  except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
