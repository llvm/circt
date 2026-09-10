#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("mlir")
parser.add_argument("-t", "--test", required=True)
parser.add_argument("-d", "--directory", required=True)
parser.add_argument("-m", "--mode")
parser.add_argument("-k", "--depth", type=int)
args = parser.parse_args()

# Use circt-opt to lower any `verif.formal` ops to `hw.module`s. Once circt-bmc
# natively supports `verif.formal`, we can directly run it on the input MLIR.
lowered_mlir = Path(args.directory) / "lowered.mlir"
cmd = ["circt-opt", args.mlir, "-o", lowered_mlir, "--verif-lower-tests"]
sys.stderr.write("# " + shlex.join(str(c) for c in cmd) + "\n")
sys.stderr.flush()
result = subprocess.call(cmd)
if result != 0:
  sys.exit(result)

# Run circt-bmc. We currently have to capture the output and look for a specific
# string to know if the verification passed or failed. Once the tool properly
# exits on test failure, we can skip this.
cmd = ["circt-bmc", lowered_mlir]
cmd += ["-b", str(args.depth or 20)]
cmd += ["--module", args.test]
cmd += ["--shared-libs", os.environ.get("Z3LIB", "/usr/lib/libz3.so")]

sys.stderr.write("# " + shlex.join(str(c) for c in cmd) + "\n")
sys.stderr.flush()
with open(Path(args.directory) / "output.log", "w") as output:
  result = subprocess.call(cmd, stdout=output, stderr=output)
with open(Path(args.directory) / "output.log", "r") as output:
  output_str = output.read()
sys.stderr.write(output_str)
if result != 0:
  sys.exit(result)
sys.exit(0 if "Bound reached with no violations" in output_str else 1)
