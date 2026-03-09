#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("verilog")
parser.add_argument("-t", "--test", required=True)
parser.add_argument("-d", "--directory", required=True)
args = parser.parse_args()

directory = Path(args.directory)
source_path = Path(args.verilog)
build_path = directory / "build"
testbench_path = directory / "testbench.sv"

# Generate the Verilog top-level that interacts with the test module.
testbench = f"""
module circt_test;
  bit clock = 0, init = 1, done, success;
  {args.test} dut(.clock, .init, .done, .success);
  initial begin
    #1 clock = 1;
    #1 clock = 0; init = 0;
    #1 clock = 1;
    while (!done) begin
      #1 clock = 0;
      #1 clock = 1;
    end
    if (success) $finish; else $fatal;
  end
endmodule
"""
with open(testbench_path, "w") as f:
  f.write(testbench)

# Verilate the test.
cmd = [
    "verilator",
    "--cc",
    "--exe",
    "--main",
    "--timing",
    testbench_path,
    "-Mdir",
    build_path,
    "--top-module",
    "circt_test",
]
if source_path.is_dir():
  cmd += ["-F", source_path / "filelist.f"]
else:
  cmd += [source_path]
sys.stderr.write("# " + shlex.join(str(c) for c in cmd) + "\n")
sys.stderr.flush()
if subprocess.call(cmd):
  sys.exit(1)

# Compile the test.
cmd = ["make", "-j", "-C", build_path, "-f", "Vcirct_test.mk"]
if arg := os.environ.get("CC"):
  cmd += ["CC=" + arg]
if arg := os.environ.get("CXX"):
  cmd += ["CXX=" + arg]
sys.stderr.write("# " + shlex.join(str(c) for c in cmd) + "\n")
sys.stderr.flush()
if subprocess.call(cmd):
  sys.exit(1)

# Run the test.
cmd = [build_path / "Vcirct_test"]
sys.stderr.write("# " + shlex.join(str(c) for c in cmd) + "\n")
sys.stderr.flush()
result = subprocess.call(cmd)
sys.exit(result)
