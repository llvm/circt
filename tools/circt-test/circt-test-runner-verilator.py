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
parser.add_argument(
    "--trace",
    type=str,
    default="",
    help="Enable waveform tracing with format preferences (e.g., 'fst,vcd')")
args = parser.parse_args()


def select_trace_format(preferences):
  if not preferences:
    return None, None, None
  supported = {
      "vcd": ("vcd", ".vcd", "--trace-vcd"),
      "fst": ("fst", ".fst", "--trace-fst"),
  }
  prefs = [p.strip().lower() for p in preferences.split(",")]
  for pref in prefs:
    if pref in supported:
      return supported[pref]
  return supported["vcd"]


directory = Path(args.directory)
source_path = Path(args.verilog)
build_path = directory / "build"
testbench_path = directory / "testbench.sv"

# Select trace format (returns Nones if args.trace is empty)
trace_format, trace_ext, trace_option = select_trace_format(args.trace)

# Only set trace_file if tracing is enabled
if trace_format:
  trace_file = directory / f"{args.test}{trace_ext}"
else:
  trace_file = ""

# Generate the Verilog top-level that interacts with the test module.
testbench = f"""
module circt_test;
  bit clock = 0, init = 1, done, success;
  {args.test} dut(.clock, .init, .done, .success);
  initial begin
"""
if trace_format:
  testbench += f"""
    $dumpfile("{trace_file}");
    $dumpvars(0, circt_test);
  """
testbench += f"""
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

# Add trace support if tracing is enabled
if trace_format:
  cmd += [trace_option]

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
