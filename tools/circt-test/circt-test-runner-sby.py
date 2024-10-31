#!/usr/bin/env python3
from pathlib import Path
import argparse
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("verilog")
parser.add_argument("-t", "--test")
parser.add_argument("-d", "--directory")
args = parser.parse_args()

directory = Path(args.directory)
source_path = Path(args.verilog)
script_path = directory / "script.sby"

# Generate the SymbiYosys script.
script = f"""
  [tasks]
  cover
  bmc
  induction

  [options]
  cover:
  mode cover
  --
  bmc:
  mode bmc
  --
  induction:
  mode prove
  --

  [engines]
  smtbmc z3

  [script]
  read -formal {source_path.name}
  prep -top {args.test}

  [files]
  {source_path}
"""
with open(script_path, "w") as f:
  for line in script.strip().splitlines():
    f.write(line.strip() + "\n")

# Run SymbiYosys.
cmd = ["sby", "-f", script_path]
sys.stderr.write("# " + shlex.join(str(c) for c in cmd) + "\n")
sys.stderr.flush()
result = subprocess.call(cmd)
sys.exit(result)
