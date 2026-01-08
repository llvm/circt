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
parser.add_argument("-m", "--mode")
parser.add_argument("-k", "--depth")
args = parser.parse_args()

directory = Path(args.directory)
source_path = Path(args.verilog)
script_path = directory / "script.sby"

if args.mode is not None:
  tasks = args.mode.split(",")
else:
  tasks = ["cover", "bmc", "induction"]

depth = f"depth {args.depth}" if args.depth is not None else ""

options = """
  [options]
"""
for task in tasks:
  mode = {"induction": "prove"}.get(task, task)
  options += f"""
  {task}:
  mode {mode}
  {depth}
  --
"""

# Collect the source files to be read in.
source_files = []
if source_path.is_dir():
  with open(source_path / "filelist.f", "r") as filelist:
    for line in filelist:
      # Ignore empty lines in the file list.
      line = line.strip()
      if not line:
        continue

      # Ignore non-Verilog files. This is an ugly hack, but sometimes C/C++/Rust
      # files get mixed into the filelist.
      p = Path(line)
      if p.suffix not in (".v", ".sv", ".vh", ".svh"):
        continue

      if not p.is_absolute():
        p = source_path / p
      source_files.append(p)
else:
  source_files.append(source_path)

read_commands = ""
for sf in source_files:
  read_commands += f"read -formal {sf.absolute()}\n"

# Generate the SymbiYosys script.
script = """
[tasks]
{tasks}

{options}

[engines]
smtbmc z3

[script]
{read_commands}
prep -top {test}
""".format(tasks='\n  '.join(tasks),
           options=options,
           read_commands=read_commands,
           test=args.test,
           source_path=source_path)
with open(script_path, "w") as f:
  for line in script.strip().splitlines():
    f.write(line.strip() + "\n")

# Run SymbiYosys.
cmd = ["sby", "-f", script_path]
sys.stderr.write("# " + shlex.join(str(c) for c in cmd) + "\n")
sys.stderr.flush()
result = subprocess.call(cmd)
sys.exit(result)
