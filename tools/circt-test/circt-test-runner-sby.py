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

# Generate the SymbiYosys script.
script = """
  [tasks]
  {tasks}

{options}

  [engines]
  smtbmc z3

  [script]
  read -formal {source_path_name}
  prep -top {test}

  [files]
  {source_path}
""".format(tasks='\n  '.join(tasks),
           options=options,
           source_path_name=source_path.name,
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
