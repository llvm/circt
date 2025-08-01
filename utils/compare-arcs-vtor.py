#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
"""
A utility that takes a piece of hardware described in the core dialects and
runs it through arcilator and verilator in lockstep with randomized inputs,
flagging any discrepancies between the two.
"""


def main():
  # Parse the command line arguments.
  parser = argparse.ArgumentParser(
      description="Run arcilator and verilator in lockstep and compare.")
  parser.add_argument("input", type=Path, help="Input file to process")
  parser.add_argument("-d",
                      "--dir",
                      type=Path,
                      help="Directory for temporary files")
  parser.add_argument("--arcilator-bin",
                      type=Path,
                      help="Path to arcilator binary")
  parser.add_argument("--verilator-bin",
                      type=Path,
                      help="Path to verilator binary")
  parser.add_argument("--llc-bin", type=Path, help="Path to llc binary")
  parser.add_argument("--firtool-bin", type=Path, help="Path to firtool binary")
  parser.add_argument("--path",
                      type=Path,
                      help="Binary search path; $PATH by default")
  args = parser.parse_args()

  # Run the simulation in the given directory or a temporary one.
  try:
    if args.dir is not None:
      args.dir.mkdir(exist_ok=True)
      run(args, args.dir)
    else:
      with tempfile.TemporaryDirectory() as dir:
        run(args, Path(dir))
  except subprocess.CalledProcessError as e:
    sys.stderr.write(f"\ncommand failed: {shlex.join(map(str, e.cmd))}\n")
    sys.exit(1)


def run(args: argparse.Namespace, dir: Path):
  input: Path = args.input

  # Figure out where the tools are.
  arcilator = args.arcilator_bin or Path(
      shutil.which("arcilator", path=args.path))
  verilator = args.verilator_bin or Path(
      shutil.which("verilator", path=args.path))
  llc = args.llc_bin or Path(shutil.which("llc", path=args.path))
  firtool = args.firtool_bin or Path(shutil.which("firtool", path=args.path))

  # Compile the design with arcilator.
  print("Compiling arcilator simulation")
  state_file = dir / "state.json"
  arc_ll = dir / "arc.ll"
  arc_o = dir / "arc.o"
  arc_h = dir / "arc.h"
  subprocess.check_call(
      [arcilator,
       input.absolute(), "--state-file", state_file, "-o", arc_ll],
      cwd=dir,
  )
  subprocess.check_call(
      [llc, "-O3", "--filetype=obj", arc_ll, "-o", arc_o],
      cwd=dir,
  )
  with open(arc_h, "wb") as out:
    subprocess.check_call(
        [
            sys.executable,
            arcilator.with_name("arcilator-header-cpp.py"), state_file,
            "--view-depth", "1"
        ],
        cwd=dir,
        stdout=out,
    )

  # Read the state file to get the name of the top-level module and the inputs
  # and outputs.
  with open(state_file, "r") as f:
    state_json = json.load(f)
  assert len(state_json) == 1, "design has multiple top-level modules"
  state = state_json[0]

  # Generate the main file that instantiates both models and runs the
  # simulation.
  main_cpp = dir / "main.cpp"
  with open(main_cpp, "w") as out:
    generate_main(out, state)

  # Compile the design with verilator.
  print("Compiling verilator simulation")
  vtor_sv = dir / "vtor.sv"
  stubs_sv = Path(__file__).with_name("compare-arcs-vtor-stubs.sv")
  subprocess.check_call(
      [
          firtool, "--lowering-options=disallowLocalVariables",
          input.absolute(), "-o", vtor_sv
      ],
      cwd=dir,
  )
  subprocess.check_output(
      [
          verilator, "-O3", "--noassert", "--x-assign", "fast", "--x-initial",
          "fast", "--threads", "1", "--sv", "-CFLAGS", f"-I{arcilator.parent}",
          "--cc", "--exe", "--build", vtor_sv, stubs_sv, arc_o, main_cpp
      ],
      cwd=dir,
  )

  # Execute the simulation.
  print("Running simulation")
  status = subprocess.run(["obj_dir/Vvtor"], cwd=dir)
  sys.exit(status.returncode)


MAIN_COMMON = """
#include "arc.h"
#include "verilated.h"
#include <iostream>
#include <memory>
#include <random>

template <typename T>
unsigned check(unsigned cycle, const char *name, T &vtor, T &arcs) {
  if (vtor == arcs)
    return 0;
  std::cout << "Cycle " << cycle << ": Mismatch in " << name << ": "
            << "vtor = " << vtor << ", arcs = " << arcs << "\\n";
  return 1;
}

template <typename T>
void update(T &vtor, T &arcs, std::mt19937_64 &gen) {
  arcs = std::uniform_int_distribution<T>()(gen);
  vtor = arcs;
}
"""


def generate_main(out, info):
  top = info["name"]
  out.write(f"// Top-level module: {top}\n")
  out.write("#include \"Vvtor.h\"\n")
  out.write(MAIN_COMMON.strip() + "\n")
  out.write("\n")
  out.write("int main(int argc, char **argv) {\n")
  out.write("  Verilated::commandArgs(argc, argv);\n")
  out.write("  std::random_device rd;\n")
  out.write("  std::mt19937_64 gen(rd());\n")
  out.write("  auto vtor = std::make_unique<Vvtor>();\n")
  out.write(f"  {top} arcs;\n")
  out.write("  unsigned e = 0;\n")
  out.write("  unsigned i = 0;\n")
  out.write("  for (; i < 1000000 && e < 10; ++i) {\n")
  out.write("    vtor->eval();\n")
  out.write("    arcs.eval();\n")
  for state in info["states"]:
    if state["type"] != "output":
      continue
    name = state["name"]
    out.write("    e += ")
    out.write(f"check(i, \"{name}\", vtor->{name}, arcs.view.{name});\n")
  for state in info["states"]:
    if state["type"] != "input":
      continue
    name = state["name"]
    out.write(f"    update(vtor->{name}, arcs.view.{name}, gen);\n")
  out.write("  }\n")
  out.write("  std::cout << \"Executed \" << i << \" cycles, \";\n")
  out.write("  std::cout << \"found \" << e << \" mismatches\\n\";\n")
  out.write("  return e > 0 ? 1 : 0;\n")
  out.write("}\n")


if __name__ == "__main__":
  main()
