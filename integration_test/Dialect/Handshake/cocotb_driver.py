import argparse
import os
import subprocess
import sys
import re
from pathlib import Path


def parseArgs(args):
  argparser = argparse.ArgumentParser(description="COCOTB driver for CIRCT")

  argparser.add_argument("--objdir",
                         type=str,
                         help="Select a directoy in which to run this test." +
                         " Must be different from other tests in the same" +
                         " directory. Defaults to 'sources[0].d'.")

  argparser.add_argument("--topLevel",
                         type=str,
                         help="Name of the top level verilog module.")

  argparser.add_argument("--simulator",
                         choices=['iverilog'],
                         default="iverilog",
                         help="Name of the simulator to use.")

  argparser.add_argument("--pythonModule",
                         type=str,
                         required=True,
                         help="Name of the python module.")

  argparser.add_argument("--pythonFolder",
                         type=str,
                         default=os.getcwd(),
                         help="The folder where the cocotb test file is.")

  argparser.add_argument("sources",
                         nargs="+",
                         help="The list of source files to be included.")

  return argparser.parse_args(args[1:])


class _IVerilogHandler:
  """ Class for handling icarus-verilog specific commands and patching."""

  def __init__(self):
    # Ensure that iverilog is available in path and it is at least iverilog v11
    try:
      out = subprocess.check_output(["iverilog", "-V"])
    except subprocess.CalledProcessError:
      raise Exception("iverilog not found in path")

    # find the 'Icarus Verilog version #' string and extract the version number
    # using a regex
    ver_re = r"Icarus Verilog version (\d+\.\d+)"
    ver_match = re.search(ver_re, out.decode("utf-8"))
    if ver_match is None:
      raise Exception("Could not find Icarus Verilog version")
    ver = ver_match.group(1)
    if float(ver) < 11:
      raise Exception(f"Icarus Verilog version must be >= 11, got {ver}")

  @property
  def sim_name(self):
    return "icarus"


def _gen_cocotb_makefile(top, testmod, sources, sim):
  """
  Creates a simple cocotb makefile suitable for driving the testbench.
  """
  template = f"""
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = {" ".join(list(sources))}
TOPLEVEL = {top}
MODULE = {testmod}
SIM={sim}

include $(shell cocotb-config --makefiles)/Makefile.sim
"""
  return template


def main():
  args = parseArgs(sys.argv)
  sources = [os.path.abspath(s) for s in args.sources]
  args.sources = sources

  if args.objdir is not None:
    objDir = args.objdir
  else:
    objDir = f"{os.path.basename(args.sources[0])}.d"
  objDir = os.path.abspath(objDir)
  if not os.path.exists(objDir):
    os.mkdir(objDir)
  os.chdir(objDir)

  # Ensure that system has 'make' available:
  try:
    subprocess.check_output(["make", "-v"])
  except subprocess.CalledProcessError:
    raise Exception(
        "'make' is not available, and is required to run cocotb tests.")

  try:
    if args.simulator == "iverilog":
      simhandler = _IVerilogHandler()
    else:
      raise Exception(f"Unknown simulator: {simulator}")
  except Exception as e:
    raise Exception(f"Failed to initialize simulator handler: {e}")

  # Generate cocotb makefile
  testmodule = "test_" + args.topLevel
  makefile_path = Path(objDir, "Makefile")
  with open(makefile_path, "w") as f:
    f.write(
        _gen_cocotb_makefile(args.topLevel, args.pythonModule, sources,
                             simhandler.sim_name))

  # Adding the original working dir to the path, as the test file lies there
  my_env = os.environ.copy()
  my_env["PYTHONPATH"] = args.pythonFolder + os.pathsep + my_env["PYTHONPATH"]
  # Run 'make' in the output directory and let cocotb do its thing.
  subprocess.run(["make"], cwd=objDir, env=my_env)


if __name__ == "__main__":
  main()
