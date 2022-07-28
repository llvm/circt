from pycde import System, module

import builtins
import inspect
from pathlib import Path
import subprocess
import inspect
import re


def unittestmodule(generate=True,
                   print=True,
                   run_passes=False,
                   print_after_passes=False,
                   emit_outputs=False,
                   **kwargs):
  """
  Like @module, but additionally performs system instantiation, generation,
  and printing to reduce boilerplate in tests.
  In case of wrapping a function, @testmodule accepts kwargs which are passed
  to the function as arguments.
  """

  def testmodule_inner(func_or_class):
    mod = module(func_or_class)

    # Apply any provided kwargs if this was a function.
    if inspect.isfunction(func_or_class):
      mod = mod(**kwargs)

    # Add the module to global scope in case it's referenced within the
    # module generator functions
    setattr(builtins, mod.__name__, mod)

    sys = System([mod])
    if generate:
      sys.generate()
      if print:
        sys.print()
      if run_passes:
        sys.run_passes()
      if print_after_passes:
        sys.print()
      if emit_outputs:
        sys.emit_outputs()

    return mod

  return testmodule_inner


def cocotest(func):
  # Set a flag on the function to indicate that it's a testbench.
  setattr(func, "_testbench", True)
  return func


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


def _gen_cocotb_testfile(tests):
  """
  Converts testbench functions to cocotb-compatible versions..
  To do this cleanly, we need to detect the indent of the function,
  and remove it from the function implementation.
  """
  template = "import cocotb\n\n"

  for test in tests:
    src = inspect.getsourcelines(test)[0]
    indent = len(src[0]) - len(src[0].lstrip())
    src = [line[indent:] for line in src]
    # Remove the '@cocotest' decorator
    src = src[1:]
    # If the function was not async, make it so.
    if not src[0].startswith("async"):
      src[0] = "async " + src[0]

    # Append to the template as a cocotb test.
    template += "@cocotb.test()\n"
    template += "".join(src)
    template += "\n\n"

  return template


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


def cocotestbench(pycde_mod, simulator="iverilog"):
  """
  Decorator class for defining a class as a PyCDE testbench.
  'pycde_mod' is the PyCDE module under test.
  Within the decorated class, functions with the '@cocotest' decorator
  will be converted to a cocotb-compatible testbench.
  """

  # Ensure that system has 'make' available:
  try:
    subprocess.check_output(["make", "-v"])
  except subprocess.CalledProcessError:
    raise Exception(
        "'make' is not available, and is required to run cocotb tests.")

  try:
    if simulator == "iverilog":
      simhandler = _IVerilogHandler()
    else:
      raise Exception(f"Unknown simulator: {simulator}")
  except Exception as e:
    raise Exception(f"Failed to initialize simulator handler: {e}")

  def testbenchmodule_inner(tb_class):
    sys = System([pycde_mod])
    sys.generate()
    sys.emit_outputs()

    # Generate cocotb makefile
    testmodule = "test_" + pycde_mod.__name__
    makefile_path = Path(sys._output_directory, "Makefile")
    with open(makefile_path, "w") as f:
      f.write(
          _gen_cocotb_makefile(pycde_mod.__name__, testmodule, sys.mod_files,
                               simhandler.sim_name))

    # Find functions with the testbench flag set.
    testbench_funcs = [
        getattr(tb_class, a)
        for a in dir(tb_class)
        if getattr(getattr(tb_class, a), "_testbench", False)
    ]

    # Generate the cocotb test file.
    testfile_path = Path(sys._output_directory, f"{testmodule}.py")
    with open(testfile_path, "w") as f:
      f.write(_gen_cocotb_testfile(testbench_funcs))

    # Run 'make' in the output directory and let cocotb do its thing.
    subprocess.run(["make"], cwd=sys._output_directory)

    return pycde_mod

  return testbenchmodule_inner
