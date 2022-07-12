from pycde import System, module

import builtins
import inspect
from pathlib import Path
import subprocess
import inspect


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


def pycdetest(func):
  # Set a flag on the function to indicate that it's a testbench.
  setattr(func, "_testbench", True)
  return func


def gen_cocotb_makefile(top, testmod, sources, sim):
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


def gen_cocotb_testfile(tests):
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
    # Remove the '@pycdetest' decorator
    src = src[1:]
    # If the function was not async, make it so.
    if not src[0].startswith("async"):
      src[0] = "async " + src[0]

    # Append to the template as a cocotb test.
    template += "@cocotb.test()\n"
    template += "".join(src)
    template += "\n\n"

  return template


class IVerilogHandler:
  """ Class for handling icarus-verilog specific commands and patching."""

  @property
  def sim_name(self):
    return "icarus"

  def fix_sv(self, filename):
    """
    Icarus verilog requires the following fixups to get a reasonable subset of
    generated stuff to simulate:
    - Replace all always_# in the output file with 'always'
    - Add a default parameter to the __INST_HIER parameter.
    """
    patterns = {
        "always_comb": "always",
        "always_latch": "always",
        "always_ff": "always",
        # need a default value for parameters, else it's considered a syntax error by iverilog
        "parameter __INST_HIER": "parameter __INST_HIER=0"
    }

    with open(filename, "r") as f:
      lines = f.readlines()
    with open(filename, "w") as f:
      for line in lines:
        for pattern, replacement in patterns.items():
          line = line.replace(pattern, replacement)
        f.write(line)


def testbench(pycde_mod, simulator="iverilog"):
  """
  Decorator class for defining a class as a PyCDE testbench.
  'pycde_mod' is the PyCDE module under test.
  Within the decorated class, functions with the '@pycdetest' decorator
  will be converted to a cocotb-compatible testbench.
  """

  # Ensure that system has 'make' available:
  try:
    subprocess.check_output(["make", "-v"])
  except subprocess.CalledProcessError:
    raise Exception(
        "'make' is not available, and is required to run cocotb tests.")

  if simulator == "iverilog":
    simhandler = IVerilogHandler()
  else:
    raise Exception(f"Unknown simulator: {simulator}")

  def testbenchmodule_inner(tb_class):
    sys = System([pycde_mod])
    sys.generate()
    sys.emit_outputs()

    # Generate cocotb makefile
    testmodule = "test_" + pycde_mod.__name__
    makefile_path = Path(sys._output_directory, "Makefile")
    with open(makefile_path, "w") as f:
      f.write(
          gen_cocotb_makefile(pycde_mod.__name__, testmodule, sys.mod_files,
                              simhandler.sim_name))

    # Find functions with the testbench flag set.
    testbench_funcs = [
        getattr(tb_class, a)
        for a in dir(tb_class)
        if getattr(getattr(tb_class, a), "_testbench", False)
    ]

    # Do simulator-specific patching of the generated .sv files.
    for mod_file in sys.mod_files:
      simhandler.fix_sv(mod_file)

    # Generate the cocotb test file.
    testfile_path = Path(sys._output_directory, f"{testmodule}.py")
    with open(testfile_path, "w") as f:
      f.write(gen_cocotb_testfile(testbench_funcs))

    # Run 'make' in the output directory and let cocotb do its thing.
    subprocess.run(["make"], cwd=sys._output_directory)

    return pycde_mod

  return testbenchmodule_inner
