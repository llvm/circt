# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import tempfile
import warnings

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'CIRCT'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.ll', '.fir', '.sv', '.py', '.tcl']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%shlibdir', config.circt_shlib_dir))
config.substitutions.append(('%INC%', config.circt_include_dir))
config.substitutions.append(
    ('%BININC%', os.path.join(config.circt_obj_root, "include")))
config.substitutions.append(
    ('%TCL_PATH%', config.circt_src_root + '/build/lib/'))
config.substitutions.append(('%CIRCT_SOURCE%', config.circt_src_root))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# Set the timeout, if requested.
if config.timeout is not None and config.timeout != "":
  lit_config.maxIndividualTestTime = int(config.timeout)

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt', 'lit.cfg.py',
    'lit.local.cfg.py'
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'integration_test')

# Tweak the PATH to include the LLVM and CIRCT tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.circt_tools_dir, append_path=True)

# Tweak the PYTHONPATH to include the binary dir.
if config.bindings_python_enabled:
  llvm_config.with_environment(
      'PYTHONPATH',
      [os.path.join(config.circt_python_packages_dir, 'circt_core')],
      append_path=True)

tool_dirs = [
    config.circt_tools_dir, config.circt_utils_dir, config.mlir_tools_dir,
    config.llvm_tools_dir
]
tools = [
    'arcilator', 'circt-opt', 'circt-translate', 'firtool', 'circt-rtl-sim.py',
    'equiv-rtl.sh', 'handshake-runner', 'hlstool', 'kanagawatool', 'circt-lec',
    'circt-bmc', 'circt-test', 'circt-test-runner-sby.py',
    'circt-test-runner-circt-bmc.py', 'circt-cocotb-driver.py'
]

# Enable python if its path was configured
if config.python_executable != "":
  tool_dirs.append(os.path.dirname(config.python_executable))
  config.available_features.add('python')
  config.substitutions.append(('%PYTHON%', f'"{config.python_executable}"'))

# Enable yosys if it has been detected.
if config.yosys_path != "":
  tool_dirs.append(os.path.dirname(config.yosys_path))
  tools.append('yosys')
  config.available_features.add('yosys')

# Enable Icarus Verilog as a fallback if no other ieee-sim was detected.
if config.iverilog_path != "":
  tool_dirs.append(os.path.dirname(config.iverilog_path))
  tools.append('iverilog')
  tools.append('vvp')
  config.available_features.add('iverilog')
  config.available_features.add('ieee-sim')
  config.available_features.add('rtl-sim')
  config.substitutions.append(('%iverilog', config.iverilog_path))
  config.substitutions.append(('%ieee-sim', config.iverilog_path))

# Enable Verilator if it has been detected.
if config.verilator_path != "":
  tool_dirs.append(os.path.dirname(config.verilator_path))
  tools.append('verilator')
  config.available_features.add('verilator')
  config.available_features.add('rtl-sim')
  llvm_config.with_environment('VERILATOR_PATH', config.verilator_path)

# Enable Quartus if it has been detected.
if config.quartus_path != "":
  tool_dirs.append(os.path.dirname(config.quartus_path))
  tools.append('quartus')
  config.available_features.add('quartus')

# Enable Vivado if it has been detected.
if config.vivado_path != "":
  tool_dirs.append(config.vivado_path)
  tools.append('xvlog')
  tools.append('xelab')
  tools.append('xsim')
  config.available_features.add('ieee-sim')
  config.available_features.add('vivado')
  config.substitutions.append(
      ('%ieee-sim', os.path.join(config.vivado_path, "xsim")))
  config.substitutions.append(('%xsim%', os.path.join(config.vivado_path,
                                                      "xsim")))

# Enable Questa if it has been detected.
if config.questa_path != "":
  config.available_features.add('questa')
  config.available_features.add('ieee-sim')
  config.available_features.add('rtl-sim')
  if 'LM_LICENSE_FILE' in os.environ:
    llvm_config.with_environment('LM_LICENSE_FILE',
                                 os.environ['LM_LICENSE_FILE'])

  tool_dirs.append(config.questa_path)
  tools.append('vlog')
  tools.append('vsim')

  config.substitutions.append(
      ('%questa', os.path.join(config.questa_path, "vsim")))
  config.substitutions.append(
      ('%ieee-sim', os.path.join(config.questa_path, "vsim")))

ieee_sims = list(filter(lambda x: x[0] == '%ieee-sim', config.substitutions))
if len(ieee_sims) > 1:
  warnings.warn(
      f"You have multiple ieee-sim simulators configured, choosing: {ieee_sims[-1][1]}"
  )
  # remove all other subsitution entries
  config.substitutions = list(
      filter(lambda x: x[0] != '%ieee-sim' or x == ieee_sims[-1],
             config.substitutions))

# If the ieee-sim was selected to be iverilog in case no other simulators are
# available, define a feature flag to allow tests which cannot be simulated
# with iverilog to be disabled.
if ieee_sims and ieee_sims[-1][1] == config.iverilog_path:
  config.available_features.add('ieee-sim-iverilog')

config.substitutions.append(("%esi_prims", config.esi_prims))

# Enable ESI runtime tests.
if config.esi_runtime == "1":
  config.available_features.add('esi-runtime')
  tools.append('esiquery')
  tools.append('esitester')

  llvm_config.with_environment('PYTHONPATH',
                               [f"{config.esi_runtime_path}/python/"],
                               append_path=True)

  # Enable ESI cosim tests if they have been built.
  if config.esi_cosim != "OFF":
    config.available_features.add('esi-cosim')
    tools.append('esi-cosim.py')

# Enable Python bindings tests if they're supported.
if config.bindings_python_enabled:
  config.available_features.add('bindings_python')
if config.bindings_tcl_enabled:
  config.available_features.add('bindings_tcl')

# Add host c/c++ compiler.
config.substitutions.append(("%host_cc", config.host_cc))
config.substitutions.append(("%host_cxx", config.host_cxx))

# Enable clang-tidy if it has been detected.
if config.clang_tidy_path != "":
  tool_dirs.append(config.clang_tidy_path)
  tools.append('clang-tidy')
  config.available_features.add('clang-tidy')

# Enable systemc if it has been detected.
if config.have_systemc != "":
  config.available_features.add('systemc')

# Enable z3 if it has been detected.
if config.z3_path != "":
  tool_dirs.append(config.z3_path)
  tools.append('z3')
  config.available_features.add('z3')

# Enable libz3 if it has been detected.
if config.z3_library != "":
  tools.append(ToolSubst(f"%libz3", config.z3_library))
  config.available_features.add('libz3')

# Enable SymbiYosys if it has been detected.
if config.sby_path != "":
  tool_dirs.append(config.sby_path)
  tools.append('sby')
  config.available_features.add('sby')

# Add mlir-runner if the execution engine is built.
if config.mlir_enable_execution_engine:
  config.available_features.add('mlir-runner')
  config.available_features.add('circt-lec-jit')
  config.available_features.add('circt-bmc-jit')
  tools.append('mlir-runner')

# Add circt-verilog if the Slang frontend is enabled.
if config.slang_frontend_enabled:
  config.available_features.add('slang')
  tools.append('circt-verilog')

# Add arcilator JIT if MLIR's execution engine is enabled.
if config.arcilator_jit_enabled:
  config.available_features.add('arcilator-jit')

config.substitutions.append(('%driver', f'{config.driver}'))
config.substitutions.append(('%circt-tools-dir', f'{config.circt_tools_dir}'))
llvm_config.add_tool_substitutions(tools, tool_dirs)

# cocotb availability
try:
  import cocotb
  import cocotb_test
  config.available_features.add('cocotb')
except ImportError:
  pass
