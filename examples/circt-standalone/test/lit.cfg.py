# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "CIRCT_STANDALONE"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.circt_standalone_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(
    ("%circt_standalone_libs", config.circt_standalone_libs_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

tool_dirs = [
    config.circt_standalone_tools_dir,
    config.circt_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "circt-standalone-opt",
]
if config.circt_standalone_has_plugin:
  tools.append("circt-opt")

llvm_config.add_tool_substitutions(tools, tool_dirs)

if not config.circt_standalone_has_plugin:
  config.available_features.add("no-circt-standalone-plugin")
