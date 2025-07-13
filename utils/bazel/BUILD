# CIRCT Bazel BUILD file
# This file provides the build entry point for CIRCT tools, following the LLVM overlay pattern

load("//:build_defs.bzl", "circt_cc_binary")

package(default_visibility = ["//visibility:public"])

# Export build definitions
exports_files([
    "build_defs.bzl",
    "configure.bzl",
])

# Alias for main tools
alias(
    name = "firtool", 
    actual = "@circt-project//:firtool",
)

alias(
    name = "arcilator",
    actual = "@circt-project//:arcilator",
)

alias(
    name = "circt-as",
    actual = "@circt-project//:circt-as",
)

alias(
    name = "circt-bmc",
    actual = "@circt-project//:circt-bmc",
)

alias(
    name = "circt-dis",
    actual = "@circt-project//:circt-dis",
)

alias(
    name = "circt-lec",
    actual = "@circt-project//:circt-lec",
)

alias(
    name = "circt-lsp-server",
    actual = "@circt-project//:circt-lsp-server",
)

alias(
    name = "circt-opt",
    actual = "@circt-project//:circt-opt",
)

alias(
    name = "circt-reduce",
    actual = "@circt-project//:circt-reduce",
)

alias(
    name = "circt-synth",
    actual = "@circt-project//:circt-synth",
)

alias(
    name = "circt-test",
    actual = "@circt-project//:circt-test",
)

alias(
    name = "circt-translate",
    actual = "@circt-project//:circt-translate",
)

alias(
    name = "firld",
    actual = "@circt-project//:firld",
)

alias(
    name = "handshake-runner",
    actual = "@circt-project//:handshake-runner",
)

alias(
    name = "hlstool",
    actual = "@circt-project//:hlstool",
)

alias(
    name = "kanagawatool",
    actual = "@circt-project//:kanagawatool",
)

alias(
    name = "om-linker",
    actual = "@circt-project//:om-linker",
)

alias(
    name = "circt-verilog",
    actual = "@circt-project//:circt-verilog",
)

alias(
    name = "circt-verilog-lsp-server",
    actual = "@circt-project//:circt-verilog-lsp-server",
)

alias(
    name = "circt-rtl-sim",
    actual = "@circt-project//:circt-rtl-sim",
) 