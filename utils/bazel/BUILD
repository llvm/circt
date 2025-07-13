# CIRCT Bazel BUILD file
# This file provides the build entry point for CIRCT tools, following the LLVM overlay pattern

load("//:build_defs.bzl", "circt_cc_binary")

package(default_visibility = ["//visibility:public"])

# Export build definitions
exports_files([
    "build_defs.bzl",
    "configure.bzl",
])

# Alias for main tool
alias(
    name = "firtool", 
    actual = "@circt-project//:firtool",
) 