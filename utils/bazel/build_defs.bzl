"""Build definitions for CIRCT Bazel overlay."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def circt_root_targets():
    """Define the root targets for the CIRCT repository."""
    
    # Root targets are now defined in individual BUILD files
    # This function is kept for compatibility
    pass

def circt_cc_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        copts = [],
        linkopts = [],
        defines = [],
        includes = [],
        strip_include_prefix = "",
        visibility = None,
        **kwargs):
    """Wrapper for cc_library with CIRCT-specific defaults."""
    
    # Default copts for CIRCT
    default_copts = [
        "-std=c++17",
        "-fno-exceptions",
        "-fno-rtti",
        "-D_GNU_SOURCE",
        "-D__STDC_CONSTANT_MACROS",
        "-D__STDC_FORMAT_MACROS",
        "-D__STDC_LIMIT_MACROS",
    ]
    
    # Don't add default deps to avoid duplicates - let the user specify all deps
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = default_copts + copts,
        linkopts = linkopts,
        defines = defines,
        includes = includes,
        strip_include_prefix = strip_include_prefix,
        visibility = visibility,
        **kwargs
    )

def circt_cc_binary(
        name,
        srcs = [],
        deps = [],
        copts = [],
        linkopts = [],
        defines = [],
        visibility = None,
        **kwargs):
    """Wrapper for cc_binary with CIRCT-specific defaults."""
    
    # Default copts for CIRCT
    default_copts = [
        "-std=c++17",
        "-fno-exceptions",
        "-fno-rtti",
        "-D_GNU_SOURCE",
        "-D__STDC_CONSTANT_MACROS",
        "-D__STDC_FORMAT_MACROS",
        "-D__STDC_LIMIT_MACROS",
    ]
    
    # Don't add default deps to avoid duplicates - let the user specify all deps
    native.cc_binary(
        name = name,
        srcs = srcs,
        deps = deps,
        copts = default_copts + copts,
        linkopts = linkopts,
        defines = defines,
        visibility = visibility,
        **kwargs
    )

def circt_tablegen(
        name,
        srcs = [],
        tbl_outs = [],
        deps = [],
        **kwargs):
    """Wrapper for tablegen with CIRCT-specific defaults."""
    
    # This would typically use the mlir_tablegen rule
    # For now, we'll use a simple genrule
    native.genrule(
        name = name,
        srcs = srcs,
        outs = [out[1] for out in tbl_outs],
        cmd = "$(location @llvm-project//mlir:mlir-tblgen) " +
              " ".join(["-" + out[0] for out in tbl_outs]) +
              " -I $(location $(SRCS[0])) " +
              " $(SRCS[0]) -o $(@)",
        tools = ["@llvm-project//mlir:mlir-tblgen"],
        **kwargs
    ) 