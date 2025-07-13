"""Module extensions for CIRCT dependencies."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _circt_deps_impl(module_ctx):
    """Implementation of the circt_deps module extension."""
    
    # Create a repository pointing to the CIRCT root directory
    new_local_repository(
        name = "circt-raw",
        build_file_content = "# empty",
        path = "../..",
    )
    
    # Download LLVM/MLIR using a git repository
    new_git_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        commit = "945ce1aa3d29e24c49720ae9e0bcfbac88f2defd",  # CIRCT compatible version
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
    )
    
    # Optional LLVM dependencies for performance
    maybe(
        http_archive,
        name = "llvm_zstd",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
        ],
    )
    
    maybe(
        http_archive,
        name = "llvm_zlib",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )

circt_deps = module_extension(
    implementation = _circt_deps_impl,
) 