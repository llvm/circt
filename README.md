<p align="center"><img src="docs/includes/img/circt-logo.svg"/></p>

[![](https://github.com/llvm/circt/actions/workflows/buildAndTest.yml/badge.svg?event=push)](https://github.com/llvm/circt/actions?query=workflow%3A%22Build+and+Test%22)
[![Nightly integration tests](https://github.com/llvm/circt/workflows/Nightly%20integration%20tests/badge.svg)](https://github.com/llvm/circt/actions?query=workflow%3A%22Nightly+integration+tests%22)

# ⚡️ "CIRCT" / Circuit IR Compilers and Tools

"CIRCT" stands for "Circuit Intermediate Representations (IR) Compilers and Tools".  One might also interpret
it as the recursively as "CIRCT IR Compiler and Tools".  The T can be
selectively expanded as Tool, Translator, Team, Technology, Target, Tree, Type,
... we're ok with the ambiguity.

The CIRCT community is an open and welcoming community.  If you'd like to
participate, you can do so in a number of different ways:

1) Join our [Discourse Forum](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/) 
on the LLVM Discourse server.  To get a "mailing list" like experience click the 
bell icon in the upper right and switch to "Watching".  It is also helpful to go 
to your Discourse profile, then the "emails" tab, and check "Enable mailing list 
mode".  You can also do chat with us on [CIRCT channel](https://discord.com/channels/636084430946959380/742572728787402763) 
of LLVM discord server.

2) Join our weekly video chat.  Please see the
[meeting notes document](https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#)
for more information.

3) Contribute code.  CIRCT follows all of the LLVM Policies: you can create pull
requests for the CIRCT repository, and gain commit access using the [standard LLVM policies](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access).

## Motivation

The EDA industry has well-known and widely used proprietary and open source
tools.  However, these tools are inconsistent, have usability concerns, and were
not designed together into a common platform.  Furthermore
these tools are generally built with
[Verilog](https://en.wikipedia.org/wiki/Verilog) (also
[VHDL](https://en.wikipedia.org/wiki/VHDL)) as the IRs that they
interchange.  Verilog has well known design issues, and limitations, e.g.
suffering from poor location tracking support.

The CIRCT project is an (experimental!) effort looking to apply MLIR and
the LLVM development methodology to the domain of hardware design tools.  Many
of us dream of having reusable infrastructure that is modular, uses
library-based design techniques, is more consistent, and builds on the best
practices in compiler infrastructure and compiler design techniques.

By working together, we hope that we can build a new center of gravity to draw
contributions from the small (but enthusiastic!) community of people who work
on open hardware tooling.  In turn we hope this will propel open tools forward,
enables new higher-level abstractions for hardware design, and
perhaps some pieces may even be adopted by proprietary tools in time.

For more information, please see our longer [charter document](docs/Charter.md).

## Getting Started

To get started hacking on CIRCT quickly, run the following commands. If you want to include `circt-verilog` in the build, add `-DCIRCT_SLANG_FRONTEND_ENABLED=ON` to the cmake call:

```sh
# Clone the repository and its submodules
git clone git@github.com:llvm/circt.git --recursive
cd circt

# Configure the build
cmake -G Ninja llvm/llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_TARGETS_TO_BUILD=host \
-DLLVM_ENABLE_PROJECTS=mlir \
-DLLVM_EXTERNAL_PROJECTS=circt \
-DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD
-DLLVM_ENABLE_LLD=ON
```

If you want to build everything about the CIRCT tools and libraries, run below command(also runs all tests):
```
ninja -C build check-circt
```

If you want to only build a specific part, for example the `circt-opt` tool:
```sh
ninja -C build bin/circt-opt
```

or the `firtool` tool:
```sh
ninja -C build bin/firtool
```

This will only build the necessary parts of LLVM, MLIR, and CIRCT, which can be a lot quicker than building everything.

### Dependencies

If you have git, ninja, python3, cmake, and a C++ toolchain installed, you should be able to build CIRCT.
For a more detailed description of dependencies, take a look at:

- [Getting Started with MLIR](https://mlir.llvm.org/getting_started/)
- [LLVM Requirements](https://llvm.org/docs/GettingStarted.html#requirements)

### Useful Options

The `-DCMAKE_BUILD_TYPE=Debug` flag enables debug information, which makes the whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.
The `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` flag generates a `compile_commands.json` file, which can be used by editors and language servers for autocomplete and other IDE-like features.

To get something that runs faster but is still easy to debug, use the `-DCMAKE_BUILD_TYPE=RelWithDebInfo` flag to do a release build with debug info.

To do a release build that runs very fast, use the `-DCMAKE_BUILD_TYPE=Release` flag.
Release mode makes a very large difference in performance.

Consult the [Getting Started](docs/GettingStarted.md) page for detailed information on configuring and compiling CIRCT.

Consult the [Python Bindings](docs/PythonBindings.md) page if you are mainly interested in using CIRCT from a Python prompt or script.

### Submodules

CIRCT contains LLVM as a git submodule.
The LLVM repository here includes staged changes to MLIR which may be necessary to support CIRCT.
It also represents the version of LLVM that has been tested.
MLIR is still changing relatively rapidly, so feel free to use the current version of LLVM, but APIs may have changed.

Whenever you checkout a new CIRCT branch that points to a different version of LLVM, run the following command to update the submodule:
```sh
git submodule update
```

The repository is set up to perform a shallow clone of the submodules, meaning it downloads just enough of the LLVM repository to check out the currently specified commit.
If you wish to work with the full history of the LLVM repository, you can manually "unshallow" the submodule:
```sh
cd llvm
git fetch --unshallow
```

### Building LLVM/MLIR Separately

You can also build LLVM/MLIR in isolation first, and then build CIRCT using that first build.
This allows you to pick different compiler options for the two builds, such as building CIRCT in debug mode but LLVM/MLIR in release mode.

First, build and test *LLVM/MLIR*:
```sh
cd llvm
cmake -G Ninja llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host
ninja -C build
ninja -C build check-mlir
cd ..
```

Then build and test *CIRCT*:
```sh
cmake -G Ninja . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm
ninja -C build
ninja -C build check-circt
ninja -C build check-circt-integration
```
