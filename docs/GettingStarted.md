# Getting Started with the CIRCT Project

## Overview

Welcome to the CIRCT project!

"CIRCT" stands for "Circuit IR Compilers and Tools".  The CIRCT project is an
(experimental!) effort looking to apply MLIR and the LLVM development
methodology to the domain of hardware design tools.

Take a look at the following diagram, which gives a brief overview of the
current [dialects and how they interact](https://circt.llvm.org/includes/img/dialects.svg):

<p align="center"><img src="https://circt.llvm.org/includes/img/dialectlegend.svg"/></p>
<p align="center"><img src="https://circt.llvm.org/includes/img/dialects.svg"/></p>

## Setting this up

These commands can be used to setup CIRCT project:

1) **Install Dependencies** of LLVM/MLIR according to [the
  instructions](https://mlir.llvm.org/getting_started/), including cmake and
ninja.

*Note:* CIRCT is known to build with at least GCC 9.4 and Clang 13.0.1, but
older versions may not be supported. It is recommended to use the same C++
toolchain to compile both LLVM and CIRCT to avoid potential issues.   
  
*Recommendation:* In order to greatly reduce memory usage during linking, we
recommend using the LLVM linker [`lld`](https://lld.llvm.org/).

If you plan to use the Python bindings, you should start by reading [the
 instructions](https://mlir.llvm.org/docs/Bindings/Python/#building) for building
the MLIR Python bindings, which describe extra dependencies, CMake variables,
and helpful Python development practices. Note the extra CMake variables, which
you will need to specify in step 3) below.

2) **Check out LLVM and CIRCT repos.**  CIRCT contains LLVM as a git
submodule.  The LLVM repo here includes staged changes to MLIR which
may be necessary to support CIRCT.  It also represents the version of
LLVM that has been tested.  MLIR is still changing relatively rapidly,
so feel free to use the current version of LLVM, but APIs may have
changed.

```
$ git clone git@github.com:circt/circt.git
$ cd circt
$ git submodule init
$ git submodule update
```

*Note:* The repository is set up so that `git submodule update` performs a
shallow clone, meaning it downloads just enough of the LLVM repository to check
out the currently specified commit. If you wish to work with the full history of
the LLVM repository, you can manually "unshallow" the submodule:

```
$ cd llvm
$ git fetch --unshallow
```

3) **Build and test LLVM/MLIR:**

```
$ cd circt
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON  
$ ninja
$ ninja check-mlir
```

4) **Build and test CIRCT:**

```
$ cd circt
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON  
$ ninja
$ ninja check-circt
$ ninja check-circt-integration # Run the integration tests.
```  
In order to use the recommended `lld` linker, use the `-DLLVM_ENABLE_LLD=ON`. 
Removing that flag will use your compiler's default linker. More details about
these problems and their solutions can be found 
[in the LLVM docs](https://llvm.org/docs/GettingStarted.html#common-problems).  

The `-DCMAKE_BUILD_TYPE=DEBUG` flag enables debug information, which makes the
whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.

To get something that runs fast, use `-DCMAKE_BUILD_TYPE=Release` or
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` if you want to go fast and optionally if
you want debug info to go with it.  `Release` mode makes a very large difference
in performance.

To use the verilog frontend add `-DCIRCT_SLANG_FRONTEND_ENABLED=ON`.

If you plan to use the Python bindings, you should also specify
`-DCIRCT_BINDINGS_PYTHON_ENABLED=ON`.

5) **Optionally configure your environment**:

It is useful to add the `.../circt/build/bin` and `.../circt/llvm/build/bin`
directories to the end of your PATH, allowing you to use the tools like `circt-opt`
in a natural way on the command line.  Similarly, you need to be in the build
directory to invoke ninja, which is super annoying.  You might find a bash/zsh
alias like this to be useful:

```bash
build() {
  (cd $HOME/Projects/circt/build/; ninja $1 $2 $3)
}
```

This allows you to invoke `build check-circt` from any directory and have it do
the right thing.

6) **Run the Verilator tests:** (optional)

[Verilator](https://github.com/verilator/verilator) can be used to check
SystemVerilog code. To run the tests, build or install a **recent** version
of Verilator (at least v4.034, ideally v4.110 or later to avoid a known bug).
(Some Linux distributions have *ancient* versions.) If Verilator is in your
PATH, `build check-circt` should run the tests which require Verilator.

We provide a script `utils/get-verilator.sh` to automate the download and
compilation of Verilator into a known location. The testing script will check
this location first. This script assumes that all the Verilator package
dependencies are installed on your system. They are:

- make
- autoconf
- g++
- flex
- bison
- libfl2     # Ubuntu only (ignore if gives error)
- libfl-dev  # Ubuntu only (ignore if gives error)

7) **Install GRPC** (optional, affects ESI runtime only)

The ESI runtime requires GRPC for cosimulation. The `utils/get-grpc.sh` script
installs a known good version of GRPC to a directory within the CIRCT source
code. Alternatively, you can install GRPC using your package manager, though the
version may not be compatible with the ESI runtime so results may vary.


8) **Install OR-Tools** (optional, enables additional schedulers)

[OR-Tools](https://developers.google.com/optimization/) is an open source
software suite for (mathematical) optimization. It provides a uniform interface
to several open-source and commercial solvers, e.g. for linear programs and
satisfiability problems. Here, it is optionally used in the static scheduling
infrastructure. Binary distributions often do not include the required CMake
build info. The `utils/get-or-tools.sh` script downloads, compiles, and
installs a known good version to a directory within the CIRCT source code,

## Setting up VS Code Workspace

We've provided an example VS Code file in `.vscode/Unified.code-workspace.jsonc` that can be used with the VS Code editor. To use the file, first copy to into a workspace file:
```sh
cp .vscode/Unified.code-workspace.jsonc .vscode/circt.code-workspace
```

Next, open the workspace file in VS code using the command palette (Ctrl + Shift + P) and selecting "Open workspace from file" and selecting the `.vscode/circt.code-workspace` file.

Alternatively, open the file using:
```
code .vscode/circt.code-workspace
```
and select "open workspace" on the bottom right.

Once the workspace is loaded, install the recommended tools and select "CMake: Build" from the command palette to start the unified build process. This will build the LLVM dependencies and CIRCT together.

where it is then picked up automatically by the build.

## Windows: notes on setting up with Ninja

Building on Windows using MSVC + Ninja + Python support is straight forward,
though full of landmines. Here are some notes:

- Ninja and cmake must be run in a VS Developer Command shell. If you use
Powershell and don't want to start the VS GUI, you can run:
```powershell
> $vsPath = &(Join-Path ${env:ProgramFiles(x86)} "\Microsoft Visual Studio\Installer\vswhere.exe") -property installationpath
> Import-Module (Get-ChildItem $vsPath -Recurse -File -Filter Microsoft.VisualStudio.DevShell.dll).FullName
> Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation
```

- VSCode's cmake configure does not operate properly with Python support. The
symptom is that the build will complete, but importing `circt` or `mlir` crashes
Python. Doing everything from the command line is the only way CIRCT compiles
have been made to work.

Cheat sheet for powershell:

```powershell
# Install cmake, ninja, and Visual Studio
> python -m pip install psutil pyyaml numpy pybind11

> $vsPath = &(Join-Path ${env:ProgramFiles(x86)} "\Microsoft Visual Studio\Installer\vswhere.exe") -property installationpath
> Import-Module (Get-ChildItem $vsPath -Recurse -File -Filter Microsoft.VisualStudio.DevShell.dll).FullName
> Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation

> cd <circt clone>
> cmake -B<build_dir> llvm/llvm `
    -GNinja `
    -DLLVM_ENABLE_PROJECTS=mlir `
    -DCMAKE_BUILD_TYPE=Debug `
    -DLLVM_TARGETS_TO_BUILD=X86 `
    -DLLVM_ENABLE_ASSERTIONS=ON `
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON `
    -DLLVM_EXTERNAL_PROJECTS=circt `
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR="$(PWD)" `
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DLLVM_USE_SPLIT_DWARF=ON 
    -DLLVM_ENABLE_LLD=ON
> ninja -C<build_dir> check-circt
```

## Submitting changes to CIRCT

The project is small so there are few formal process yet.  We generally follow
the LLVM and MLIR community practices, but we currently use pull requests and
GitHub issues.  Here are some high-level guidelines:

 * Please use clang-format in the LLVM style.  There are good plugins
   for common editors like VSCode, Atom, etc, or you can run it
   manually.  This makes code easier to read and understand.

 * Beyond mechanical formatting issues, please follow the [LLVM Coding
   Standards](https://llvm.org/docs/CodingStandards.html).

 * Please practice "[incremental development](https://llvm.org/docs/DeveloperPolicy.html#incremental-development)",
   preferring to send a small series of incremental patches rather than large
   patches.  There are other policies in the LLVM Developer Policy document that
   are worth skimming.

 * Please use "Squash and Merge" in PRs when they are approved - we don't
   need the intra-change history in the repository history.

 * Please create a PR to get a code review.  For reviewers, it is good to look
   at the primary author of the code you are touching to make sure they are at
   least CC'd on the PR.

## Submitting changes to LLVM / MLIR

This project depends on MLIR and LLVM, and it is occasionally useful to improve
them.
To get set up for this:

1) Follow the "[How to Contribute](https://mlir.llvm.org/getting_started/Contributing/)"
instructions, and install the right tools, e.g. `clang-format`.
2) Optional: [Ask for LLVM commit access](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access),
the barrier is low. Alternatively, you can ask one of the reviewers on the
GitHub pull-request to merge for you.

### Submitting a patch

Patches are submitted to LLVM/MLIR via GitHub pull-requests, the basic flow is as follows:

1) Check out the LLVM mono repo (as described above) or your fork of the same.
2) Make changes to your codebase in a dedicated branch for your patch.
3) Stage your changes with `git add`.
4) Run clang-format to tidy up the patch with `git clang-format origin/main`.
5) Run tests with `ninja check-mlir`  (or whatever other target makes sense).
6) Publish the branch on your fork of the repository and create a GitHub pull-request.

When your review converges and your patch is approved, it can be merged directly on GitHub.
If you have commit access, you can do this yourself, otherwise a reviewer can do it for you.  
  
## Writing a basic CIRCT Pass  

Passes can be added at several levels in CIRCT. Here we illustrate this with a simple pass, targetting 
the `hw` dialect, that replaces all of the wire names with `foo. This example is very basic and is 
meant for people who want to get a quick and dirty start at writing passes for CIRCT. For more detailed
tutorials, we recommend looking at the [MLIR docs](https://mlir.llvm.org/docs/PassManagement/), and their
[Toy tutorials](https://mlir.llvm.org/docs/Tutorials/Toy/).  
To add a simple dialect pass, that doesn't perform any dialect conversion, you can do the following:  
1) Update your dialect’s Passes.td file to define what your pass will do in a high-level, well documented way, 
e.g. in `include/circt/Dialect/HW/Passes.td`:  
```
def FooWires : Pass<"hw-foo-wires", "hw::HwModuleOp"> {
  let summary = "Change all wires' name to foo_<n>.";
  let description = [{
    Very basic pass that numbers all of the wires in a given module.
    The wires' names are then all converte to foo_<that number>.
  }];
  let constructor = "circt::hw::createFooWiresPass()";
}
```    
Once this is added, compile CIRCT. This will generate a base class for your pass following the naming
defined in the [tablegen description](https://mlir.llvm.org/docs/PassManagement/#tablegen-specification). 
This base class will contain some methods that need to be implemented. 
Your goal is to now implement those.  

2) Create a new file that contains your pass in the dialect's pass folder, e.g. in `lib/Dialect/HW/Transforms`. 
Don't forget to also add it in the folder's CMakeLists.txt. 
This file should implement:  
  - A struct with the name of your pass, which extends the pass base class generated by the tablegen description.
  - This struct should define an override of the `void runOnOperation()` method, e.g.  
  ```cpp
  namespace {
    // A test pass that simply replaces all wire names with foo_<n>
    struct FooWiresPass : FooWiresBase<FooWiresPass> {
      void runOnOperation() override;
    };
  }
  ```  
  - The `runOnOperation` method will contain the actual logic of your pass, which you can now implement, e.g.
  ```cpp
  void FooWiresPass::runOnOperation() {
    size_t nWires = 0; // Counts the number of wires modified
    getOperation().walk([&](hw::WireOp wire) { // Walk over every wire in the module
      wire.setName("foo_" + std::to_string(nWires++)); // Rename said wire
    });
  }
  ``` 
  > *Note*: Here `getOperation().walk([&](WireOp wire) { ... });` is used to traverse every wire in the design, you can also
  > use it to generically walk over all operations and then visit them individually 
  > using a [visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern) by overloading type visitors 
  > defined by each dialect. More details can be found in the 
  > [MLIR docs](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/), e.g.
  ```cpp
  #include "circt/Dialect/HW/HWVisitors.h" // defines dispatchTypeOpVisitor which calls visit(op)

  namespace {
    // A test pass that simply replaces all wire names with foo_<n>
    struct VisitorExamplePass 
      : public VisitorExampleBase<VisitorExamplePass>,
        public hw::TypeOpVisitor<VisitorExamplePass> // Allows for the visitor overloads to be added
    {
    public:  
      void runOnOperation() override;

      // Vistor overloads
      void visitTypeOp(hw::ConstantOp op) { /*...*/ }
      void visitTypeOp(/*other hw operations*/) { /*...*/ }
    };
  }

  void VisitorExamplePass::runOnOperation() {
    module.walk([&](Operation* op) { dispatchTypeOpVisitor(op); });
  }
  ```
  - Finally implement the constructor for the pass, as defined in the tablegen description, e.g.  
  ```cpp
  std::unique_ptr<mlir::Pass> circt::hw::createFooWiresPass() {
    return std::make_unique<FooWiresPass>();
  }
  ```  
    
3) Define the pass constructor in the dialect's `Passes.h` file, e.g. in `include/circt/Dialect/hw/HWPasses.h` add:  
```cpp
std::unique_ptr<mlir::Pass> createFooWiresPass();
```
4) Make sure to add a test to check your pass, e.g. in `test/Dialect/HW`:  
```mlir
// RUN: circt-opt --hw-foo-wires %s | FileCheck %s

hw.module @foo(in %a: i32, in %b: i32, out out: i32) {
  // CHECK:   %c1_i32 = hw.constant 1 : i32
  %c1 = hw.constant 1 : i32
  // CHECK:   %foo_0 = hw.wire %c1_i32  : i32
  %wire_1 = hw.wire %c1 : i32
  // CHECK:   %foo_1 = hw.wire %a  : i32
  %wire_a = hw.wire %a : i32
  // CHECK:   %foo_2 = hw.wire %b  : i32
  %wire_b = hw.wire %b : i32
  // CHECK:   %0 = comb.add bin %foo_1, %foo_0 : i32
  %ap1 = comb.add bin %wire_a, %wire_1 : i32
  // CHECK:   %foo_3 = hw.wire %0  : i32
  %wire_ap1 = hw.wire %ap1 : i32
  // CHECK:   %1 = comb.add bin %foo_3, %foo_2 : i32
  %ap1pb = comb.add bin %wire_ap1, %wire_b : i32
  // CHECK:   %foo_4 = hw.wire %1  : i32
  %wire_ap1pb = hw.wire %ap1pb : i32
  // CHECK:   hw.output %foo_4 : i32
  hw.output %wire_ap1pb : i32
}
```
Now re-run cmake and compile CIRCT and you should be able to run your new pass! 
By default, every pass is accessible using `circt-opt` through the name that was defined in the tablegen description, 
e.g. `circt-opt --hw-foo-wires <file_name>.mlir`. 
These passes can also be included in certain compilation pipelines that are well packaged in tools like firtool.  
We recommend looking at recently merged Pull-Requests or other [MLIR tutorials](https://mlir.llvm.org/docs/Tutorials/) 
to learn how to go beyond what we've shown in this quick start guide.  
The full source code of this pass is available at the following links:  
- Pass Source: [FooWires.cpp](https://github.com/llvm/circt/blob/main/lib/Dialect/HW/Transforms/FooWires.cpp)   
- Pass Test: [foo.mlir](https://github.com/llvm/circt/blob/main/test/Dialect/HW/foo.mlir)  
- HW Dialect Pass Header: [HWPasses.h](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/HWPasses.h#L32)  
- Tablegen Description: [Passes.td](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/Passes.td#L81-L88)  
- [CMakeLists.txt](https://github.com/llvm/circt/blob/main/lib/Dialect/HW/Transforms/CMakeLists.txt#L8)
