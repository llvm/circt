# Introduction

PyCDE stands for Python circuit design entry. It is an experimental, opinionated, Python-based fronted for CIRCT's Python bindings. The goal is to make the definition of hardware modules using the bindings simple.

# Installation

## Installing and Building with Wheels

The simplest way to get started using PyCDE is to install it with the `pip install` command:

```
$ cd circt
$ pip install frontends/PyCDE --use-feature=in-tree-build
```

If you just want to build the wheel, use the `pip wheel` command:

```
$ cd circt
$ pip wheel frontends/PyCDE --use-feature=in-tree-build
```

This will create a `pycde-<version>-<python version>-<platform>.whl` file in the root of the repo.

## Manual Compilation

If you are interested in developing PyCDE, or simply want to install it yourself with CMake, you can configure CMake similarly to the [Python bindings](/PythonBindings/#manual-compilation):

```
$ cd circt
$ mkdir build
$ cd build
$ cmake -G Ninja ../llvm/llvm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DCIRCT_ENABLE_FRONTENDS=PyCDE
```

Afterwards, use `ninja check-pycde` to ensure that PyCDE is built and the tests pass.

If you want to use PyCDE after compiling it, you must add the core CIRCT bindings and PyCDE to your PYTHONPATH:

```
export PYTHONPATH="$PWD/build/tools/circt/python_packages/circt_core:$PWD/build/tools/circt/python_packages/pycde"
```

If you are installing PyCDE through `ninja install`, the libraries and Python modules will be installed into the correct location automatically.

## Usage

The following example demonstrates a simple module that adds two integers:

```
import pycde

from circt.dialects import comb


@pycde.module
class AddInts:
    a = pycde.Input(pycde.types.i32)
    b = pycde.Input(pycde.types.i32)
    c = pycde.Output(pycde.types.i32)

    @pycde.generator
    def construct(mod):
        mod.c = comb.AddOp.create(mod.a, mod.b)


system = pycde.System([AddInts])
system.print()
system.generate()
system.print()
system.print_verilog()
```

Modules are decorated with `@pycde.module`, and define their ports using `pycde.Input` and `pycde.Output`. To control how the body of the module is generated, decorate a method with `@pycde.generator`. The generator is passed an object of the module, and ports can be accessed through named attributes.

In order to work with modules, you must add them into a `pycde.System`. This is constructed by passing a list of modules. Calling the `generate` method will run the module(s) generators. Calling the `print_verilog` method will export the system to System Verilog.

The above example will print out three times:

  1. The `AddInts` module with an empty body
  2. The `AddInts` module after its body has been generated
  3. The `AddInts` module represented in System Verilog
