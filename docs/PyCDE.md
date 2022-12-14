# PyCDE

PyCDE stands for Python circuit design entry. It is an experimental, opinionated, Python-based fronted for CIRCT's Python bindings. The goal is to make the definition of hardware modules using the bindings simple.

## To Install PyCDE only via pip

PyCDE is now being released on PyPI: <https://pypi.org/project/pycde/>

## For CIRCT-based Developers

### Cloning the repo

If you havent already, you need to clone the CIRCT repo. Unless you already
have contributor permissions to the LLVM project, the easiest way to develop
(with the ability to create and push branches) is to fork the repo in your
GitHub account. You can then clone your fork. The clone command should look
like this:

```bash
git clone git@github.com:<your_github_username>/circt.git <optional_repo_name>
```

If you don't envision needing that ability, you can clone the main repo
following the directions in step 2 of the [GettingStarted](GettingStarted.md) page.

After cloning, navigate to your repo root (circt is the default) and use the
following to pull down LLVM:

```bash
git submodule update --init
```

## PyCDE Installation

### Installing and Building with Wheels

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

### Manual Compilation

Follow these steps to setup your repository for installing PyCDE via CMake.
Ensure that your repo has the proper Python requirements by running the
following from your CIRCT repo root:

```bash
python -m pip install -r PyCDE/python/requirements.txt
```

Although not scrictly needed for PyCDE develoment, scripts for some tools you
might want to install are located in utils/
(Cap'n Proto, Verilator, OR-Tools):

```bash
utils/get-capnp.sh
utils/get-verilator.sh
utils/get-or-tools
```

Install PyCDE with CMake:

```bash
mkdir build
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DCIRCT_ENABLE_FRONTENDS=PyCDE
    -G Ninja ../llvm/llvm
```

Alternatively, you can pass the source and build paths to the CMake command and
build in the specified folder:

```bash
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DCIRCT_ENABLE_FRONTENDS=PyCDE \
    -G Ninja \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=<your_circt_repo_root_path> \
    -B<path_to_desired_build_dir> \
    <your_circt_repo_root_path>/llvm/llvm
```

Afterwards, use the following commands to ensure that CIRCT and PyCDE are built
and the tests pass:

```bash
ninja -C <path_to_your_circt_build> check-circt
ninja -C <path_to_your_circt_build> check-pycde
ninja -C <path_to_your_circt_build> check-pycde-integration
```

If you want to use PyCDE after compiling it, you must add the core CIRCT
bindings and PyCDE to your PYTHONPATH:

```bash
export PYTHONPATH="<full_path_to_your_circt_build>/tools/circt/python_packages/circt_core:<full_path_to_your_circt_build>/tools/circt/python_packages/pycde"
```

If you are installing PyCDE through `ninja install`, the libraries and Python modules will be installed into the correct location automatically.

## Usage

### Getting Started

The following example demonstrates a simple module that adds two integers:

```python
import pycde

from pycde.dialects import comb


@pycde.module
class AddInts:
    a = pycde.Input(pycde.types.i32)
    b = pycde.Input(pycde.types.i32)
    c = pycde.Output(pycde.types.i32)

    @pycde.generator
    def construct(mod):
        mod.c = comb.AddOp(mod.a, mod.b)


system = pycde.System([AddInts], name="ExampleSystem")
system.print()
system.generate()
system.print()
system.emit_outputs()
```

### Modules, Generators, and Systems

Modules are decorated with `@pycde.module`, and define their ports using `pycde.Input` and `pycde.Output`. To control how the body of the module is generated, decorate a method with `@pycde.generator`.

The generator is passed an object representing the ports of the module, where each port can be accessed through named attributes according to port names. The value on an input port `a` of `mod` is accessed like `expr(mod.a)`. The value on an output port is assigned like `mod.c = expr`.

In order to use `@pycde.module`s, you must add them into a `pycde.System`. This is constructed by passing a list of `@pycde.module`s, and optionally a name. The name defaults to `"PyCDESystem"`, and dictates the output directory for System Verilog files. Calling the `generate` method will run the module(s) generators. Calling the `emit_outputs` method will export the system to System Verilog.

The above example will print out three times:

  1. The `AddInts` module with an empty body
  2. The `AddInts` module after its body has been generated
  3. The `AddInts` module represented in System Verilog, in `ExampleSystem/AddInts.sv`

### Using CIRCT dialects

Note that in the [Getting Started](#getting-started) example, we import `pycde.dialects` rather than `circt.dialects`. The `pycde.dialects` provide thin wrappers around the classes defined in `circt.dialects` and adapt them to [PyCDE Values](#pycde-values) by overriding each operation's constructor.

### PyCDE Values

PyCDE Values are how PyCDE supports named access to module input and output ports. They are also how PyCDE represents the operands and results of any operation imported from `pycde.dialects`. This allows PyCDE generators to stitch together instances of modules, external modules, and any operations from the CIRCT dialects.

#### ListValues and NumPy features

PyCDE supports a subset of numpy array transformations (see `pycde/ndarray.py`) that can be used to do complex reshaping and transformation of multidimensional arrays.

The numpy functionality is provided by the `NDArray` class, which creates a view on top of existing SSA values. Users may choose to perform transformations directly on `ListValue`s:

```python
@module
class M1:
  in1 = Input(dim(types.i32, 4, 8))
  out = Output(dim(types.i32, 2, 16))

  @generator
  def build(ports):
    ports.out = ports.in1.transpose((1, 0)).reshape((16, 2))
    # Under the hood, this resolves to
    # Matrix(from_value=
    #    Matrix(from_value=ports.in1).transpose((1,0)).to_circt())
    #  .reshape(16, 2).to_circt()
```

or manually manage a `NDArray` object.

```python
@module
class M1:
  in1 = Input(dim(types.i32, 4, 8))
  out = Output(dim(types.i32, 2, 16))

  @generator
  def build(ports):
    m = NDArray(from_value=ports.in1).transpose((1, 0)).reshape((16, 2))
    ports.out = m.to_circt()
```

Manually managing the NDArray object allows for postponing materialization (`to_circt()`) until all transformations have been applied.
In short, this allows us to do as many transformations as possible in software, before emitting IR.
Note however, that this might reduce debugability of the generated hardware due to the lack of `sv.wire`s in between each matrix transformation.

For further usage examples, see `PyCDE/test/test_ndarray.py`, and inspect `ListValue` in `pycde/value.py` for the full list of implemented numpy functions.

### Instantiating Modules

Modules defined with `@pycde.module` can be included in other modules. The following example defines a module that instantiates the `AddInts` module defined in [Getting Started](#getting-started).

```python

@pycde.module
class Top:
    a = pycde.Input(pycde.types.i32)
    b = pycde.Input(pycde.types.i32)
    c = pycde.Output(pycde.types.i32)

    @pycde.generator
    def construct(mod):
        add_ints = AddInts(a=mod.a, b=mod.b)
        mod.c = add_ints.c


system = pycde.System([Top], name="ExampleSystem")
system.print()
system.generate()
system.print()
system.emit_outputs()
```

The `Top` module in `ExampleSystem/Top.sv` instantiates the `AddInts` module we defined before.

This demonstrates how to instantiate a module, as well as a more complex usage of [PyCDE Values](#pycde-values). The constructor of a `@pycde.module` expects named keyword arguments for each input port. These keyword arguments can be any PyCDE Values, and the above example uses module inputs. Objects of `@pycde.module` classes represent instances of the module, and support named access to output port values like `add_insts.c`. These output port values are PyCDE Values, and can be use to further connect modules or instantiate CIRCT dialect operations.

### External Modules

External modules are how PyCDE and CIRCT support interacting with existing System Verilog or Verilog modules. They are declared in PyCDE similarly to normal modules. The difference is external modules only contain a module interface but no generator for the implementation. The following example demonstrates using external modules.

```python
@pycde.externmodule("MyMultiplier")
class MulInts:
    a = pycde.Input(pycde.types.i32)
    b = pycde.Input(pycde.types.i32)
    c = pycde.Output(pycde.types.i32)

@pycde.module
class Top:
    a = pycde.Input(pycde.types.i32)
    b = pycde.Input(pycde.types.i32)
    c = pycde.Output(pycde.types.i32)

    @pycde.generator
    def construct(mod):
        add_ints = MulInts(a=mod.a, b=mod.b)
        mod.c = add_ints.c


system = pycde.System([Top], name="ExampleSystem")
system.print()
system.generate()
system.print()
system.emit_outputs()
```

The `Top` module in `ExampleSystem/Top.sv` instantiates the `MyMultiplier` external module.

The `MyMultiplier` module is declared in the default output file, `ExampleSystem/ExampleSystem.sv`.

External modules are decorated with `@pycde.externmodule`, and the decorator accepts a string name to use for the external module name. The ports are declared just like for `@pycde.module`, using `pycde.Input`, `pycde.Output`, and `pycde.types`. External modules are instantiated just like normal modules.
