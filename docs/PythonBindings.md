# Using the Python Bindings

If you are mainly interested in using CIRCT from Python scripts, you need to compile both LLVM/MLIR and CIRCT with Python bindings enabled. Furthermore, you must use a unified build, where LLVM/MLIR and CIRCT are compiled together in one step. To do this, you can use a single CMake invocation like this:

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
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON
```

Afterwards, use `ninja check-circt-integration` to ensure that the bindings work. (This will now additionally spin up a couple of Python scripts to test that they are accessible.)

## Without Installation

If you want to try the bindings fresh from the compiler without installation, you need to ensure Python can find the generated modules:

```
export PYTHONPATH="$PWD/llvm/build/tools/circt/python_packages/circt_core"
```

## With Installation

If you are installing CIRCT through `ninja install` anyway, the libraries and Python modules will be installed into the correct location automatically.

## Trying things out

Now you are able to use the CIRCT dialects and infrastructure from a Python interpreter and script:

```python
# silicon.py
import circt
import mlir
from mlir.ir import *
from circt.dialects import hw, comb

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  i42 = IntegerType.get_signless(42)
  m = Module.create()
  with InsertionPoint(m.body):
    def magic(a, b):
      return comb.XorOp(i42, [a, b]).result
    hw.HWModuleOp(name="magic", body_builder=magic)
  print(m)
```

Running this script through `python3 silicon.py` should print the following MLIR:

```mlir
module  {
  hw.module @magic(%a: i42, %b: i42) -> (%result0: i42) {
    %0 = comb.xor %a, %b : i42
    hw.output %0 : i42
  }
}
```
