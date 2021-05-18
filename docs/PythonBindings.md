# Using the Python Bindings

If you are mainly interested in using CIRCT from Python scripts, you need to compile both LLVM/MLIR and CIRCT with Python bindings enabled. To do this, follow the steps in ["Setting this up"](GettingStarted.md#setting-this-up), adding the following options:

```
# in the LLVM/MLIR build directory
cmake [...] -DMLIR_BINDINGS_PYTHON_ENABLED=ON

# in the CIRCT build directory
cmake [...] -DCIRCT_BINDINGS_PYTHON_ENABLED=ON
```

Afterwards, use `ninja check-circt-integration` to ensure that the bindings work. (This will now additionally spin up a couple of Python scripts to test that they are accessible.)

## Without Installation

If you want to try the bindings fresh from the compiler without installation, you need to ensure Python can find the generated modules:

```
export PYTHONPATH="$PWD/llvm/build/python:$PWD/build/python:$PYTHONPATH"
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
    @hw.HWModuleOp.from_py_func(i42, i42)
    def magic(a, b):
      return comb.XorOp(i42, [a, b]).result
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
