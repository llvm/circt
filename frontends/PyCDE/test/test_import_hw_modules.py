# RUN: %PYTHON% %s %t | FileCheck %s

from pycde.circt.dialects import hw

from pycde import Input, Output, System, generator, Module, types
from pycde.module import Metadata
from pycde.signals import Struct
from pycde.types import Bit, SInt, TypeAlias

import inspect
import sys

mlir_module = """
hw.module @add(in %a: i1, in %b: i1, out out: i1) {
  %0 = comb.add %a, %b : i1
  hw.output %0 : i1
}

hw.module @and(in %a: i1, in %b: i1, out out: i1) {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}

hw.module.extern @skip()

hw.type_scope @otherScope {}
"""

imported_modules = {}


class ExStruct(Struct):
  a: SInt(8)


# CHECK:  hw.type_scope @pycde {
# CHECK:    hw.typedecl @ExStruct : !hw.struct<a: si8>
# CHECK:  }


class Top(Module):
  a = Input(Bit)
  b = Input(Bit)
  out0 = Output(Bit)
  out1 = Output(Bit)

  @generator
  def generate(ports):
    outs = []
    for mod in imported_modules.values():
      if inspect.isclass(mod) and issubclass(mod, Module):
        outs.append(mod(a=ports.a, b=ports.b).out)

    ports.out0 = outs[0]
    ports.out1 = outs[1]


def preprocess_op(op):
  """Preprocess the operation to skip the `skip` module."""
  if isinstance(op, hw.HWModuleExternOp) and op.name.value == "skip":
    return None
  return op


system = System([Top], output_directory=sys.argv[1])
imported_modules = system.import_mlir(mlir_module, preprocess_op=preprocess_op)
imported_modules["add"].add_metadata(Metadata(name="add"))
system.generate()
TypeAlias.declare_aliases(system.mod)

# CHECK: hw.module @Top(in %a : i1, in %b : i1, out out0 : i1, out out1 : i1)
# CHECK:   %add.out = hw.instance "add" sym @add @add(a: %a: i1, b: %b: i1) -> (out: i1)
# CHECK:   %and.out = hw.instance "and" sym @and @and(a: %a: i1, b: %b: i1) -> (out: i1)
# CHECK:   hw.output %add.out, %and.out : i1, i1

# CHECK: hw.module @add(in %a : i1, in %b : i1, out out : i1)
# CHECK:   %0 = comb.add %a, %b : i1
# CHECK:   hw.output %0 : i1

# CHECK: hw.module @and(in %a : i1, in %b : i1, out out : i1)
# CHECK:   %0 = comb.and %a, %b : i1
# CHECK:   hw.output %0 : i1

# CHECK: esi.manifest.sym @add name "add"
system.print()
