# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw, sim
from circt.ir import (Context, InsertionPoint, IntegerAttr, IntegerType,
                      Location, Module, StringAttr, Type, TypeAttr)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    i1 = IntegerType.get_signless(1)
    true = hw.ConstantOp(IntegerAttr.get(i1, 1))
    fmtOp = sim.FormatLiteralOp(StringAttr.get("foo"))
    # CHECK: sim.fmt.literal "foo"
    print(fmtOp)

  module = Module.parse('''
    module {
      sim.func.dpi @step(out result: i7, in %cycle: i32,
                         inout %state: i1024, return status: i64)
        attributes {verilogName = "step_c"}
      sim.func.dpi @open_array(ref %data: !llvm.ptr)
      sim.func.dpi @finish()
    }
  ''')
  step, open_array, finish = module.body.operations
  assert isinstance(step, sim.DPIFuncOp)
  assert step.verilogName.value == "step_c"
  signature = sim.DPIFunctionType(step.dpi_function_type.value)
  expected = [
      ("result", IntegerType.get_signless(7), sim.DPIDirection.OUTPUT),
      ("cycle", IntegerType.get_signless(32), sim.DPIDirection.INPUT),
      ("state", IntegerType.get_signless(1024), sim.DPIDirection.INOUT),
      ("status", IntegerType.get_signless(64), sim.DPIDirection.RETURN),
  ]
  assert signature.arguments == expected
  assert sim.DPIFunctionType.get(expected) == signature
  # CHECK: !sim.dpi_functy<out "result" : i7, in "cycle" : i32, inout "state" : i1024, return "status" : i64>
  print(signature)
  # CHECK: (i32, i1024) -> (i7, i1024, i64)
  print(signature.function_type)

  pointer_signature = sim.DPIFunctionType(open_array.dpi_function_type.value)
  assert pointer_signature.arguments == [("data", Type.parse("!llvm.ptr"),
                                          sim.DPIDirection.REF)]
  assert sim.DPIFunctionType.get(
      pointer_signature.arguments) == pointer_signature
  empty = sim.DPIFunctionType(finish.dpi_function_type.value)
  assert empty.arguments == []
  assert sim.DPIFunctionType.get([]) == empty

  # Binding access preserves the IR's signedness and does not impose SV rules.
  signedness = sim.DPIFunctionType.get([
      ("s", IntegerType.get_signed(32), sim.DPIDirection.INPUT),
      ("u", IntegerType.get_unsigned(32), sim.DPIDirection.INPUT),
  ])
  assert IntegerType(signedness.arguments[0][1]).is_signed
  assert IntegerType(signedness.arguments[1][1]).is_unsigned

  # Non-integer types can be inspected without parsing their printed syntax.
  float_type = sim.DPIFunctionType.get([
      ("value", Type.parse("f32"), sim.DPIDirection.INPUT),
  ])
  assert str(float_type.arguments[0][1]) == "f32"

  with InsertionPoint(module.body):
    built = sim.DPIFuncOp("built", TypeAttr.get(signature))
  assert sim.DPIFunctionType(built.dpi_function_type.value) == signature
  assert module.operation.verify()

  assert sim.DPIFunctionType.isinstance(signature)
  assert not sim.DPIFunctionType.isinstance(IntegerType.get_signless(32))
  try:
    sim.DPIFunctionType(IntegerType.get_signless(32))
    assert False, "expected an invalid type cast to fail"
  except ValueError:
    pass

  try:
    sim.DPIFunctionType.get([("missing_direction", IntegerType.get_signless(1))
                            ])
    assert False, "expected an invalid argument tuple to fail"
  except ValueError:
    pass

  with Context() as other:
    circt.register_dialects(other)
    assert sim.DPIFunctionType.get(expected, context=ctx) == signature
    try:
      sim.DPIFunctionType.get(expected)
      assert False, "expected arguments from a different context to fail"
    except ValueError:
      pass

  # CHECK: DPI Python bindings verified
  print("DPI Python bindings verified")
