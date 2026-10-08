# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import io
import json

import circt
from circt.dialects import emit, hw, sim
from circt.ir import (ArrayAttr, Context, FlatSymbolRefAttr, IntegerType,
                      Location, Module, StringAttr, WalkResult)
from circt.passmanager import PassManager


def serialize_integer_dpi(module):
  """Example consumer for the current SV exporter's scalar integer mapping."""
  directions = {
      sim.DPIDirection.INPUT: "in",
      sim.DPIDirection.OUTPUT: "out",
      sim.DPIDirection.INOUT: "inout",
      sim.DPIDirection.RETURN: "return",
  }
  functions = []

  def visit(operation):
    func = operation.opview
    if not isinstance(func, sim.DPIFuncOp):
      return WalkResult.ADVANCE
    signature = sim.DPIFunctionType(func.dpi_function_type.value)
    arguments = []
    for name, type, direction in signature.arguments:
      if not isinstance(type, IntegerType) or direction not in directions:
        raise ValueError("this example only handles integer DPI arguments")
      width = IntegerType(type).width
      # ExportVerilog emits native signed atoms at these widths, and unsigned
      # bit vectors at other widths, regardless of the IR integer signedness.
      signed = width in (8, 16, 32, 64)
      if direction == sim.DPIDirection.RETURN and width not in (1, 8, 16, 32,
                                                                64):
        raise ValueError("packed vectors cannot be DPI function return values")
      arguments.append({
          "name": name,
          "direction": directions[direction],
          "width": width,
          "signed": signed,
      })
    name = func.verilogName or func.sym_name
    functions.append({"function": name.value, "arguments": arguments})
    return WalkResult.ADVANCE

  module.operation.walk(visit)
  return json.dumps({"dpi_functions": functions})


with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  module = Module.parse('''
    module {
      sim.func.dpi @step(out result: i7, in %flag: i1, in %byte_value: i8,
                         in %half: i16, in %cycle: i32, in %wide: i63,
                         in %long_value: i64, in %extra: i65,
                         inout %state: i1024, return status: i64)
        attributes {verilogName = "step_c"}
      sim.func.dpi @signedness(in %signed_arg: si32, in %unsigned_arg: ui32)
      sim.func.dpi @finish()
      hw.module @Testbench() { hw.output }
    }
  ''')
  original = str(module)
  schema = json.loads(serialize_integer_dpi(module))
  assert str(module) == original
  functions = schema["dpi_functions"]
  assert [func["function"] for func in functions
         ] == ["step_c", "signedness", "finish"]
  assert functions[2]["arguments"] == []
  assert [arg["direction"] for arg in functions[0]["arguments"]] == [
      "out", "in", "in", "in", "in", "in", "in", "in", "inout", "return"
  ]
  assert [(arg["width"], arg["signed"]) for arg in functions[0]["arguments"]
         ] == [(7, False), (1, False), (8, True), (16, True), (32, True),
               (63, False), (64, True), (65, False), (1024, False), (64, True)]
  assert all(arg["signed"] for arg in functions[1]["arguments"])

  # CHECK: JSON state: width=1024 signed=False direction=inout
  state = functions[0]["arguments"][8]
  print(f"JSON state: width={state['width']} signed={state['signed']} "
        f"direction={state['direction']}")

  # Lower a copy and compare the actual emitted declarations with the JSON.
  lowered = Module.parse(original)
  PassManager.parse("builtin.module(lower-sim-to-sv)").run(lowered.operation)
  fragments = [
      FlatSymbolRefAttr.get(op.sym_name.value)
      for op in lowered.body.operations
      if isinstance(op, emit.FragmentOp)
  ]
  testbench = next(
      op for op in lowered.body.operations if isinstance(op, hw.HWModuleOp))
  testbench.attributes["emit.fragments"] = ArrayAttr.get(fragments)
  lowered.operation.attributes["circt.loweringOptions"] = StringAttr.get(
      "disallowPortDeclSharing")
  output = io.StringIO()
  circt.export_verilog(lowered, output)
  print(output.getvalue())
  # CHECK: import "DPI-C" context function longint step_c(
  # CHECK-NEXT: output bit [6:0] result,
  # CHECK-NEXT: input bit flag,
  # CHECK-NEXT: input byte byte_value,
  # CHECK-NEXT: input shortint half,
  # CHECK-NEXT: input int cycle,
  # CHECK-NEXT: input bit [62:0] wide,
  # CHECK-NEXT: input longint long_value,
  # CHECK-NEXT: input bit [64:0] extra,
  # CHECK-NEXT: inout bit [1023:0] state
  # CHECK: import "DPI-C" context function void signedness(
  # CHECK-NEXT: input int signed_arg,
  # CHECK-NEXT: input int unsigned_arg
  # CHECK: import "DPI-C" context function void finish();
