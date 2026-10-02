# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import io

import circt
from circt.dialects import hw, sv

from circt import ir

with ir.Context() as ctx, ir.Location.unknown() as loc:
  circt.register_dialects(ctx)
  ctx.allow_unregistered_dialects = True

  sv_attr = sv.SVAttributeAttr.get("fold", "false")
  print(f"sv_attr: {sv_attr} {sv_attr.name} {sv_attr.expression}")
  # CHECK: sv_attr: #sv.attribute<"fold" = "false"> fold false

  sv_attr = sv.SVAttributeAttr.get("no_merge")
  print(f"sv_attr: {sv_attr} {sv_attr.name} {sv_attr.expression}")
  # CHECK: sv_attr: #sv.attribute<"no_merge"> no_merge None

  i1 = ir.IntegerType.get_signless(1)
  i1_inout = hw.InOutType.get(i1)

  m = ir.Module.create()
  with ir.InsertionPoint(m.body):
    wire_op = sv.WireOp(i1_inout, "wire1")
    wire_op.attributes["sv.attributes"] = ir.ArrayAttr.get([sv_attr])
    print(wire_op)
    # CHECK: %wire1 = sv.wire {sv.attributes = [#sv.attribute<"no_merge">]} : !hw.inout<i1>

    reg_op = sv.RegOp(i1_inout, "reg1")
    reg_op.attributes["sv.attributes"] = ir.ArrayAttr.get([sv_attr])
    print(reg_op)
    # CHECK: %reg1 = sv.reg  {sv.attributes = [#sv.attribute<"no_merge">]} : !hw.inout<i1>

    package_op = sv.PackageExternOp("ExternalTypesSymbol",
                                    verilogName="ExternalTypes")
    with ir.InsertionPoint(package_op.body.blocks.append()):
      hw.TypedeclOp.create("word", i1)
    assert package_op.operation.verify()
    assert package_op.verilogName.value == "ExternalTypes"
    print(package_op)
    # CHECK: sv.package.extern @ExternalTypesSymbol {
    # CHECK-NEXT: hw.typedecl @word : i1
    # CHECK-NEXT: } {verilogName = "ExternalTypes"}

  # Renaming a private MLIR symbol must not rename its external Verilog package.
  m = ir.Module.parse("""
    sv.package.extern @types {
      hw.typedecl @word : i8
      hw.typedecl @State : !hw.enum<Idle, Busy>
    } {sym_visibility = "private", verilogName = "ExternalTypes"}
  """)
  package_op = m.body.operations[0]
  ir.SymbolTable.set_symbol_name(package_op, "renamed_types")
  consumer = ir.Module.parse("""
    hw.module @Consumer(
        in %word: !hw.typealias<@renamed_types::@word, i8>,
        out state: !hw.typealias<@renamed_types::@State, !hw.enum<Idle, Busy>>) {
      %idle = hw.enum.constant Idle : !hw.typealias<@renamed_types::@State, !hw.enum<Idle, Busy>>
      hw.output %idle : !hw.typealias<@renamed_types::@State, !hw.enum<Idle, Busy>>
    }
  """)
  m.body.append(consumer.body.operations[0])
  assert m.operation.verify()
  buffer = io.StringIO()
  circt.export_verilog(m, buffer)
  verilog = buffer.getvalue()
  assert "renamed_types::" not in verilog
  assert "package ExternalTypes;" not in verilog
  print(verilog)
  # CHECK-LABEL: module Consumer(
  # CHECK: input {{ *}}ExternalTypes::word word
  # CHECK: output ExternalTypes::State state
  # CHECK: assign state = ExternalTypes::State_Idle;
