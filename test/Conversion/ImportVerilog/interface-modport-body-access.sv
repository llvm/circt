// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test for in-body access to modport members of a modport-typed interface port.
// The receiving module reads `bus.x` (modport input) and writes `bus.y`
// (modport output). Slang resolves these as HierarchicalValueExpressions whose
// symbol is the ModportPortSymbol; ImportVerilog must resolve them through the
// flattened interface port instead of registering them as cross-instance
// hierPath inputs (which would add spurious unfilled module ports).

interface IfaceModportBody;
  logic [7:0] x;
  logic [7:0] y;
  modport mp (input x, output y);
endinterface

// CHECK-LABEL: moore.module private @ModportBodyUse(in %bus_x : !moore.l8, out bus_y : !moore.l8) {
// CHECK:         moore.output %bus_x : !moore.l8
// CHECK:       }
module ModportBodyUse(IfaceModportBody.mp bus);
  assign bus.y = bus.x;
endmodule

// CHECK-LABEL: moore.module @TopModportBodyUse() {
// CHECK:         %ifc_x = moore.variable : <l8>
// CHECK:         %ifc_y = moore.assigned_variable %dut.bus_y : l8
// CHECK:         [[X:%.+]] = moore.read %ifc_x : <l8>
// CHECK:         %dut.bus_y = moore.instance "dut" @ModportBodyUse(bus_x: [[X]]: !moore.l8) -> (bus_y: !moore.l8)
// CHECK:         moore.output
// CHECK:       }
module TopModportBodyUse;
  IfaceModportBody ifc();
  ModportBodyUse dut(.bus(ifc));
endmodule
