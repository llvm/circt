// RUN: circt-opt %s  -export-verilog -verify-diagnostics | FileCheck %s

// Test bug in function type conversion
// CHECK-LABEL: InOutWire
module attributes {circt.loweringOptions = "disallowExpressionInliningInPorts"} {
  hw.module.extern @Bar(inout %a: i1, out b: i1)
  hw.module private @InOutWire() {
// CHECK: wire a;
    %a = sv.wire : !hw.inout<i1>
// CHECK: .a (a),
    %bar.b = hw.instance "bar" @Bar(a: %a: !hw.inout<i1>) -> (b: i1)
  }
}