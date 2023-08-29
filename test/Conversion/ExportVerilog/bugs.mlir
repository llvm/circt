// RUN: circt-opt %s  -export-verilog -verify-diagnostics | FileCheck %s

// Test bug in function type conversion
module attributes {circt.loweringOptions = "disallowExpressionInliningInPorts"} {
  hw.module.extern @Bar(%a: !hw.inout<i1>) -> (b: i1)
  hw.module private @CoreIPSubsystemWithPeripherals() {
    %a = sv.wire : !hw.inout<i1>
    %bar.b = hw.instance "bar" @Bar(a: %a: !hw.inout<i1>) -> (b: i1)
  }
}