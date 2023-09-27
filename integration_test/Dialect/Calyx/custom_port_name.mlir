// This test checks that the custom port name gets propagated all the way down to Verilog
// RUN: hlstool --calyx-hw --ir %s | FileCheck %s

// CHECK: hw.module @control
// CHECK: hw.module @main(input %a: i32, input %b: i32, input %clk: i1, input %reset: i1, input %go: i1, output out: i1, done: i1)
// CHECK: hw.instance "controller" @control
func.func @main(%arg0 : i32 {calyx.port_name = "a"}, %arg1 : i32 {calyx.port_name = "b"}) -> (i1 {calyx.port_name = "out"}) {
  %0 = arith.cmpi slt, %arg0, %arg1 : i32
  return %0 : i1
}
