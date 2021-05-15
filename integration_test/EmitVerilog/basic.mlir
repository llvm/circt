// REQUIRES: verilator
// RUN: circt-translate %s -export-verilog -verify-diagnostics > %t1.sv
// RUN: circt-rtl-sim.py %t1.sv --cycles 8 2>&1 | FileCheck %s

module {
  // The HW dialect doesn't have any sequential constructs yet. So don't do
  // much.
  rtl.module @top(%clk: i1, %rstn: i1) {
    %c1 = rtl.instance "aaa" @AAA () : () -> (i1)
    %c1Shl = rtl.instance "shl" @shl (%c1) : (i1) -> (i1)
    sv.always posedge %clk {
      sv.fwrite "tick\n"
    }
  }

  rtl.module @AAA() -> (%f: i1) {
    %z = rtl.constant 1 : i1
    rtl.output %z : i1
  }

  rtl.module @shl(%a: i1) -> (%b: i1) {
    %0 = comb.shl %a, %a : i1
    rtl.output %0 : i1
  }
}

// CHECK:      [driver] Starting simulation
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: [driver] Ending simulation at tick #25
