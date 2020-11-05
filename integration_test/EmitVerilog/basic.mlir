// REQUIRES: verilator
// RUN: circt-translate %s -emit-verilog -verify-diagnostics > %t1.sv
// RUN: verilator --cc --top-module main -sv %t1.sv --build --exe %S/driver.cpp
// RUN: ./obj_dir/Vmain --cycles 8 2>&1 | FileCheck %s

module {
  // The RTL dialect doesn't have any sequential constructs yet. So don't do
  // much.
  rtl.module @main(%clk: i1, %rstn: i1) {
    %c1 = rtl.instance "aaa" @AAA () : () -> (i1)
    %c1Shl = rtl.instance "shl" @shl (%c1) : (i1) -> (i1)
    sv.alwaysat_posedge %clk {
      sv.fwrite "tick\n"
    }
  }

  rtl.module @AAA() -> (i1 {rtl.name = "f"}) {
    %z = rtl.constant ( 1 : i1 ) : i1
    rtl.output %z : i1
  }

  rtl.module @shl(%a: i1) -> (i1 {rtl.name = "b"}) {
    %0 = rtl.shl %a, %a : i1
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
// CHECK-NEXT: [driver] Ending simulation at tick #17
