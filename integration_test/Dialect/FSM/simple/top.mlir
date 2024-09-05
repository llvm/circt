// REQUIRES: verilator
// RUN: circt-opt %s --convert-fsm-to-sv --canonicalize --lower-seq-to-sv --export-split-verilog -o %t2.mlir
// RUN: circt-rtl-sim.py --compileargs="-I%T/.." top.sv %S/driver.cpp --no-default-driver | FileCheck %s
// CHECK: out: A
// CHECK: out: B
// CHECK: out: B
// CHECK: out: C
// CHECK: out: B
// CHECK: out: C
// CHECK: out: A

fsm.machine @top(%arg0: i16) -> (i8) attributes {initialState = "A"} {

  %c1 = hw.constant 1: i16
  %c0 = hw.constant 0: i16

  fsm.state @A output  {
    %c_0 = hw.constant 0 : i8
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @B
  }

  fsm.state @B output  {
    %c_1 = hw.constant 1 : i8
    fsm.output %c_1 : i8
  } transitions {
    fsm.transition @C guard {
      %j = comb.icmp eq %arg0, %c1 : i16
      fsm.return %j
    }
    fsm.transition @B guard {
      %g = comb.icmp ne %arg0, %c1 : i16
      fsm.return %g
    }
  }

  fsm.state @C output  {
    %c_2 = hw.constant 2 : i8
    fsm.output %c_2 : i8
  } transitions {
    fsm.transition @A guard {
      %g = comb.icmp eq %arg0, %c1 : i16
      fsm.return %g
    }
    fsm.transition @B guard {
      %g = comb.icmp ne %arg0, %c1 : i16
      fsm.return %g
    }
  }
}
