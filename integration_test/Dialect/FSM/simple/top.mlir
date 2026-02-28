// REQUIRES: verilator
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-opt --pass-pipeline="builtin.module(convert-fsm-to-sv,canonicalize,lower-seq-to-sv,export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --compileargs="-I%t.dir/" %t.dir/top.sv %S/driver.cpp --no-default-driver | FileCheck %s
// CHECK: out: A
// CHECK: out: B
// CHECK: out: B
// CHECK: out: C
// CHECK: out: B
// CHECK: out: C
// CHECK: out: A

fsm.machine @top(%arg0: i1, %arg1: i1) -> (i8) attributes {initialState = "A"} {

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
      %g = comb.and %arg0, %arg1 : i1
      fsm.return %g
    }
  }

  fsm.state @C output  {
    %c_2 = hw.constant 2 : i8
    fsm.output %c_2 : i8
  } transitions {
    fsm.transition @A guard {
      %g = comb.and %arg0, %arg1 : i1
      fsm.return %g
    }
    fsm.transition @B
  }
}
