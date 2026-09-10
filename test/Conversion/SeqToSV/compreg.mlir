// RUN: circt-opt %s --lower-seq-to-sv=lower-to-always-ff | FileCheck %s

// CHECK-LABEL: hw.module @basic(in %clk : i1, in %d : i8, out q : i8) {
// CHECK:         %[[REG:.*]] = sv.reg : !hw.inout<i8>
// CHECK:         %[[RD:.*]] = sv.read_inout %[[REG]] : !hw.inout<i8>
// CHECK:         sv.alwaysff(posedge %clk) {
// CHECK-NEXT:      sv.passign %[[REG]], %d : i8
// CHECK-NEXT:    }
// CHECK-NEXT:    hw.output %[[RD]] : i8
// CHECK-NEXT:  }
hw.module @basic(in %clk: !seq.clock, in %d: i8, out q: i8) {
  %q = seq.compreg %d, %clk : i8
  hw.output %q : i8
}

// CHECK-LABEL: hw.module @basicWithInit(in %clk : i1, in %d : i8, out q : i8) {
// CHECK:         sv.initial {
// CHECK:         }
// CHECK:         %[[CST:.*]] = hw.constant 19 : i8
// CHECK:         %[[REG:.*]] = sv.reg init %[[CST]] : !hw.inout<i8>
// CHECK:         %[[RD:.*]] = sv.read_inout %[[REG]] : !hw.inout<i8>
// CHECK:         sv.always posedge %clk {
// CHECK-NEXT:      sv.passign %[[REG]], %d : i8
// CHECK-NEXT:    }
// CHECK-NEXT:    hw.output %[[RD]] : i8
// CHECK-NEXT:  }
hw.module @basicWithInit(in %clk: !seq.clock, in %d: i8, out q: i8) {
  %init = seq.initial () {
    %cst = hw.constant 19 : i8
    seq.yield %cst : i8
  } : () -> !seq.immutable<i8>

  %q = seq.compreg %d, %clk initial %init : i8
  hw.output %q : i8
}
