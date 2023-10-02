// RUN: circt-opt --lower-seq-to-sv %s | FileCheck %s

// CHECK:   hw.module @cg1(in %[[VAL_0:.*]] : i1, in %[[VAL_1:.*]] : i1, out gclk : i1) {
// CHECK:           %[[VAL_2:.*]] = sv.reg : !hw.inout<i1>
// CHECK:           sv.always  {
// CHECK:             %[[VAL_3:.*]] = hw.constant true
// CHECK:             %[[VAL_4:.*]] = comb.xor %[[VAL_0]], %[[VAL_3]] : i1
// CHECK:             sv.if %[[VAL_4]] {
// CHECK:               sv.passign %[[VAL_2]], %[[VAL_1]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = sv.read_inout %[[VAL_2]] : !hw.inout<i1>
// CHECK:           %[[VAL_6:.*]] = comb.and %[[VAL_0]], %[[VAL_5]] : i1
// CHECK:           hw.output %[[VAL_6]] : i1
// CHECK:         }
hw.module @cg1(in %clk : !seq.clock, in %enable : i1, out gclk : !seq.clock) {
    %0 = seq.clock_gate %clk, %enable
    hw.output %0 : !seq.clock
}

// CHECK:   hw.module @cg2(in %[[VAL_0:.*]] : i1, in %[[VAL_1:.*]] : i1, in %[[VAL_2:.*]] : i1, out gclk : i1) {
// CHECK:           %[[VAL_3:.*]] = comb.or %[[VAL_1]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_4:.*]] = sv.reg : !hw.inout<i1>
// CHECK:           sv.always  {
// CHECK:             %[[VAL_5:.*]] = hw.constant true
// CHECK:             %[[VAL_6:.*]] = comb.xor %[[VAL_0]], %[[VAL_5]] : i1
// CHECK:             sv.if %[[VAL_6]] {
// CHECK:               sv.passign %[[VAL_4]], %[[VAL_3]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = sv.read_inout %[[VAL_4]] : !hw.inout<i1>
// CHECK:           %[[VAL_8:.*]] = comb.and %[[VAL_0]], %[[VAL_7]] : i1
// CHECK:           hw.output %[[VAL_8]] : i1
// CHECK:         }
hw.module @cg2(in %clk : !seq.clock, in %enable : i1, in %test_enable : i1, out gclk : !seq.clock) {
    %0 = seq.clock_gate %clk, %enable, %test_enable
    hw.output %0 : !seq.clock
}
