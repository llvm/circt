// RUN: circt-opt -split-input-file -test-cppcde -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: hw.module @MyAdder_32(%in0: i32, %in1: i32, %clk: i1) -> (out0: i32) {
// CHECK-NEXT:    %0 = comb.add %in0, %in1 : i32
// CHECK-NEXT:    %add_reg = seq.compreg sym @add_reg %0, %clk : i32
// CHECK-NEXT:    hw.output %add_reg : i32
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @MyESIAdder(%in0: !esi.channel<i32>, %in1: !esi.channel<i32>, %clk: i1) -> (out0: !esi.channel<i32>) {
// CHECK-NEXT:    %rawOutput, %valid = esi.unwrap.vr %in0, %ready : i32
// CHECK-NEXT:    %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %ready : i32
// CHECK-NEXT:    %myAdder.out0 = hw.instance "myAdder" @MyAdder_32(in0: %rawOutput: i32, in1: %rawOutput_0: i32, clk: %clk: i1) -> (out0: i32)
// CHECK-NEXT:    %0 = comb.and %valid, %valid_1 : i1
// CHECK-NEXT:    %chanOutput, %ready = esi.wrap.vr %myAdder.out0, %0 : i32
// CHECK-NEXT:    hw.output %chanOutput : !esi.channel<i32>
// CHECK-NEXT:  }

module {}
