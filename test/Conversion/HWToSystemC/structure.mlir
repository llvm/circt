// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics %s | FileCheck %s

// CHECK: emitc.include <"systemc.h">

// CHECK-LABEL: systemc.module @emptyModule ()
hw.module @emptyModule () -> () {}

// CHECK-LABEL: systemc.module @onlyInputs (%a: !systemc.in<i32>, %b: !systemc.in<i32>)
hw.module @onlyInputs (%a: i32, %b: i32) -> () {}

// CHECK-LABEL: systemc.module @onlyOutputs (%sum: !systemc.out<i32>)
hw.module @onlyOutputs () -> (sum: i32) {
  // CHECK-NEXT: systemc.ctor {
  // CHECK-NEXT:   systemc.method %innerLogic
  // CHECK-NEXT: }
  // CHECK-NEXT: %innerLogic = systemc.func  {
  // CHECK-NEXT:   %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT:   systemc.signal.write %sum, %c0_i32 : !systemc.out<i32>
  // CHECK-NEXT: }
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @adder (%a: !systemc.in<i32>, %b: !systemc.in<i32>, %sum: !systemc.out<i32>)
hw.module @adder (%a: i32, %b: i32) -> (sum: i32) {
  // CHECK-NEXT: systemc.ctor {
  // CHECK-NEXT:   systemc.method %innerLogic
  // CHECK-NEXT: }
  // CHECK-NEXT: %innerLogic = systemc.func  {
  // CHECK-NEXT:   [[A:%.*]] = systemc.signal.read %a : !systemc.in<i32>
  // CHECK-NEXT:   [[B:%.*]] = systemc.signal.read %b : !systemc.in<i32>
  // CHECK-NEXT:   [[RES:%.*]] = comb.add [[A]], [[B]] : i32
  // CHECK-NEXT:   systemc.signal.write %sum, [[RES]] : !systemc.out<i32>
  // CHECK-NEXT: }
  %0 = comb.add %a, %b : i32
  hw.output %0 : i32
// CHECK-NEXT: }
}

// CHECK-LABEL: systemc.module private @moduleVisibility
hw.module private @moduleVisibility () -> () {}

// CHECK-LABEL: systemc.module @argAttrs (%port0: !systemc.in<i32> {hw.attrname = "sometext"}, %port1: !systemc.in<i32>, %out0: !systemc.out<i32>)
hw.module @argAttrs (%port0: i32 {hw.attrname = "sometext"}, %port1: i32) -> (out0: i32) {
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @resultAttrs (%port0: !systemc.in<i32>, %out0: !systemc.out<i32> {hw.attrname = "sometext"})
hw.module @resultAttrs (%port0: i32) -> (out0: i32 {hw.attrname = "sometext"}) {
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @submodule
hw.module @submodule (%in0: i16, %in1: i32) -> (out0: i16, out1: i32, out2: i64) {
  %0 = hw.constant 0 : i64
  hw.output %in0, %in1, %0 : i16, i32, i64
}

// CHECK-LABEL:  systemc.module @instanceLowering (%port0: !systemc.in<i32>, %out0: !systemc.out<i16>, %out1: !systemc.out<i32>, %out2: !systemc.out<i64>) {
hw.module @instanceLowering (%port0: i32) -> (out0: i16, out1: i32, out2: i64) {
// CHECK-NEXT:    %inst1 = systemc.instance.decl  @submodule : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>
// CHECK-NEXT:    %inst1_in0 = systemc.signal  : !systemc.signal<i16>
// CHECK-NEXT:    %inst1_out0 = systemc.signal  : !systemc.signal<i16>
// CHECK-NEXT:    %inst1_out1 = systemc.signal  : !systemc.signal<i32>
// CHECK-NEXT:    %inst2 = systemc.instance.decl  @submodule : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>
// CHECK-NEXT:    %inst2_out2 = systemc.signal  : !systemc.signal<i64>
// CHECK-NEXT:    systemc.ctor {
// CHECK-NEXT:      systemc.method [[UPDATEFUNC:%.+]]
// CHECK-NEXT:      systemc.instance.bind_port %inst1["in0"] to %inst1_in0 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.signal<i16>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["in1"] to %port0 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.in<i32>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["out0"] to %inst1_out0 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.signal<i16>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["out1"] to %inst1_out1 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.signal<i32>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["out2"] to %out2 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.out<i64>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["in0"] to %inst1_out0 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.signal<i16>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["in1"] to %inst1_out1 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.signal<i32>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["out0"] to %out0 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.out<i16>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["out1"] to %out1 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.out<i32>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["out2"] to %inst2_out2 : !systemc.module<submodule(in0: !systemc.in<i16>, in1: !systemc.in<i32>, out0: !systemc.out<i16>, out1: !systemc.out<i32>, out2: !systemc.out<i64>)>, !systemc.signal<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    [[UPDATEFUNC]] = systemc.func  {
// CHECK-NEXT:      systemc.signal.read %port0 : !systemc.in<i32>
// CHECK-NEXT:      %c0_i16 = hw.constant 0 : i16
// CHECK-NEXT:      systemc.signal.write %inst1_in0, %c0_i16 : !systemc.signal<i16>
// CHECK-NEXT:      systemc.signal.read %inst1_out0 : !systemc.signal<i16>
// CHECK-NEXT:      systemc.signal.read %inst1_out1 : !systemc.signal<i32>
// CHECK-NEXT:      systemc.signal.read %inst2_out2 : !systemc.signal<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
  %0 = hw.constant 0 : i16
  %inst1.out0, %inst1.out1, %inst1.out2 = hw.instance "inst1" @submodule (in0: %0: i16, in1: %port0: i32) -> (out0: i16, out1: i32, out2: i64)
  %inst2.out0, %inst2.out1, %inst2.out2 = hw.instance "inst2" @submodule (in0: %inst1.out0: i16, in1: %inst1.out1: i32) -> (out0: i16, out1: i32, out2: i64)
  hw.output %inst2.out0, %inst2.out1, %inst1.out2 : i16, i32, i64
}

// CHECK-LABEL:  systemc.module @instanceLowering2
hw.module @instanceLowering2 () -> () {
// CHECK-NEXT: %inst1 = systemc.instance.decl @emptyModule : !systemc.module<emptyModule()>
// CHECK-NEXT: systemc.ctor {
// CHECK-NEXT:   systemc.method %
// CHECK-NEXT: }
// CHECK-NEXT: systemc.func {
// CHECK-NEXT: }
  hw.instance "inst1" @emptyModule () -> ()
// CHECK-NEXT: }
}
