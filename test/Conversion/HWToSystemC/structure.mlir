// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics %s | FileCheck %s

// CHECK: emitc.include <"systemc">

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
