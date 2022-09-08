// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics %s | FileCheck %s

// CHECK: emitc.include <"systemc.h">

// CHECK-LABEL: systemc.module @emptyModule ()
hw.module @emptyModule () -> () {}

// CHECK-LABEL: systemc.module @onlyInputs (%a: !systemc.in<!systemc.uint<32>>, %b: !systemc.in<!systemc.biguint<256>>, %c: !systemc.in<!systemc.bv<1024>>, %d: !systemc.in<i1>)
hw.module @onlyInputs (%a: i32, %b: i256, %c: i1024, %d: i1) -> () {}

// CHECK-LABEL: systemc.module @onlyOutputs (%sum: !systemc.out<!systemc.uint<32>>)
hw.module @onlyOutputs () -> (sum: i32) {
  // CHECK-NEXT: systemc.ctor {
  // CHECK-NEXT:   systemc.method %innerLogic
  // CHECK-NEXT: }
  // CHECK-NEXT: %innerLogic = systemc.func  {
  // CHECK-NEXT:   %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT:   [[CAST:%.+]] = systemc.convert %c0_i32 : (i32) -> !systemc.uint<32>
  // CHECK-NEXT:   systemc.signal.write %sum, [[CAST]] : !systemc.out<!systemc.uint<32>>
  // CHECK-NEXT: }
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @adder (%a: !systemc.in<!systemc.uint<32>>, %b: !systemc.in<!systemc.uint<32>>, %sum: !systemc.out<!systemc.uint<32>>)
hw.module @adder (%a: i32, %b: i32) -> (sum: i32) {
  // CHECK-NEXT: systemc.ctor {
  // CHECK-NEXT:   systemc.method %innerLogic
  // CHECK-NEXT: }
  // CHECK-NEXT: %innerLogic = systemc.func  {
  // CHECK-NEXT:   [[A:%.+]] = systemc.signal.read %a : !systemc.in<!systemc.uint<32>>
  // CHECK-NEXT:   [[AC:%.+]] = systemc.convert [[A]] : (!systemc.uint<32>) -> i32
  // CHECK-NEXT:   [[B:%.+]] = systemc.signal.read %b : !systemc.in<!systemc.uint<32>>
  // CHECK-NEXT:   [[BC:%.+]] = systemc.convert [[B]] : (!systemc.uint<32>) -> i32
  // CHECK-NEXT:   [[RES:%.*]] = comb.add [[AC]], [[BC]] : i32
  // CHECK-NEXT:   [[RESC:%.+]] = systemc.convert [[RES]] : (i32) -> !systemc.uint<32>
  // CHECK-NEXT:   systemc.signal.write %sum, [[RESC]] : !systemc.out<!systemc.uint<32>>
  // CHECK-NEXT: }
  %0 = comb.sub %a, %b : i32
  %1 = comb.add %0, %b : i32
  hw.output %1 : i32
// CHECK-NEXT: }
}

// CHECK-LABEL: systemc.module private @moduleVisibility
hw.module private @moduleVisibility () -> () {}

// CHECK-LABEL: systemc.module @argAttrs (%port0: !systemc.in<!systemc.uint<32>> {hw.attrname = "sometext"}, %port1: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>>)
hw.module @argAttrs (%port0: i32 {hw.attrname = "sometext"}, %port1: i32) -> (out0: i32) {
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @resultAttrs (%port0: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>> {hw.attrname = "sometext"})
hw.module @resultAttrs (%port0: i32) -> (out0: i32 {hw.attrname = "sometext"}) {
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}
