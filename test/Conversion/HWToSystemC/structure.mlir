// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: systemc.module @emptyModule ()
hw.module @emptyModule () -> () {}

// CHECK-LABEL: systemc.module @onlyInputs (sc_in %a: i32, sc_in %b: i32)
hw.module @onlyInputs (%a: i32, %b: i32) -> () {}

// CHECK-LABEL: systemc.module @onlyOutputs (sc_out %sum: i32)
hw.module @onlyOutputs () -> (sum: i32) {
  // CHECK-NEXT: systemc.ctor {
  // CHECK-NEXT:   systemc.method %innerLogic
  // CHECK-NEXT: }
  // CHECK-NEXT: %innerLogic = systemc.func  {
  // CHECK-NEXT:   %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT:   systemc.alias %sum, %c0_i32 : i32
  // CHECK-NEXT: }
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @adder (sc_in %a: i32, sc_in %b: i32, sc_out %sum: i32)
hw.module @adder (%a: i32, %b: i32) -> (sum: i32) {
  // CHECK-NEXT: systemc.ctor {
  // CHECK-NEXT:   systemc.method %innerLogic
  // CHECK-NEXT: }
  // CHECK-NEXT: %innerLogic = systemc.func  {
  // CHECK-NEXT:   [[RES:%.*]] = comb.add %a, %b : i32
  // CHECK-NEXT:   systemc.alias %sum, [[RES]] : i32
  // CHECK-NEXT: }
  %0 = comb.add %a, %b : i32
  hw.output %0 : i32
// CHECK-NEXT: }
}
