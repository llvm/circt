// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: systemc.module @adder (sc_in %a: i32, sc_in %b: i32, sc_out %sum: i32)
hw.module @adder (%a: i32, %b: i32) -> (sum: i32) {
    // CHECK-NEXT: [[RES:%.*]] = comb.add
    %0 = comb.add %a, %b : i32
    // CHECK-NEXT: systemc.alias %sum, [[RES]] : i32
    hw.output %0 : i32
}
