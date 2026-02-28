// RUN: circt-opt -hw-bypass-inner-symbols %s | FileCheck %s

// CHECK-LABEL: hw.module @BypassMultipleWires
// CHECK-SAME: in %a : i8, in %b : i8, out out : i8
hw.module @BypassMultipleWires(in %a: i8, in %b: i8, out out: i8) {
  // Multiple wires with symbols should all be bypassed
  // CHECK: %wire1 = hw.wire %a sym @wire1 : i8
  // CHECK: %wire2 = hw.wire %b sym @wire2 : i8
  // CHECK: %[[ADD:.+]] = comb.add bin %a, %b
  %wire1 = hw.wire %a sym @wire1 : i8
  %wire2 = hw.wire %b sym @wire2 : i8
  %result = comb.add bin %wire1, %wire2 : i8

  // CHECK: hw.output %[[ADD]]
  hw.output %result : i8
}

// CHECK-LABEL: hw.module @BypassInputPortWithSym
// CHECK-SAME: in %a : i32, in %b : i32, out out : i32
hw.module @BypassInputPortWithSym(in %a: i32 {hw.exportPort = #hw<innerSym@in>}, in %b: i32, out out: i32 {hw.exportPort = #hw<innerSym@out>}) {
  // Port symbols are moved to wires, then wires are bypassed
  // CHECK: %[[WIRE_IN:.+]] = hw.wire %a sym @in : i32
  // CHECK: %[[ADD:.+]] = comb.add bin %a, %b
  // CHECK: %[[WIRE_OUT:.+]] = hw.wire %[[ADD]] sym @out : i32
  %result = comb.add bin %a, %b : i32

  // CHECK: hw.output %[[ADD]]
  hw.output %result : i32
}
