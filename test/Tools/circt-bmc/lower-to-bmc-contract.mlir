// Test that lower-to-bmc erases leftover hw.module ops after absorbing the
// target module (e.g., apply-mode modules from contract lowering).

// RUN: circt-opt --lower-to-bmc="top-module=Mul9_CheckContract_0 bound=10" %s | FileCheck %s
// RUN: circt-opt --lower-to-bmc="top-module=ShiftLeft_CheckContract_0 bound=10" %s | FileCheck %s --check-prefix=SHIFT

// All hw.module ops should be erased after lower-to-bmc absorbs the target.
// CHECK-NOT: hw.module

// The check module should be absorbed into a verif.bmc op and replaced by a
// func.func of the same name.
// CHECK:  func.func @Mul9_CheckContract_0() {
// CHECK:    [[BMC:%.+]] = verif.bmc bound 20 num_regs 0 initial_values [] init {
// CHECK:    } loop {
// CHECK:    } circuit {
// CHECK:    ^bb0([[A:%.+]]: i42):
// CHECK:      [[C9:%.+]] = hw.constant 9 : i42
// CHECK:      [[C3:%.+]] = hw.constant 3 : i42
// CHECK:      [[SHL:%.+]] = comb.shl [[A]], [[C3]]
// CHECK:      [[ADD:%.+]] = comb.add [[A]], [[SHL]]
// CHECK:      [[MUL:%.+]] = comb.mul [[A]], [[C9]]
// CHECK:      [[EQ:%.+]] = comb.icmp eq [[ADD]], [[MUL]]
// CHECK:      verif.assert [[EQ]]
// CHECK:    }

// All hw.module ops should be erased.
// SHIFT-NOT: hw.module

// Test with require (precondition) + ensure (postcondition).
// SHIFT:  func.func @ShiftLeft_CheckContract_0() {
// SHIFT:    [[BMC:%.+]] = verif.bmc bound 20 num_regs 0 initial_values [] init {
// SHIFT:    } loop {
// SHIFT:    } circuit {
// SHIFT:    ^bb0([[B:%.+]]: i8, [[A:%.+]]: i8):
// SHIFT:      verif.assume
// SHIFT:      verif.assert
// SHIFT:    }

// Input: pre-lowered contract IR (after lower-contracts, lower-tests,
// flatten-modules, and externalize-registers).
//
// hw.module @Mul9 is the apply-mode module — it has verif.assume and
// verif.symbolic_value but no assertions.  lower-to-bmc targets
// @Mul9_CheckContract_0 and should erase @Mul9 afterward.

hw.module @Mul9(in %a : i42, out z : i42) attributes {initial_values = [], num_regs = 0 : i32} {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42
  %2 = verif.symbolic_value : i42
  %3 = comb.mul %a, %c9_i42 : i42
  %4 = comb.icmp eq %2, %3 : i42
  verif.assume %4 : i1
  hw.output %2 : i42
}

hw.module @Mul9_CheckContract_0(in %symbolic_value_0 : i42) attributes {initial_values = [], num_regs = 0 : i32} {
  %c9_i42 = hw.constant 9 : i42
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %symbolic_value_0, %c3_i42 : i42
  %1 = comb.add %symbolic_value_0, %0 : i42
  %2 = comb.mul %symbolic_value_0, %c9_i42 : i42
  %3 = comb.icmp eq %1, %2 : i42
  verif.assert %3 : i1
  hw.output
}

// hw.module @ShiftLeft is the apply-mode module for a contract with both
// require (precondition) and ensure (postcondition).

hw.module @ShiftLeft(in %a : i8, in %b : i8, out z : i8) attributes {initial_values = [], num_regs = 0 : i32} {
  %c4_i8 = hw.constant 4 : i8
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %0 = comb.extract %b from 2 : (i8) -> i1
  %1 = comb.extract %b from 1 : (i8) -> i1
  %2 = comb.extract %b from 0 : (i8) -> i1
  %3 = comb.shl %a, %c4_i8 : i8
  %4 = comb.mux %0, %3, %a : i8
  %5 = comb.shl %4, %c2_i8 : i8
  %6 = comb.mux %1, %5, %4 : i8
  %7 = comb.shl %6, %c1_i8 : i8
  %8 = comb.mux %2, %7, %6 : i8
  %9 = verif.symbolic_value : i8
  %c8_i8 = hw.constant 8 : i8
  %10 = comb.icmp ult %b, %c8_i8 : i8
  verif.assert %10 : i1
  %11 = comb.shl %a, %b : i8
  %12 = comb.icmp eq %9, %11 : i8
  verif.assume %12 : i1
  hw.output %9 : i8
}

hw.module @ShiftLeft_CheckContract_0(in %symbolic_value_0 : i8, in %symbolic_value_1 : i8) attributes {initial_values = [], num_regs = 0 : i32} {
  %0 = comb.extract %symbolic_value_0 from 0 : (i8) -> i1
  %c1_i8 = hw.constant 1 : i8
  %1 = comb.extract %symbolic_value_0 from 1 : (i8) -> i1
  %c2_i8 = hw.constant 2 : i8
  %2 = comb.extract %symbolic_value_0 from 2 : (i8) -> i1
  %c4_i8 = hw.constant 4 : i8
  %3 = comb.shl %symbolic_value_1, %c4_i8 : i8
  %4 = comb.mux %2, %3, %symbolic_value_1 : i8
  %5 = comb.shl %4, %c2_i8 : i8
  %6 = comb.mux %1, %5, %4 : i8
  %7 = comb.shl %6, %c1_i8 : i8
  %8 = comb.mux %0, %7, %6 : i8
  %c8_i8 = hw.constant 8 : i8
  %9 = comb.icmp ult %symbolic_value_0, %c8_i8 : i8
  verif.assume %9 : i1
  %10 = comb.shl %symbolic_value_1, %symbolic_value_0 : i8
  %11 = comb.icmp eq %8, %10 : i8
  verif.assert %11 : i1
  hw.output
}
