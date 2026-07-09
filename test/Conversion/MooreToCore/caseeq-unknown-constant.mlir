// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// Case equality against a constant carrying x/z bits folds to its static
// verdict in this two-valued lowering: runtime values never hold x/z, so an
// x/z constant bit can never case-match (IEEE 1800-2017 11.4.5). Previously
// the unknown bits were dropped from the converted constant, turning
// `v === 'x` into `v === 0` — the `$isunknown` payload shape, which then
// reported every all-zero sample as unknown.

// CHECK-LABEL: func.func @CaseEqUnknownConst
func.func @CaseEqUnknownConst(%arg0: !moore.l1, %arg1: !moore.l8) {
  %x1 = moore.constant bX : l1
  %mixed = moore.constant b1XX01ZZ0 : l8

  // `v === 1'bx` can never hold on two-valued runtime data.
  // CHECK: hw.constant false
  // CHECK-NOT: comb.icmp ceq
  %0 = moore.case_eq %arg0, %x1 : l1
  // CHECK: hw.constant true
  // CHECK-NOT: comb.icmp cne
  %1 = moore.case_ne %arg0, %x1 : l1
  // Constant operand order does not matter.
  // CHECK: hw.constant false
  %2 = moore.case_eq %x1, %arg0 : l1
  // Partially-unknown constants also force a static verdict: the x/z bit
  // positions can never match.
  // CHECK: hw.constant false
  %3 = moore.case_eq %arg1, %mixed : l8

  // Fully two-valued case comparisons keep the genuine comparison.
  // CHECK: comb.icmp ceq
  %k = moore.constant 42 : l8
  %4 = moore.case_eq %arg1, %k : l8
  return
}
