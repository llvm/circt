// RUN: circt-synth %s --use-transformDialect | FileCheck %s

// ---- PAYLOAD: a tiny HW module with a duplicated comb.and that CSE should fold. ----
hw.module @dup(in %a: i1, in %b: i1, out o: i1) {
  %0 = comb.and %a, %b : i1
  %1 = comb.and %b, %a : i1
  %2 = comb.xor %0, %1 : i1
  hw.output %2 : i1
}

// ---- TRANSFORM SCRIPT: tells the interpreter to run the registered "cse" pass on the root. ----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_hw_module(%op: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    transform.match.operation_name %op ["hw.module"] : !transform.any_op
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %target = transform.collect_matching @match_hw_module in %root
      : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "cse" to %target
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// ---- ASSERTIONS ----
// After CSE, the two identical comb.and ops collapse into one, and the comb.xor
// uses that single value twice.
//
// CHECK-LABEL: hw.module @dup
// CHECK:         %[[AND:.+]] = comb.and %a, %b : i1
// CHECK-NOT:     comb.and
// CHECK:         %{{.+}} = comb.xor %[[AND]], %[[AND]] : i1
// CHECK:         hw.output

