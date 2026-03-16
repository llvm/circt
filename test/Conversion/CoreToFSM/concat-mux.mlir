// RUN: circt-opt -convert-core-to-fsm %s | FileCheck %s

// Regression test: when next-state logic is a mux with one constant branch
// and one comb.concat branch, the BFS must not lose the constant's value.
// Without the fix, generateConcatenatedValues overwrote previously accumulated
// possible values via std::move, producing only 2 states instead of 3.

// The next-state logic is: mux(b, 3, concat(a, 0))
//   b=1 -> next = 3 (0b011)
//   b=0 -> next = concat(a, 00) = {0, 4}
// The full set of reachable values is {0, 3, 4}, giving 3 states.

// CHECK-LABEL: fsm.machine @concat_mux
// CHECK: fsm.state @state_0
// CHECK: fsm.state @state_3
// CHECK: fsm.state @state_4
hw.module @concat_mux(in %clk : !seq.clock, in %rst : i1, in %a : i1, in %b : i1) {
    %c0_i3 = hw.constant 0 : i3
    %c3_i3 = hw.constant 3 : i3
    %c0_i2 = hw.constant 0 : i2
    %state = seq.compreg sym @state name "state" %next, %clk reset %rst, %c0_i3 : i3
    %concat_val = comb.concat %a, %c0_i2 : i1, i2
    %next = comb.mux %b, %c3_i3, %concat_val : i3
}
