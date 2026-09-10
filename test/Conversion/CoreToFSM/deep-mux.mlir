// RUN: circt-opt -convert-core-to-fsm %s | FileCheck %s

// Test that the pass handles deep mux trees without stack overflow.
// The isConstantLike and addPossibleValues functions use iterative traversal
// with visited sets to handle potential cycles and prevent stack overflow.

// CHECK-LABEL: fsm.machine @deep_mux_fsm
hw.module @deep_mux_fsm(in %clk : !seq.clock, in %rst : i1, in %sel : i1, out output : i1) {
    %c0_i2 = hw.constant 0 : i2
    %c1_i2 = hw.constant 1 : i2
    %c2_i2 = hw.constant 2 : i2
    %c3_i2 = hw.constant 3 : i2

    // State register
    %state = seq.compreg sym @state name "state" %next_state, %clk reset %rst, %c0_i2 : i2

    // Create a deep mux tree for next state computation
    // This exercises the iterative traversal in isConstantLike and addPossibleValues
    %mux1 = comb.mux %sel, %c1_i2, %c0_i2 : i2
    %mux2 = comb.mux %sel, %c2_i2, %mux1 : i2
    %mux3 = comb.mux %sel, %c3_i2, %mux2 : i2
    %mux4 = comb.mux %sel, %mux1, %mux3 : i2
    %mux5 = comb.mux %sel, %mux2, %mux4 : i2
    %mux6 = comb.mux %sel, %mux3, %mux5 : i2
    %mux7 = comb.mux %sel, %mux4, %mux6 : i2
    %mux8 = comb.mux %sel, %mux5, %mux7 : i2

    // Use the deep mux result for next state
    %next_state = comb.mux %sel, %mux8, %state : i2

    // Output based on state
    %c2_i2_out = hw.constant 2 : i2
    %is_2 = comb.icmp eq %state, %c2_i2_out : i2
    hw.output %is_2 : i1
}

// CHECK: fsm.state @state_0
// CHECK: fsm.state @state_2
