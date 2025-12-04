// RUN: circt-opt -convert-core-to-fsm %s | FileCheck %s

// Test RTL-to-FSM conversion with a counter that increments based on input

// CHECK-LABEL: fsm.machine @counter
// CHECK-SAME: (%[[INP:.*]]: i1) -> (i1)
// CHECK-SAME: attributes {initialState = "state_0"}
hw.module @counter(in %clk : !seq.clock, in %rst : i1, in %inp : i1, out output : i1){
    %c0_i2 = hw.constant 0 : i2
    %c0_i1 = hw.constant 0 : i1
    %c2_i2 = hw.constant 2 : i2
    %true = hw.constant true
    %state = seq.compreg name "state" %next_state, %clk reset %rst, %c0_i2 : i2
    %add = comb.concat %c0_i1, %inp: i1, i1
    %next_state = comb.add %state, %add : i2
    %is_2 = comb.icmp eq %state, %c2_i2 : i2
    hw.output %is_2 : i1
}

// Verify that states are created
// CHECK: fsm.state @state_0 output {
// CHECK:   %[[VAL:.*]] = hw.constant false
// CHECK:   fsm.output %[[VAL]] : i1
// CHECK: } transitions {

// CHECK: fsm.state @state_1 output {
// CHECK:   fsm.output
// CHECK: } transitions {

// CHECK: fsm.state @state_2 output {
// CHECK:   %[[VAL:.*]] = hw.constant true
// CHECK:   fsm.output %[[VAL]] : i1
// CHECK: } transitions {
