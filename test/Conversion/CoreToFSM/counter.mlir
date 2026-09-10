// RUN: circt-opt -convert-core-to-fsm %s | FileCheck %s

// Test RTL-to-FSM conversion with a counter that increments based on input.
// This test verifies that states and transitions are correctly extracted.
// The counter adds inp (0 or 1) to the current state value.

// CHECK-LABEL: fsm.machine @counter
// CHECK-SAME: (%[[INP:.*]]: i1) -> i1
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

// Verify states are created with correct outputs
// CHECK: fsm.state @state_0 output {
// CHECK:   %[[VAL:.*]] = hw.constant false
// CHECK:   fsm.output %[[VAL]] : i1
// CHECK: } transitions {
// From state_0: inp=1 -> state_1, inp=0 -> state_0
// CHECK:   fsm.transition @state_1 guard {
// CHECK:     fsm.return %[[INP]]
// CHECK:   }
// CHECK:   fsm.transition @state_0 guard {

// CHECK: fsm.state @state_1 output {
// CHECK:   %[[VAL1:.*]] = hw.constant false
// CHECK:   fsm.output %[[VAL1]] : i1
// CHECK: } transitions {
// From state_1: transitions to various states based on inp
// CHECK:   fsm.transition @state_{{[0-3]}} guard {

// CHECK: fsm.state @state_2 output {
// CHECK:   %[[VAL2:.*]] = hw.constant true
// CHECK:   fsm.output %[[VAL2]] : i1
// CHECK: } transitions {
// From state_2: transitions to various states based on inp
// CHECK:   fsm.transition @state_{{[0-3]}} guard {
