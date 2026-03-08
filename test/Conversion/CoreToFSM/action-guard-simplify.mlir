// RUN: circt-opt -convert-core-to-fsm %s | FileCheck %s

// Test that action blocks do not redundantly re-check the guard condition.
// When the guard ensures an external input condition holds, the action block
// should use the simplified value directly instead of wrapping it in a mux.

// The module has:
// - "state" register: 1-bit FSM state
// - "counter" register: variable that increments when x is true (in state 0)
// Both state and counter transitions are guarded by the same condition (x).

// CHECK-LABEL: fsm.machine @action_guard_simplify
// CHECK-SAME: (%[[X:.*]]: i1) -> i2
// CHECK-SAME: attributes {initialState = "state_0"}
// CHECK: %[[COUNTER:.*]] = fsm.variable "counter" {initValue = 0 : i2} : i2
hw.module @action_guard_simplify(in %clk : !seq.clock, in %rst : i1, in %x : i1, out count : i2) {
    %c0_i2 = hw.constant 0 : i2
    %c1_i2 = hw.constant 1 : i2
    %false = hw.constant false
    %true = hw.constant true

    // State logic: in state 0, if x then go to state 1; in state 1, go to 0.
    %state_is_0 = comb.xor %state, %true : i1
    %go_to_1 = comb.and %x, %state_is_0 : i1
    %stay_at_0 = comb.and %state_is_0, %not_x : i1
    %not_x = comb.xor %x, %true : i1
    %next_state = comb.mux %stay_at_0, %false, %go_to_1 : i1
    %state = seq.compreg name "state" %next_state, %clk reset %rst, %false : i1

    // Counter logic: same condition as state 0->1 transition
    %counter_plus_1 = comb.add %counter, %c1_i2 : i2
    %not_go = comb.xor %go_to_1, %true : i1
    %next_counter = comb.mux %not_go, %counter, %counter_plus_1 : i2
    %counter = seq.compreg name "counter" %next_counter, %clk reset %rst, %c0_i2 : i2

    hw.output %counter : i2
}

// State 0: guard for 0->1 is x; action should just be counter+1 (no mux on x).
// CHECK: fsm.state @state_0 output {
// CHECK:   fsm.output %[[COUNTER]] : i2
// CHECK: } transitions {
// CHECK:   fsm.transition @state_1 guard {
// CHECK:     fsm.return %[[X]]
// CHECK:   } action {
// The action should NOT contain a comb.mux on %[[X]].
// It should directly compute counter + 1.
// CHECK:     %[[C1:.*]] = hw.constant 1 : i2
// CHECK:     %[[SUM:.*]] = comb.add %[[COUNTER]], %[[C1]] : i2
// CHECK:     fsm.update %[[COUNTER]], %[[SUM]] : i2
// CHECK:   }
// CHECK:   fsm.transition @state_0 guard {
// CHECK:   } action {
// When the guard is !x, counter should stay unchanged.
// CHECK:     fsm.update %[[COUNTER]], %[[COUNTER]] : i2
// CHECK:   }

// -----

// Test that complex guard expressions (e.g., icmp eq (add x, 1), 2) are also
// propagated into action blocks. The guard computes a compound condition;
// the action should not duplicate it inside a mux.

// CHECK-LABEL: fsm.machine @action_guard_complex
// CHECK-SAME: (%[[X2:.*]]: i2) -> i2
// CHECK-SAME: attributes {initialState = "state_0"}
// CHECK: %[[COUNTER2:.*]] = fsm.variable "counter" {initValue = 0 : i2} : i2
hw.module @action_guard_complex(in %clk : !seq.clock, in %rst : i1, in %x : i2, out count : i2) {
    %c0_i2 = hw.constant 0 : i2
    %c1_i2 = hw.constant 1 : i2
    %c2_i2 = hw.constant 2 : i2
    %false = hw.constant false
    %true = hw.constant true

    // Complex condition: x + 1 == 2
    %x_plus_1 = comb.add %x, %c1_i2 : i2
    %condition = comb.icmp eq %x_plus_1, %c2_i2 : i2

    // State logic
    %state_is_0 = comb.xor %state, %true : i1
    %go_to_1 = comb.and %condition, %state_is_0 : i1
    %not_condition = comb.xor %condition, %true : i1
    %stay_at_0 = comb.and %state_is_0, %not_condition : i1
    %next_state = comb.mux %stay_at_0, %false, %go_to_1 : i1
    %state = seq.compreg name "state" %next_state, %clk reset %rst, %false : i1

    // Counter logic: mux conditioned on the same complex expression
    %counter_plus_1 = comb.add %counter, %c1_i2 : i2
    %not_go = comb.xor %go_to_1, %true : i1
    %next_counter = comb.mux %not_go, %counter, %counter_plus_1 : i2
    %counter = seq.compreg name "counter" %next_counter, %clk reset %rst, %c0_i2 : i2

    hw.output %counter : i2
}

// State 0: guard for 0->1 is icmp eq (add x, 1), 2.
// Action should just be counter+1 (no mux on the complex condition).
// CHECK: fsm.state @state_0 output {
// CHECK:   fsm.output %[[COUNTER2]] : i2
// CHECK: } transitions {
// CHECK:   fsm.transition @state_1 guard {
// CHECK:   } action {
// CHECK-NOT: comb.mux
// CHECK-NOT: comb.icmp
// CHECK:     %[[C1_2:.*]] = hw.constant 1 : i2
// CHECK:     %[[SUM2:.*]] = comb.add %[[COUNTER2]], %[[C1_2]] : i2
// CHECK:     fsm.update %[[COUNTER2]], %[[SUM2]] : i2
// CHECK:   }
// CHECK:   fsm.transition @state_0 guard {
// CHECK:   } action {
// When the guard is icmp ne (...), counter should stay unchanged.
// CHECK:     fsm.update %[[COUNTER2]], %[[COUNTER2]] : i2
// CHECK:   }
