// RUN: circt-opt -convert-core-to-fsm -verify-diagnostics -split-input-file %s

// Test: Non-integer register type should produce an error.
// The pass only supports registers with integer types.
hw.module @non_integer_register(in %clk : !seq.clock, in %rst : i1, out output : i1) {
    %c0_i1 = hw.constant 0 : i1
    %zero_array = hw.aggregate_constant [0 : i8, 0 : i8] : !hw.array<2xi8>
    // expected-error @+1 {{FSM extraction only supports integer-typed registers}}
    %state = seq.compreg name "state" %state, %clk reset %rst, %zero_array : !hw.array<2xi8>
    hw.output %c0_i1 : i1
}

// -----

// Test: No state register found should produce an error
// The error is emitted at the module location since no state register was found.
// expected-error @+1 {{Cannot find state register in this FSM}}
hw.module @no_state_register(in %clk : !seq.clock, in %rst : i1, in %inp : i1, out output : i1) {
    %c0_i2 = hw.constant 0 : i2
    // Register without "state" in name and no --state-regs option
    %counter = seq.compreg name "counter" %next, %clk reset %rst, %c0_i2 : i2
    %next = comb.add %counter, %counter : i2
    hw.output %inp : i1
}

// -----

// Test: Register without constant reset value should produce an error
hw.module @non_constant_reset(in %clk : !seq.clock, in %rst : i1, in %inp : i2, out output : i1) {
    %c0_i1 = hw.constant 0 : i1
    // Using input as reset value instead of constant
    // expected-error @+1 {{cannot find defining constant for reset value of register}}
    %state = seq.compreg name "state" %state, %clk reset %rst, %inp : i2
    hw.output %c0_i1 : i1
}

// -----

// Test: Any register (state or variable) without constant reset value should produce an error.
// All registers must have constant reset values for FSM extraction.
hw.module @register_non_constant_reset(in %clk : !seq.clock, in %rst : i1, in %inp : i4, out output : i1) {
    %c0_i2 = hw.constant 0 : i2
    %c2_i2 = hw.constant 2 : i2
    %c1_i4 = hw.constant 1 : i4
    // State register with constant reset - this is fine
    %state = seq.compreg name "state" %next_state, %clk reset %rst, %c0_i2 : i2
    // Variable register with non-constant reset value - error
    // (caught by the initial reset value check for all registers)
    // expected-error @+1 {{cannot find defining constant for reset value of register}}
    %counter = seq.compreg name "counter" %next_counter, %clk reset %rst, %inp : i4
    %c0_i1 = hw.constant 0 : i1
    %inp_ext = comb.concat %c0_i1, %c0_i1 : i1, i1
    %next_state = comb.add %state, %inp_ext : i2
    %next_counter = comb.add %counter, %c1_i4 : i4
    %is_2 = comb.icmp eq %state, %c2_i2 : i2
    hw.output %is_2 : i1
}

// -----

// Test: Reset warning is emitted when reset signals are removed from FSM.
// The FSM dialect does not support reset signals, so these signals are excluded
// and a warning is emitted.
// expected-warning @+1 {{reset signals detected and removed from FSM}}
hw.module @async_reset_warning(in %clk : !seq.clock, in %rst : i1, in %inp : i1, out output : i1) {
    %c0_i2 = hw.constant 0 : i2
    %c2_i2 = hw.constant 2 : i2
    %state = seq.compreg name "state" %next_state, %clk reset %rst, %c0_i2 : i2
    %c0_i1 = hw.constant 0 : i1
    %add = comb.concat %c0_i1, %inp : i1, i1
    %next_state = comb.add %state, %add : i2
    %is_2 = comb.icmp eq %state, %c2_i2 : i2
    hw.output %is_2 : i1
}

// -----

// Test: Instance operations are not supported.
// The pass checks for hw.instance operations upfront and emits an error.
hw.module @instantiated_module(in %clk : !seq.clock, in %rst : i1, in %inp : i1, out output : i1) {
    %c0_i1 = hw.constant 0 : i1
    hw.output %c0_i1 : i1
}

hw.module @top_module(in %clk : !seq.clock, in %rst : i1, in %inp : i1, out output : i1) {
    // expected-error @+1 {{instance conversion is not yet supported}}
    %out = hw.instance "child" @instantiated_module(clk: %clk: !seq.clock, rst: %rst: i1, inp: %inp: i1) -> (output: i1)
    hw.output %out : i1
}

// -----

// Test: Cyclic combinational logic should produce an error.
// The pass cannot convert modules with cycles in purely combinational logic
// (e.g., cyclic muxes that don't involve registers).
// expected-error @+1 {{cannot convert module with combinational cycles to FSM}}
hw.module @cyclic_muxes(in %clk : !seq.clock, in %rst : i1, out output : i1) {
    %c0_i1 = hw.constant 0 : i1
    %state = seq.compreg name "state" %state, %clk reset %rst, %c0_i1 : i1
    // These two muxes form a combinational cycle
    %mux1 = comb.mux %rst, %mux2, %state : i1
    %mux2 = comb.mux %state, %mux1, %rst : i1
    hw.output %c0_i1 : i1
}
