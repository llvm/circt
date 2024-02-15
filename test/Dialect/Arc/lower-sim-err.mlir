// RUN: arcilator %s --disable-output --verify-diagnostics --split-input-file

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @model_not_found() {
    // expected-error @+2 {{failed to legalize}}
    // expected-error @+1 {{model not found}}
    %model = arc.sim.instantiate : !arc.sim.instance<"unknown">
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @port_not_found() {
    %model = arc.sim.instantiate : !arc.sim.instance<"id">
    // expected-error @+2 {{failed to legalize}}
    // expected-error @+1 {{port not found}}
    arc.sim.get_port %model, "unknown" : i8, !arc.sim.instance<"id">
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @port_wrong_size() {
    %model = arc.sim.instantiate : !arc.sim.instance<"id">
    // expected-error @+2 {{failed to legalize}}
    // expected-error @+1 {{expected port of width 8, got 16}}
    arc.sim.get_port %model, "i" : i16, !arc.sim.instance<"id">
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @input_not_found() {
    %model = arc.sim.instantiate : !arc.sim.instance<"id">
    %v = arith.constant 24 : i8

    // expected-error @+2 {{failed to legalize}}
    // expected-error @+1 {{input not found}}
    arc.sim.set_input %model, "unknown" = %v : i8, !arc.sim.instance<"id">
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @set_port_wrong_size() {
    %model = arc.sim.instantiate : !arc.sim.instance<"id">
    %v = arith.constant 24 : i16

    // expected-error @+2 {{failed to legalize}}
    // expected-error @+1 {{expected input of width 8, got 16}}
    arc.sim.set_input %model, "i" = %v : i16, !arc.sim.instance<"id">
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @set_port_not_input() {
    %model = arc.sim.instantiate : !arc.sim.instance<"id">
    %v = arith.constant 24 : i8

    // expected-error @+2 {{failed to legalize}}
    // expected-error @+1 {{provided port is not an input port}}
    arc.sim.set_input %model, "o" = %v : i8, !arc.sim.instance<"id">
    return
}
