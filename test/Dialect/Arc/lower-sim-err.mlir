// RUN: arcilator %s --disable-output --verify-diagnostics --split-input-file

func.func @model_not_found() {
    // This test uses the generic operation format to test the verifier, as this case cannot happen
    // when using the custom format.
    // expected-error @+1 {{entry block of body region must have the model instance as a single argument}}
    "arc.sim.instantiate"() ({
        ^entry:
    }) : () -> ()
    return
}

// -----

func.func @invalid_arg() {
    // This test uses the generic operation format to test the verifier, as this case cannot happen
    // when using the custom format.
    // expected-error @+1 {{entry block argument type is not a model instance}}
    "arc.sim.instantiate"() ({
        ^entry(%model: i32):
    }) : () -> ()
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @model_not_found() {
    // expected-error @+2 {{failed to legalize}}
    // expected-error @+1 {{model not found}}
    arc.sim.instantiate @unknown as %model {}
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @port_not_found() {
    arc.sim.instantiate @id as %model {
        // expected-error @+2 {{failed to legalize}}
        // expected-error @+1 {{port not found}}
        %res = arc.sim.get_port %model, "unknown" : i8, !arc.sim.instance<"id">
        arc.sim.emit "use", %res : i8
    }
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @port_wrong_size() {
    arc.sim.instantiate @id as %model {
        // expected-error @+2 {{failed to legalize}}
        // expected-error @+1 {{expected port of width 8, got 16}}
        %res = arc.sim.get_port %model, "i" : i16, !arc.sim.instance<"id">
        arc.sim.emit "use", %res : i16
    }
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @input_not_found() {
    %v = arith.constant 24 : i8
    arc.sim.instantiate @id as %model {
        // expected-error @+2 {{failed to legalize}}
        // expected-error @+1 {{input not found}}
        arc.sim.set_input %model, "unknown" = %v : i8, !arc.sim.instance<"id">
    }
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @set_port_wrong_size() {
    %v = arith.constant 24 : i16
    arc.sim.instantiate @id as %model {
        // expected-error @+2 {{failed to legalize}}
        // expected-error @+1 {{expected input of width 8, got 16}}
        arc.sim.set_input %model, "i" = %v : i16, !arc.sim.instance<"id">
    }
    return
}

// -----

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

func.func @set_port_not_input() {
    %v = arith.constant 24 : i8
    arc.sim.instantiate @id as %model {
        // expected-error @+2 {{failed to legalize}}
        // expected-error @+1 {{provided port is not an input port}}
        arc.sim.set_input %model, "o" = %v : i8, !arc.sim.instance<"id">
    }
    return
}
