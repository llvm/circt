// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{entry block must have 3 arguments to match function signature}}
"systemc.module"() ({
  ^bb0(%arg0: i4, %arg1: i32, %arg2: i4, %arg3: i8):
  }) {function_type = (i4, i32, i4) -> (), portDirections = #systemc.port_directions<[sc_out, sc_in, sc_out, sc_inout]>, portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of port directions}}
"systemc.module"() ({
  ^bb0(%arg0: i4, %arg1: i32, %arg2: i4, %arg3: i8):
  }) {function_type = (i4, i32, i4, i8) -> (), portDirections = #systemc.port_directions<[sc_in, sc_out, sc_inout]>, portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of port names}}
"systemc.module"() ({
  ^bb0(%arg0: i4, %arg1: i32, %arg2: i4, %arg3: i8):
  }) {function_type = (i4, i32, i4, i8) -> (), portDirections = #systemc.port_directions<[sc_out, sc_in, sc_out, sc_inout]>, portNames = ["port0", "port1", "port2"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of function results (always has to be 0)}}
"systemc.module"() ({
  ^bb0(%arg0: i4, %arg1: i32, %arg2: i4, %arg3: i8):
  }) {function_type = (i4, i32, i4, i8) -> (i1), portDirections = #systemc.port_directions<[sc_out, sc_in, sc_out, sc_inout]>, portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{port name must not be empty}}
"systemc.module"() ({
  ^bb0(%arg0: i4, %arg1: i32, %arg2: i4, %arg3: i8):
  }) {function_type = (i4, i32, i4, i8) -> (), portDirections = #systemc.port_directions<[sc_out, sc_in, sc_out, sc_inout]>, portNames = ["port0", "port1", "port2", ""], sym_name = "verifierTest"} : () -> ()

// -----

// expected-note @+1 {{in module '@verifierTest'}}
"systemc.module"() ({
  // expected-error @+2 {{redefines port name 'port2'}}
  // expected-note @+1 {{'port2' first defined here}}
  ^bb0(%arg0: i4, %arg1: i32, %arg2: i4, %arg3: i8):
  }) {function_type = (i4, i32, i4, i8) -> (), portDirections = #systemc.port_directions<[sc_out, sc_in, sc_out, sc_inout]>, portNames = ["port0", "port1", "port2", "port2"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{expected 'sc_in', 'sc_out', or 'sc_inout'}}
systemc.module @parserTest (sc_invalid %arg: i32) { }

// -----

// expected-note @+1 {{in module '@signalNameConflict'}}
systemc.module @signalNameConflict () {
  // expected-note @+1 {{'signal0' first defined here}}
  %0 = "systemc.signal"() {name = "signal0"} : () -> i32
  // expected-error @+1 {{redefines name 'signal0'}}
  %1 = "systemc.signal"() {name = "signal0"} : () -> i32
}

// -----

// expected-note @+2 {{in module '@signalNameConflictWithArg'}}
// expected-note @+1 {{'in' first defined here}}
systemc.module @signalNameConflictWithArg (sc_in %in: i32) {
  // expected-error @+1 {{redefines name 'in'}}
  %0 = "systemc.signal"() {name = "in"} : () -> i32
}

// -----

systemc.module @signalNameNotEmpty () {
  // expected-error @+1 {{'name' attribute must not be empty}}
  %0 = "systemc.signal"() {name = ""} : () -> i32
}

// -----

systemc.module @moduleDoesNotAccessNameBeforeExistanceVerified () {
  // expected-error @+1 {{requires attribute 'name'}}
  %0 = "systemc.signal"() {} : () -> i32
}

// -----

systemc.module @signalMustBeDirectChildOfModule () {
  systemc.ctor {
    // expected-error @+1 {{expects parent op 'systemc.module'}}
    %signal = systemc.signal : i32
  }
}
