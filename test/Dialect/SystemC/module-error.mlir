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

// expected-error @+1 {{expected 'sc_in', 'sc_out', or 'sc_inout'}}
systemc.module @parserTest (sc_invalid %arg: i32) { }
