// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{entry block must have 3 arguments to match function signature}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.sc_out<i4>, %arg1: !systemc.sc_in<i32>, %arg2: !systemc.sc_out<i4>, %arg3: !systemc.sc_inout<i8>):
  }) {function_type = (!systemc.sc_out<i4>, !systemc.sc_in<i32>, !systemc.sc_out<i4>) -> (), portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of port names}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.sc_out<i4>, %arg1: !systemc.sc_in<i32>, %arg2: !systemc.sc_out<i4>, %arg3: !systemc.sc_inout<i8>):
  }) {function_type = (!systemc.sc_out<i4>, !systemc.sc_in<i32>, !systemc.sc_out<i4>, !systemc.sc_inout<i8>) -> (), portNames = ["port0", "port1", "port2"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of function results (always has to be 0)}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.sc_out<i4>, %arg1: !systemc.sc_in<i32>, %arg2: !systemc.sc_out<i4>, %arg3: !systemc.sc_inout<i8>):
  }) {function_type = (!systemc.sc_out<i4>, !systemc.sc_in<i32>, !systemc.sc_out<i4>, !systemc.sc_inout<i8>) -> (i1), portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{port name must not be empty}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.sc_out<i4>, %arg1: !systemc.sc_in<i32>, %arg2: !systemc.sc_out<i4>, %arg3: !systemc.sc_inout<i8>):
  }) {function_type = (!systemc.sc_out<i4>, !systemc.sc_in<i32>, !systemc.sc_out<i4>, !systemc.sc_inout<i8>) -> (), portNames = ["port0", "port1", "port2", ""], sym_name = "verifierTest"} : () -> ()

// -----

// expected-note @+1 {{in module '@verifierTest'}}
"systemc.module"() ({
  // expected-error @+2 {{redefines port name 'port2'}}
  // expected-note @+1 {{'port2' first defined here}}
  ^bb0(%arg0: !systemc.sc_out<i4>, %arg1: !systemc.sc_in<i32>, %arg2: !systemc.sc_out<i4>, %arg3: !systemc.sc_inout<i8>):
  }) {function_type = (!systemc.sc_out<i4>, !systemc.sc_in<i32>, !systemc.sc_out<i4>, !systemc.sc_inout<i8>) -> (), portNames = ["port0", "port1", "port2", "port2"], sym_name = "verifierTest"} : () -> ()

// -----

"systemc.module"() ({
// expected-error @+1 {{module port must be of type 'sc_in', 'sc_out', or 'sc_inout'}}
  ^bb0(%arg0: i4):
  }) {function_type = (i4) -> (), portNames = ["port0"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-note @+1 {{in module '@signalNameConflict'}}
systemc.module @signalNameConflict () {
  // expected-note @+1 {{'signal0' first defined here}}
  %0 = "systemc.signal"() {name = "signal0"} : () -> !systemc.sc_signal<i32>
  // expected-error @+1 {{redefines name 'signal0'}}
  %1 = "systemc.signal"() {name = "signal0"} : () -> !systemc.sc_signal<i32>
}

// -----

// expected-note @+2 {{in module '@signalNameConflictWithArg'}}
// expected-note @+1 {{'in' first defined here}}
systemc.module @signalNameConflictWithArg (%in: !systemc.sc_in<i32>) {
  // expected-error @+1 {{redefines name 'in'}}
  %0 = "systemc.signal"() {name = "in"} : () -> !systemc.sc_signal<i32>
}

// -----

systemc.module @signalNameNotEmpty () {
  // expected-error @+1 {{'name' attribute must not be empty}}
  %0 = "systemc.signal"() {name = ""} : () -> !systemc.sc_signal<i32>
}

// -----

systemc.module @moduleDoesNotAccessNameBeforeExistanceVerified () {
  // expected-error @+1 {{requires attribute 'name'}}
  %0 = "systemc.signal"() {} : () -> !systemc.sc_signal<i32>
}

// -----

systemc.module @signalMustBeDirectChildOfModule () {
  systemc.ctor {
    // expected-error @+1 {{expects parent op 'systemc.module'}}
    %signal = systemc.signal : !systemc.sc_signal<i32>
  }
}

// -----

systemc.module @ctorNoBlockArguments () {
  // expected-error @+1 {{op must not have any arguments}} 
  "systemc.ctor"() ({
    ^bb0(%arg0: i32):
    }) : () -> ()
}

// -----

systemc.module @funcNoBlockArguments () {
  // expected-error @+1 {{op must not have any arguments}} 
  %0 = "systemc.func"() ({
    ^bb0(%arg0: i32):
    }) {name="funcname"}: () -> (!systemc.func_handle)
}

// -----

// expected-note @+1 {{in module '@signalFuncNameConflict'}}
systemc.module @signalFuncNameConflict () {
  // expected-note @+1 {{'name' first defined here}}
  %0 = "systemc.signal"() {name="name"} : () -> !systemc.sc_signal<i32>
  // expected-error @+1 {{redefines name 'name'}}
  %1 = "systemc.func"() ({
    ^bb0:
    }) {name="name"}: () -> (!systemc.func_handle)
}
