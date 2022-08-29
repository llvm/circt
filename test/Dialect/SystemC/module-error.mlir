// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{entry block must have 3 arguments to match function signature}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>) -> (), portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of port names}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>, !systemc.inout<i8>) -> (), portNames = ["port0", "port1", "port2"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of function results (always has to be 0)}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>, !systemc.inout<i8>) -> (i1), portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{port name must not be empty}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>, !systemc.inout<i8>) -> (), portNames = ["port0", "port1", "port2", ""], sym_name = "verifierTest"} : () -> ()

// -----

// expected-note @+1 {{in module '@verifierTest'}}
"systemc.module"() ({
  // expected-error @+2 {{redefines port name 'port2'}}
  // expected-note @+1 {{'port2' first defined here}}
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>, !systemc.inout<i8>) -> (), portNames = ["port0", "port1", "port2", "port2"], sym_name = "verifierTest"} : () -> ()

// -----

"systemc.module"() ({
  // expected-error @+1 {{module port must be of type 'sc_in', 'sc_out', or 'sc_inout'}}
  ^bb0(%arg0: i4):
  }) {function_type = (i4) -> (), portNames = ["port0"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{failed to satisfy constraint: string array attribute}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.in<i4>):
  }) {function_type = (!systemc.in<i4>) -> (), portNames = [i32], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{attribute 'portNames' occurs more than once in the attribute list}}
systemc.module @noExplicitPortNamesAttr () attributes {portNames=["p1"]} {}

// -----

// expected-note @+1 {{in module '@signalNameConflict'}}
systemc.module @signalNameConflict () {
  // expected-note @+1 {{'signal0' first defined here}}
  %0 = "systemc.signal"() {name = "signal0"} : () -> !systemc.signal<i32>
  // expected-error @+1 {{redefines name 'signal0'}}
  %1 = "systemc.signal"() {name = "signal0"} : () -> !systemc.signal<i32>
}

// -----

// expected-note @+2 {{in module '@signalNameConflictWithArg'}}
// expected-note @+1 {{'in' first defined here}}
systemc.module @signalNameConflictWithArg (%in: !systemc.in<i32>) {
  // expected-error @+1 {{redefines name 'in'}}
  %0 = "systemc.signal"() {name = "in"} : () -> !systemc.signal<i32>
}

// -----

systemc.module @signalNameNotEmpty () {
  // expected-error @+1 {{'name' attribute must not be empty}}
  %0 = "systemc.signal"() {name = ""} : () -> !systemc.signal<i32>
}

// -----

systemc.module @moduleDoesNotAccessNameBeforeExistanceVerified () {
  // expected-error @+1 {{requires attribute 'name'}}
  %0 = "systemc.signal"() {} : () -> !systemc.signal<i32>
}

// -----

systemc.module @signalMustBeDirectChildOfModule () {
  systemc.ctor {
    // expected-error @+1 {{expects parent op 'systemc.module'}}
    %signal = systemc.signal : !systemc.signal<i32>
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
    }) {name="funcname"}: () -> (() -> ())
}

// -----

systemc.module @funcNoBlockArguments () {
  // expected-error @+1 {{result #0 must be FunctionType with no inputs and results, but got '(i32) -> ()'}}
  %0 = "systemc.func"() ({
    ^bb0():
    }) {name="funcname"}: () -> ((i32) -> ())
}

// -----

// expected-note @+1 {{in module '@signalFuncNameConflict'}}
systemc.module @signalFuncNameConflict () {
  // expected-note @+1 {{'name' first defined here}}
  %0 = "systemc.signal"() {name="name"} : () -> !systemc.signal<i32>
  // expected-error @+1 {{redefines name 'name'}}
  %1 = "systemc.func"() ({
    ^bb0:
    }) {name="name"}: () -> (() -> ())
}

// -----

systemc.module @cannotReadFromOutPort (%port0: !systemc.out<i32>) {
  // expected-error @+1 {{op operand #0 must be InputType or InOutType or SignalType, but got '!systemc.out<i32>'}}
  %0 = systemc.signal.read %port0 : !systemc.out<i32>
}

// -----

systemc.module @inferredTypeDoesNotMatch (%port0: !systemc.in<i32>) {
  // expected-error @+1 {{op inferred type(s) 'i32' are incompatible with return type(s) of operation 'i4'}}
  %0 = "systemc.signal.read"(%port0) : (!systemc.in<i32>) -> i4
}

// -----

systemc.module @cannotWriteToInputPort (%port0: !systemc.in<i32>) {
  %0 = hw.constant 0 : i32
  // expected-error @+1 {{'dest' must be OutputType or InOutType or SignalType, but got '!systemc.in<i32>'}}
  systemc.signal.write %port0, %0 : !systemc.in<i32>
}

// -----

systemc.module @invalidSignalOpReturnType () {
  // expected-error @+1 {{invalid kind of type specified}}
  %signal0 = systemc.signal : i32
}
