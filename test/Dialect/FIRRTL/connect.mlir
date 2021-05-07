
// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "reset0" {

// Reset destination.
firrtl.module @reset0(%a : !firrtl.uint<1>, %b : !firrtl.flip<reset>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<reset>, !firrtl.uint<1>
}

firrtl.module @reset1(%a : !firrtl.asyncreset, %b : !firrtl.flip<reset>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<reset>, !firrtl.asyncreset
}

/// Reset types can be connected to Reset, UInt<1>, or AsyncReset types.

// Reset source.
firrtl.module @reset2(%a : !firrtl.reset, %b : !firrtl.flip<reset>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<reset>, !firrtl.reset
}

firrtl.module @reset3(%a : !firrtl.reset, %b : !firrtl.flip<uint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.reset
}

firrtl.module @reset4(%a : !firrtl.reset, %b : !firrtl.flip<asyncreset>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<asyncreset>, !firrtl.reset
}

// AsyncReset source.
firrtl.module @asyncreset0(%a : !firrtl.asyncreset, %b : !firrtl.flip<asyncreset>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<asyncreset>, !firrtl.asyncreset
}

// Clock source.
firrtl.module @clock0(%a : !firrtl.clock, %b : !firrtl.flip<clock>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<clock>, !firrtl.clock
}

/// Ground types can be connected if they are the same ground type.

// SInt<> source.
firrtl.module @sint0(%a : !firrtl.sint<1>, %b : !firrtl.flip<sint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<sint<1>>, !firrtl.sint<1>
}

// UInt<> source.
firrtl.module @uint0(%a : !firrtl.uint<1>, %b : !firrtl.flip<uint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}
firrtl.module @uint1(%a : !firrtl.uint<1>, %b : !firrtl.flip<uint<2>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<uint<2>>, !firrtl.uint<1>
}

/// Vector types can be connected if they have the same size and element type.
firrtl.module @vect0(%a : !firrtl.vector<uint<1>, 3>, %b : !firrtl.flip<vector<uint<1>, 3>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<vector<uint<1>, 3>>, !firrtl.vector<uint<1>, 3>
}

/// Bundle types can be connected if they have the same size, element names, and
/// element types.

firrtl.module @bundle0(%a : !firrtl.bundle<f1: uint<1>, f2: flip<sint<1>>>, %b : !firrtl.flip<bundle<f1: uint<1>, f2: flip<sint<1>>>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<bundle<f1: uint<1>, f2: flip<sint<1>>>>, !firrtl.bundle<f1: uint<1>, f2: flip<sint<1>>>
}

firrtl.module @bundle1(%a : !firrtl.bundle<f1: uint<1>, f2: flip<sint<2>>>, %b : !firrtl.flip<bundle<f1: uint<2>, f2: flip<sint<1>>>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<bundle<f1: uint<2>, f2: flip<sint<1>>>>, !firrtl.bundle<f1: uint<1>, f2: flip<sint<2>>>
}

/// Destination bitwidth must be greater than or equal to source bitwidth.
firrtl.module @bitwidth(%a : !firrtl.uint<1>, %b : !firrtl.flip<uint<2>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.flip<uint<2>>, !firrtl.uint<1>
}

/// Partial connects may truncate.
firrtl.module @partial_bitwidth(%a : !firrtl.uint<2>, %b : !firrtl.flip<uint<1>>) {
  // CHECK: firrtl.partialconnect %b, %a
  firrtl.partialconnect %b, %a : !firrtl.flip<uint<1>>, !firrtl.uint<2>
}

firrtl.module @wires0(%in : !firrtl.uint<1>, %out : !firrtl.flip<uint<1>>) {
  %w = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %w, %in : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %w : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %w, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %w : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

firrtl.module @wires1(%in : !firrtl.uint<1>, %out : !firrtl.flip<uint<1>>) {
  %wf = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %wf, %in : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %wf : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %wf, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %wf : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

firrtl.module @wires2() {
  %w0 = firrtl.wire : !firrtl.uint<1>
  %w1 = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %w0, %w1
  firrtl.connect %w0, %w1 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires3(%out : !firrtl.flip<uint<1>>) {
  %wf = firrtl.wire : !firrtl.uint<1>
  // check that we can read from an output port
  // CHECK: firrtl.connect %wf, %out
  firrtl.connect %wf, %out : !firrtl.uint<1>, !firrtl.flip<uint<1>>
}

firrtl.module @wires4(%in : !firrtl.uint<1>, %out : !firrtl.flip<uint<1>>) {
  %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
  %0 = firrtl.subfield %w("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
  // CHECK: firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

firrtl.module @registers0(%clock : !firrtl.clock, %in : !firrtl.uint<1>, %out : !firrtl.flip<uint<1>>) {
  %0 = firrtl.reg %clock : (!firrtl.clock) -> !firrtl.uint<1>
  // CHECK: firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

firrtl.module @registers1(%clock : !firrtl.clock) {
  %0 = firrtl.reg %clock : (!firrtl.clock) -> !firrtl.uint<1>
  %1 = firrtl.reg %clock : (!firrtl.clock) -> !firrtl.uint<1>
  // CHECK: firrtl.connect %0, %1
  firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

}
