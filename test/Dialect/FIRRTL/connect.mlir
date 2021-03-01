
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

firrtl.module @bundle0(%a : !firrtl.bundle<f1: uint<1>, f2: flip<sint<1>>>, %b : !firrtl.bundle<f1: flip<uint<1>>, f2: sint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.bundle<f1: flip<uint<1>>, f2: sint<1>>, !firrtl.bundle<f1: uint<1>, f2: flip<sint<1>>>
}

firrtl.module @bundle1(%a : !firrtl.bundle<f1: uint<1>, f2: flip<sint<2>>>, %b : !firrtl.bundle<f1: flip<uint<2>>, f2: sint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.bundle<f1: flip<uint<2>>, f2: sint<1>>, !firrtl.bundle<f1: uint<1>, f2: flip<sint<2>>>
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

}