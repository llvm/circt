
// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.uint<1>, %b : !firrtl.flip<uint<1>>) {
  // expected-error @+1 {{connection destination must be a non-passive type or a duplex value}}
  firrtl.connect %a, %b : !firrtl.uint<1>, !firrtl.flip<uint<1>>
}
}

/// Analog types cannot be connected and must be attached.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.analog, %b : !firrtl.flip<analog>) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.flip<analog>, !firrtl.analog
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.analog, %b : !firrtl.flip<uint<1>>) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.analog
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.uint<1>, %b : !firrtl.flip<analog>) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.flip<analog>, !firrtl.uint<1>
}
}

/// Reset types can be connected to Reset, UInt<1>, or AsyncReset types.

// Reset source.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.reset, %b : !firrtl.flip<uint<2>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<2>' and source '!firrtl.reset'}}
  firrtl.connect %b, %a : !firrtl.flip<uint<2>>, !firrtl.reset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.reset, %b : !firrtl.flip<sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.reset'}}
  firrtl.connect %b, %a : !firrtl.flip<sint<1>>, !firrtl.reset
}
}

// Reset destination.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.uint<2>, %b : !firrtl.flip<reset>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.reset' and source '!firrtl.uint<2>'}}
  firrtl.connect %b, %a : !firrtl.flip<reset>, !firrtl.uint<2>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.sint<1>, %b : !firrtl.flip<reset>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.reset' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.flip<reset>, !firrtl.sint<1>
}
}

/// Ground types can be connected if they are the same ground type.

// UInt<> source.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.uint<1>, %b : !firrtl.flip<sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.flip<sint<1>>, !firrtl.uint<1>
}
}

// -----

firrtl.circuit "test" {

firrtl.module @test(%a : !firrtl.uint<1>, %b : !firrtl.flip<clock>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.flip<clock>, !firrtl.uint<1>
}

}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.uint<1>, %b : !firrtl.flip<asyncreset>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.flip<asyncreset>, !firrtl.uint<1>
}
}

// SInt<> source.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.sint<1>, %b : !firrtl.flip<uint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.sint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.sint<1>, %b : !firrtl.flip<clock>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.flip<clock>, !firrtl.sint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.sint<1>, %b : !firrtl.flip<asyncreset>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.flip<asyncreset>, !firrtl.sint<1>
}
}

// Clock source.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.clock, %b : !firrtl.flip<uint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.clock
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.clock, %b : !firrtl.flip<sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.flip<sint<1>>, !firrtl.clock
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.clock, %b : !firrtl.flip<asyncreset>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.flip<asyncreset>, !firrtl.clock
}
}

// AsyncReset source.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.asyncreset, %b : !firrtl.flip<uint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.asyncreset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.asyncreset, %b : !firrtl.flip<sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.flip<sint<1>>, !firrtl.asyncreset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.asyncreset, %b : !firrtl.flip<clock>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.flip<clock>, !firrtl.asyncreset
}
}

/// Vector types can be connected if they have the same size and element type.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.vector<uint<1>, 3>, %b : !firrtl.flip<vector<uint<1>, 2>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.vector<uint<1>, 2>' and source '!firrtl.vector<uint<1>, 3>'}}
  firrtl.connect %b, %a : !firrtl.flip<vector<uint<1>, 2>>, !firrtl.vector<uint<1>, 3>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.vector<uint<1>, 3>, %b : !firrtl.flip<vector<sint<1>, 3>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.vector<sint<1>, 3>' and source '!firrtl.vector<uint<1>, 3>'}}
  firrtl.connect %b, %a : !firrtl.flip<vector<sint<1>, 3>>, !firrtl.vector<uint<1>, 3>
}
}

/// Bundle types can be connected if they have the same size, element names, and
/// element types.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.bundle<f1: uint<1>>, %b : !firrtl.bundle<f1: flip<uint<1>>, f2: sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f1: uint<1>, f2: sint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f1: flip<uint<1>>, f2: sint<1>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.bundle<f1: uint<1>>, %b : !firrtl.bundle<f2: flip<uint<1>>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f2: uint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f2: flip<uint<1>>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.bundle<f1: uint<1>>, %b : !firrtl.bundle<f1: flip<sint<1>>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f1: sint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f1: flip<sint<1>>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.bundle<f1: uint<1>>, %b : !firrtl.flip<bundle<f1: uint<1>>>) {
  %0 = firrtl.subfield %a("f1") : (!firrtl.bundle<f1: uint<1>>) -> !firrtl.uint<1>
  %1 = firrtl.subfield %b("f1") : (!firrtl.flip<bundle<f1: uint<1>>>) -> !firrtl.flip<uint<1>>
  // expected-error @+1 {{connection destination must be a non-passive type or a duplex value}}
  firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.flip<uint<1>>
}
}

/// Destination bitwidth must be greater than or equal to source bitwidth.

// -----

firrtl.circuit "test" {
firrtl.module @test(%a : !firrtl.uint<2>, %b : !firrtl.flip<uint<1>>) {
  // expected-error @+1 {{destination width 1 is not greater than or equal to source width 2}}
  firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.uint<2>
}
}

// -----

// Two duplex values with bundle types can not be bulk connected.
firrtl.circuit "test" {
firrtl.module @test(%clock : !firrtl.clock) {
  %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  %r = firrtl.reg %clock : (!firrtl.clock) -> !firrtl.bundle<a: uint<1>>
  // expected-error @+1 {{ambiguous bulk connection between two duplex values of bundle type}}
  firrtl.connect %r, %w : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}
}
