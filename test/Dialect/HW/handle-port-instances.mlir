// RUN: circt-opt %s --verify-diagnostics --split-input-file

// Matching handle kinds connect without error: net-to-net.
hw.module @netTarget(in %n: !sv.net<i4>) {
}
hw.module @netCaller(in %n: !sv.net<i4>) {
  hw.instance "inst" @netTarget(n: %n: !sv.net<i4>) -> ()
}

// -----

// A plain SSA value may not connect to a net-typed port: the caller must
// supply an actual handle, not a bare value.
// expected-note @+1 {{module declared here}}
hw.module @netTarget(in %n: !sv.net<i4>) {
}
hw.module @caller(in %a: i4) {
  // expected-error @+1 {{'hw.instance' op operand type #0 must be '!sv.net<i4>', but got 'i4'}}
  hw.instance "inst" @netTarget(n: %a: i4) -> ()
}

// -----

// An inout port cannot be left unconnected.
// expected-note @+1 {{module declared here}}
hw.module @netTarget(in %n: !sv.net<i4>) {
}
hw.module @caller() {
  // expected-error @+1 {{'hw.instance' op has a wrong number of operands; expected 1 but got 0}}
  hw.instance "inst" @netTarget() -> ()
}
