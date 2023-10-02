// RUN: circt-opt --firrtl-probe-dce --split-input-file %s | FileCheck %s


// Input probe used locally, but not within module it is port for.
// CHECK-LABEL: "UTurnTH"
firrtl.circuit "UTurnTH" {
  // CHECK-NOT: probe
  firrtl.module private @UTurn(in %in: !firrtl.probe<uint<5>>) { }
  // CHECK: @UTurnTH
  firrtl.module @UTurnTH(in %in: !firrtl.uint<5>) {
    // CHECK: firrtl.wire : !firrtl.probe
    %u_in = firrtl.instance u @UTurn(in in: !firrtl.probe<uint<5>>)
    %0 = firrtl.ref.send %in : !firrtl.uint<5>
    firrtl.ref.define %u_in, %0 : !firrtl.probe<uint<5>>
    %1 = firrtl.ref.resolve %u_in : !firrtl.probe<uint<5>>
  }
}

// -----

// Wire, connect, ref.sub's
// CHECK-LABEL: "Split"
firrtl.circuit "Split" {
  // CHECK-NOT: probe
  firrtl.module private @SPassthrough(in %in: !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>) {
    %w = firrtl.wire : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    firrtl.ref.define %w, %in : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    %0 = firrtl.ref.sub %w[1] : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    %1 = firrtl.ref.sub %w[0] : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
  }
  // CHECK: module @Split
  firrtl.module @Split(in %x: !firrtl.bundle<a: uint<5>, b: uint<5>>, out %y: !firrtl.probe<uint<5>>, out %z: !firrtl.probe<uint<5>>) {
    // CHECK: %[[WIRE:.+]] = firrtl.wire : !firrtl.probe
    // CHECK: firrtl.instance sp interesting_name @SPassthrough()
    %sp_in = firrtl.instance sp interesting_name @SPassthrough(in in: !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>)
    // CHECK: ref.sub %[[WIRE]][1]
    %0 = firrtl.ref.sub %sp_in[1] : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    %1 = firrtl.ref.sub %sp_in[0] : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    %2 = firrtl.ref.send %x : !firrtl.bundle<a: uint<5>, b: uint<5>>
    firrtl.ref.define %sp_in, %2 : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    firrtl.ref.define %y, %1 : !firrtl.probe<uint<5>>
    firrtl.ref.define %z, %0 : !firrtl.probe<uint<5>>
  }
}

// -----

// Multi-level.
// CHECK-LABEL: "ProbeMultiLevel"
firrtl.circuit "ProbeMultiLevel" {
  // CHECK: @Child()
  firrtl.module private @Child(in %x: !firrtl.probe<uint<5>>) { }
  // CHECK: @Mid() {
  firrtl.module private @Mid(in %x: !firrtl.probe<uint<5>>) {
    // CHECK-NEXT: firrtl.instance c interesting_name @Child()
    // CHECK-NEXT: }
    %c_x = firrtl.instance c interesting_name @Child(in x: !firrtl.probe<uint<5>>)
    firrtl.ref.define %c_x, %x : !firrtl.probe<uint<5>>
  }
  // CHECK: @ProbeMultiLevel
  firrtl.module @ProbeMultiLevel(in %x: !firrtl.uint<5>, in %x2: !firrtl.uint<5>) {
    // CHECK: %[[W1:.+]] = firrtl.wire : !firrtl.probe
    %m_x = firrtl.instance m interesting_name @Mid(in x: !firrtl.probe<uint<5>>)
    // CHECK: %[[W2:.+]] = firrtl.wire : !firrtl.probe
    %m2_x = firrtl.instance m2 interesting_name @Mid(in x: !firrtl.probe<uint<5>>)
    // CHECK: %[[REF:.+]] = firrtl.ref.send
    %ref = firrtl.ref.send %x : !firrtl.uint<5>
    // CHECK: ref.define %[[W1]], %[[REF]]
    // CHECK: ref.define %[[W2]], %[[REF]]
    firrtl.ref.define %m_x, %ref : !firrtl.probe<uint<5>>
    firrtl.ref.define %m2_x, %ref : !firrtl.probe<uint<5>>
  }
}

// -----

// CHECK-LABEL: "RefCast"
firrtl.circuit "RefCast" {
  // CHECK: @UTurnCast() {
  // CHECK-NEXT: }
  firrtl.module private @UTurnCast(in %in: !firrtl.rwprobe<uint<5>>) {
    %0 = firrtl.ref.cast %in : (!firrtl.rwprobe<uint<5>>) -> !firrtl.probe<uint<5>>
  }
  // CHECK: @RefCast(
  firrtl.module @RefCast(in %in: !firrtl.uint<5>) {
    // CHECK: @UTurnCast()
    %u_in = firrtl.instance u @UTurnCast(in in: !firrtl.rwprobe<uint<5>>)
    %0 = firrtl.ref.cast %u_in : (!firrtl.rwprobe<uint<5>>) -> !firrtl.probe<uint<5>>
    %n, %n_ref = firrtl.node %in forceable : !firrtl.uint<5>
    firrtl.ref.define %u_in, %n_ref : !firrtl.rwprobe<uint<5>>
    %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<5>>
  }
}

// -----
// Wire + cast.

// CHECK-LABEL: "RefWire"
firrtl.circuit "RefWire" {
  // CHECK: @UWire() {
  // CHECK-NEXT: }
  firrtl.module private @UWire(in %in: !firrtl.rwprobe<uint<5>>) {
    %w = firrtl.wire : !firrtl.rwprobe<uint<5>>
    firrtl.ref.define %w, %in : !firrtl.rwprobe<uint<5>>
    %0 = firrtl.ref.cast %w : (!firrtl.rwprobe<uint<5>>) -> !firrtl.probe<uint<5>>
  }
  // CHECK: @RefWire(
  firrtl.module @RefWire(in %in: !firrtl.uint<5>) {
   // CHECK: UWire()
    %u_in = firrtl.instance u @UWire(in in: !firrtl.rwprobe<uint<5>>)
    %0 = firrtl.ref.cast %u_in : (!firrtl.rwprobe<uint<5>>) -> !firrtl.probe<uint<5>>
    %n, %n_ref = firrtl.node %in forceable : !firrtl.uint<5>
    firrtl.ref.define %u_in, %n_ref : !firrtl.rwprobe<uint<5>>
    %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<5>>
  }
}

// -----

// CHECK-LABEL: "WireWhen"
firrtl.circuit "WireWhen" {
  // CHECK: @WireWhen(
  firrtl.module @WireWhen(in %in_cond: !firrtl.uint<1>, in %data: !firrtl.vector<uint<1>, 2>) {
    // CHECK: @Child(in cond: !firrtl.uint<1>)
    %c_cond, %c_in = firrtl.instance c @Child(in cond: !firrtl.uint<1>, in in: !firrtl.probe<vector<uint<1>, 2>>)
    firrtl.strictconnect %c_cond, %in_cond : !firrtl.uint<1>
    %0 = firrtl.ref.sub %c_in[1] : !firrtl.probe<vector<uint<1>, 2>>
    %ref = firrtl.ref.send %data : !firrtl.vector<uint<1>, 2>
    firrtl.ref.define %c_in, %ref : !firrtl.probe<vector<uint<1>, 2>>
  }
  // CHECK: @Child(in %cond: !firrtl.uint<1>) {
  // CHECK-NEXT: firrtl.when
  // CHECK-NEXT: } else {
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  firrtl.module private @Child(in %cond: !firrtl.uint<1>, in %in: !firrtl.probe<vector<uint<1>, 2>>) {
    %0 = firrtl.wire : !firrtl.probe<uint<1>>
    firrtl.when %cond : !firrtl.uint<1> {
      %2 = firrtl.ref.sub %in[1] : !firrtl.probe<vector<uint<1>, 2>>
      firrtl.ref.define %0, %2 : !firrtl.probe<uint<1>>
    } else {
      %w = firrtl.wire : !firrtl.probe<uint<1>>
      firrtl.ref.define %w, %0 : !firrtl.probe<uint<1>>
    }
  }
}

