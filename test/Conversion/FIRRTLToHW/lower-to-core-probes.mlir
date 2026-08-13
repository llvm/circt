// RUN: circt-opt --pass-pipeline='builtin.module(lower-firrtl-to-hw{lower-to-core=true})' %s | FileCheck %s

firrtl.circuit "Top" {
  // CHECK-LABEL: hw.module @Producer
  // CHECK-SAME: out p : !probe.ref<i8>
  firrtl.module @Producer(in %in: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>) {
    // CHECK: %[[P:.+]] = probe.send %in : i8
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    firrtl.ref.define %p, %ref : !firrtl.probe<uint<8>>
    // CHECK: hw.output %[[P]] : !probe.ref<i8>
  }

  // CHECK-LABEL: hw.module @Top
  firrtl.module @Top(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK: %[[P:.+]] = hw.instance "producer" @Producer(in: %in: i8) -> (p: !probe.ref<i8>)
    %p_in, %p_p = firrtl.instance producer @Producer(in in: !firrtl.uint<8>, out p: !firrtl.probe<uint<8>>)
    firrtl.connect %p_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: %[[V:.+]] = probe.read %[[P]] : <i8>
    %v = firrtl.ref.resolve %p_p : !firrtl.probe<uint<8>>
    firrtl.connect %out, %v : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: hw.output %[[V]] : i8
  }

  // CHECK-LABEL: hw.module @ClockProbe
  // CHECK-SAME: in %clock : !seq.clock
  // CHECK-SAME: out p : !probe.ref<!seq.clock>
  // CHECK-SAME: out out : !seq.clock
  firrtl.module @ClockProbe(in %clock: !firrtl.clock, out %p: !firrtl.probe<clock>, out %out: !firrtl.clock) {
    // CHECK: %[[P:.+]] = probe.send %clock : !seq.clock
    %ref = firrtl.ref.send %clock : !firrtl.clock
    firrtl.ref.define %p, %ref : !firrtl.probe<clock>
    // CHECK: %[[V:.+]] = probe.read %[[P]] : <!seq.clock>
    %v = firrtl.ref.resolve %ref : !firrtl.probe<clock>
    firrtl.connect %out, %v : !firrtl.clock, !firrtl.clock
    // CHECK: hw.output %[[P]], %[[V]] : !probe.ref<!seq.clock>, !seq.clock
  }

  // CHECK-LABEL: hw.module @ClockAggregateProbe
  // CHECK-SAME: in %in : !hw.struct<clk: !seq.clock, data: i1>
  // CHECK-SAME: out p : !probe.ref<!hw.struct<clk: !seq.clock, data: i1>>
  firrtl.module @ClockAggregateProbe(in %in: !firrtl.bundle<clk: clock, data: uint<1>>, out %p: !firrtl.probe<bundle<clk: clock, data: uint<1>>>) {
    // CHECK: %[[P:.+]] = probe.send %in : !hw.struct<clk: !seq.clock, data: i1>
    %ref = firrtl.ref.send %in : !firrtl.bundle<clk: clock, data: uint<1>>
    firrtl.ref.define %p, %ref : !firrtl.probe<bundle<clk: clock, data: uint<1>>>
    // CHECK: hw.output %[[P]] : !probe.ref<!hw.struct<clk: !seq.clock, data: i1>>
  }

  // CHECK-LABEL: hw.module @Aggregates
  firrtl.module @Aggregates(
      in %s: !firrtl.bundle<a: uint<1>, b: uint<5>>,
      in %a: !firrtl.vector<uint<3>, 4>,
      out %out_s: !firrtl.uint<5>,
      out %out_a: !firrtl.uint<3>) {
    %sp = firrtl.ref.send %s : !firrtl.bundle<a: uint<1>, b: uint<5>>
    // CHECK: probe.subfield {{%.+}}["b"] : <!hw.struct<a: i1, b: i5>> -> <i5>
    %bp = firrtl.ref.sub %sp[1] : !firrtl.probe<bundle<a: uint<1>, b: uint<5>>>
    %b = firrtl.ref.resolve %bp : !firrtl.probe<uint<5>>
    firrtl.connect %out_s, %b : !firrtl.uint<5>, !firrtl.uint<5>

    %ap = firrtl.ref.send %a : !firrtl.vector<uint<3>, 4>
    // CHECK: probe.subindex {{%.+}}[2] : <!hw.array<4xi3>>
    %ep = firrtl.ref.sub %ap[2] : !firrtl.probe<vector<uint<3>, 4>>
    %e = firrtl.ref.resolve %ep : !firrtl.probe<uint<3>>
    firrtl.connect %out_a, %e : !firrtl.uint<3>, !firrtl.uint<3>
  }

  // CHECK-LABEL: hw.module @RefCast
  firrtl.module @RefCast(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    // CHECK: probe.cast {{%.+}} : <i8> -> <i8>
    %cast = firrtl.ref.cast %ref : (!firrtl.probe<uint<8>>) -> !firrtl.probe<uint<8>>
    %v = firrtl.ref.resolve %cast : !firrtl.probe<uint<8>>
    firrtl.connect %out, %v : !firrtl.uint<8>, !firrtl.uint<8>
  }

  // CHECK-LABEL: hw.module @ProbeWire
  // CHECK-SAME: out p : !probe.ref<i8>
  firrtl.module @ProbeWire(in %in: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>) {
    // CHECK-NOT: sv.constantZ {{.*}}!probe.ref
    %w = firrtl.wire : !firrtl.probe<uint<8>>
    // CHECK: %[[P:.+]] = probe.send %in : i8
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    firrtl.ref.define %w, %ref : !firrtl.probe<uint<8>>
    firrtl.ref.define %p, %w : !firrtl.probe<uint<8>>
    // CHECK: hw.output %[[P]] : !probe.ref<i8>
  }

  // CHECK-LABEL: hw.module @UnusedProbeWire
  firrtl.module @UnusedProbeWire() {
    %w = firrtl.wire : !firrtl.probe<uint<8>>
  }

  // CHECK-LABEL: hw.module @OutputProbeUsed
  // CHECK-SAME: out p : !probe.ref<i8>
  firrtl.module @OutputProbeUsed(in %in: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>, out %out: !firrtl.uint<8>) {
    // CHECK: %[[P:.+]] = probe.send %in : i8
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    firrtl.ref.define %p, %ref : !firrtl.probe<uint<8>>
    // CHECK: %[[V:.+]] = probe.read %[[P]] : <i8>
    %v = firrtl.ref.resolve %p : !firrtl.probe<uint<8>>
    firrtl.connect %out, %v : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: hw.output %[[P]], %[[V]] : !probe.ref<i8>, i8
  }

  // CHECK-LABEL: hw.module @ZeroWidthResolve
  // CHECK-SAME: out p : !probe.ref<i0>
  firrtl.module @ZeroWidthResolve(in %in: !firrtl.uint<0>, out %out: !firrtl.uint<0>, out %p: !firrtl.probe<uint<0>>) {
    // CHECK: %[[P:.+]] = probe.send
    %ref = firrtl.ref.send %in : !firrtl.uint<0>
    firrtl.ref.define %p, %ref : !firrtl.probe<uint<0>>
    // CHECK-NOT: probe.read
    %v = firrtl.ref.resolve %ref : !firrtl.probe<uint<0>>
    firrtl.connect %out, %v : !firrtl.uint<0>, !firrtl.uint<0>
    // CHECK: hw.output %[[P]] : !probe.ref<i0>
  }
}
