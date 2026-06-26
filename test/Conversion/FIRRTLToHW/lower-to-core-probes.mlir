// RUN: circt-opt --pass-pipeline='builtin.module(lower-firrtl-to-hw{lower-to-core=true})' %s | FileCheck %s

firrtl.circuit "Top" {
  // CHECK-LABEL: hw.module @Producer
  // CHECK-SAME: out p : !probe.ref<i8>
  firrtl.module @Producer(in %in: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>) {
    // CHECK: %[[P:.+]] = probe.send %in : i8 -> !probe.ref<i8>
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    firrtl.ref.define %p, %ref : !firrtl.probe<uint<8>>
    // CHECK: hw.output %[[P]] : !probe.ref<i8>
  }

  // CHECK-LABEL: hw.module @Top
  firrtl.module @Top(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK: %[[P:.+]] = hw.instance "producer" @Producer(in: %in: i8) -> (p: !probe.ref<i8>)
    %p_in, %p_p = firrtl.instance producer @Producer(in in: !firrtl.uint<8>, out p: !firrtl.probe<uint<8>>)
    firrtl.connect %p_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: %[[V:.+]] = probe.read %[[P]] : !probe.ref<i8> -> i8
    %v = firrtl.ref.resolve %p_p : !firrtl.probe<uint<8>>
    firrtl.connect %out, %v : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: hw.output %[[V]] : i8
  }

  // CHECK-LABEL: hw.module @Aggregates
  firrtl.module @Aggregates(
      in %s: !firrtl.bundle<a: uint<1>, b: uint<5>>,
      in %a: !firrtl.vector<uint<3>, 4>,
      out %out_s: !firrtl.uint<5>,
      out %out_a: !firrtl.uint<3>) {
    %sp = firrtl.ref.send %s : !firrtl.bundle<a: uint<1>, b: uint<5>>
    // CHECK: probe.subfield {{%.+}}["b"] : !probe.ref<!hw.struct<a: i1, b: i5>> -> !probe.ref<i5>
    %bp = firrtl.ref.sub %sp[1] : !firrtl.probe<bundle<a: uint<1>, b: uint<5>>>
    %b = firrtl.ref.resolve %bp : !firrtl.probe<uint<5>>
    firrtl.connect %out_s, %b : !firrtl.uint<5>, !firrtl.uint<5>

    %ap = firrtl.ref.send %a : !firrtl.vector<uint<3>, 4>
    // CHECK: probe.subindex {{%.+}}[2] : !probe.ref<!hw.array<4xi3>> -> !probe.ref<i3>
    %ep = firrtl.ref.sub %ap[2] : !firrtl.probe<vector<uint<3>, 4>>
    %e = firrtl.ref.resolve %ep : !firrtl.probe<uint<3>>
    firrtl.connect %out_a, %e : !firrtl.uint<3>, !firrtl.uint<3>
  }

  // CHECK-LABEL: hw.module @RefCast
  firrtl.module @RefCast(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    // CHECK: probe.cast {{%.+}} : !probe.ref<i8> -> !probe.ref<i8>
    %cast = firrtl.ref.cast %ref : (!firrtl.probe<uint<8>>) -> !firrtl.probe<uint<8>>
    %v = firrtl.ref.resolve %cast : !firrtl.probe<uint<8>>
    firrtl.connect %out, %v : !firrtl.uint<8>, !firrtl.uint<8>
  }

  // CHECK-LABEL: hw.module @ProbeWire
  // CHECK-SAME: out p : !probe.ref<i8>
  firrtl.module @ProbeWire(in %in: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>) {
    // CHECK-NOT: sv.constantZ {{.*}}!probe.ref
    %w = firrtl.wire : !firrtl.probe<uint<8>>
    // CHECK: %[[P:.+]] = probe.send %in : i8 -> !probe.ref<i8>
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    firrtl.ref.define %w, %ref : !firrtl.probe<uint<8>>
    firrtl.ref.define %p, %w : !firrtl.probe<uint<8>>
    // CHECK: hw.output %[[P]] : !probe.ref<i8>
  }
}
