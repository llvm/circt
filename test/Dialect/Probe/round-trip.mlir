// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: hw.module @Basic
hw.module @Basic(in %in: i8, out out: i8) {
  // CHECK: %[[P:.+]] = probe.send %in : i8 -> !probe.ref<i8>
  %p = probe.send %in : i8 -> !probe.ref<i8>
  // CHECK: %[[V:.+]] = probe.read %[[P]] : !probe.ref<i8> -> i8
  %v = probe.read %p : !probe.ref<i8> -> i8
  hw.output %v : i8
}

// CHECK-LABEL: hw.module @ProbePort
// CHECK-SAME: in %p : !probe.ref<i8>
hw.module @ProbePort(in %p: !probe.ref<i8>, out out: i8) {
  %v = probe.read %p : !probe.ref<i8> -> i8
  hw.output %v : i8
}

// CHECK-LABEL: hw.module @ProbeProducer
// CHECK-SAME: out p : !probe.ref<i8>
hw.module @ProbeProducer(in %in: i8, out p: !probe.ref<i8>) {
  %p = probe.send %in : i8 -> !probe.ref<i8>
  hw.output %p : !probe.ref<i8>
}

// CHECK-LABEL: hw.module @ProbeInstanceRead
hw.module @ProbeInstanceRead(in %in: i8, out out: i8) {
  // CHECK: %[[P:.+]] = hw.instance "producer" @ProbeProducer(in: %in: i8) -> (p: !probe.ref<i8>)
  %p = hw.instance "producer" @ProbeProducer(in: %in: i8) -> (p: !probe.ref<i8>)
  // CHECK: %[[V:.+]] = probe.read %[[P]] : !probe.ref<i8> -> i8
  %v = probe.read %p : !probe.ref<i8> -> i8
  hw.output %v : i8
}

// CHECK-LABEL: hw.module @Aggregates
hw.module @Aggregates(in %s: !hw.struct<a: i1, b: i5>, in %a: !hw.array<4xi3>, out out_s: i5, out out_a: i3) {
  %sp = probe.send %s : !hw.struct<a: i1, b: i5> -> !probe.ref<!hw.struct<a: i1, b: i5>>
  // CHECK: probe.subfield {{%.+}}["b"] : !probe.ref<!hw.struct<a: i1, b: i5>> -> !probe.ref<i5>
  %bp = probe.subfield %sp["b"] : !probe.ref<!hw.struct<a: i1, b: i5>> -> !probe.ref<i5>
  %b = probe.read %bp : !probe.ref<i5> -> i5

  %ap = probe.send %a : !hw.array<4xi3> -> !probe.ref<!hw.array<4xi3>>
  // CHECK: probe.subindex {{%.+}}[2] : !probe.ref<!hw.array<4xi3>> -> !probe.ref<i3>
  %ep = probe.subindex %ap[2] : !probe.ref<!hw.array<4xi3>> -> !probe.ref<i3>
  %e = probe.read %ep : !probe.ref<i3> -> i3
  hw.output %b, %e : i5, i3
}

hw.type_scope @types {
  hw.typedecl @byte : i8
}

// CHECK-LABEL: hw.module @TypeAlias
hw.module @TypeAlias(in %in: !hw.typealias<@types::@byte, i8>, out out: !hw.typealias<@types::@byte, i8>) {
  %p = probe.send %in : !hw.typealias<@types::@byte, i8> -> !probe.ref<!hw.typealias<@types::@byte, i8>>
  // CHECK: probe.cast {{%.+}} : !probe.ref<!hw.typealias<@types::@byte, i8>> -> !probe.ref<i8>
  %cast = probe.cast %p : !probe.ref<!hw.typealias<@types::@byte, i8>> -> !probe.ref<i8>
  %v = probe.read %cast : !probe.ref<i8> -> i8
  %out = hw.bitcast %v : (i8) -> !hw.typealias<@types::@byte, i8>
  hw.output %out : !hw.typealias<@types::@byte, i8>
}
