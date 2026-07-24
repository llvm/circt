// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --canonicalize --cse | FileCheck %s --check-prefix=OPT

// CHECK-LABEL: hw.module @Basic
hw.module @Basic(in %in: i8, out forwarded: i8, out observed: i8) {
  // CHECK: %[[F:.+]], %[[P:.+]] = probe.send %in : i8
  %forwarded, %p = probe.send %in : i8
  // CHECK: %[[V:.+]] = probe.read %[[P]] : <i8>
  %v = probe.read %p : <i8>
  hw.output %forwarded, %v : i8, i8
}

// CHECK-LABEL: hw.module @Clock
// CHECK-SAME: in %clock : !seq.clock
// CHECK-SAME: out forwarded : !seq.clock
// CHECK-SAME: out observed : !seq.clock
hw.module @Clock(in %clock: !seq.clock, out forwarded: !seq.clock,
                 out observed: !seq.clock) {
  // CHECK: %[[F:.+]], %[[P:.+]] = probe.send %clock : !seq.clock
  %forwarded, %p = probe.send %clock : !seq.clock
  // CHECK: %[[V:.+]] = probe.read %[[P]] : <!seq.clock>
  %v = probe.read %p : <!seq.clock>
  hw.output %forwarded, %v : !seq.clock, !seq.clock
}

// CHECK-LABEL: hw.module @Aggregate
hw.module @Aggregate(
    in %in: !hw.struct<data: i8, clock: !seq.clock>,
    out out: !hw.struct<data: i8, clock: !seq.clock>) {
  // CHECK: %[[F:.+]], %[[P:.+]] = probe.send %in : !hw.struct<data: i8, clock: !seq.clock>
  %forwarded, %p = probe.send %in : !hw.struct<data: i8, clock: !seq.clock>
  %v = probe.read %p : <!hw.struct<data: i8, clock: !seq.clock>>
  hw.output %v : !hw.struct<data: i8, clock: !seq.clock>
}

// CHECK-LABEL: hw.module @Expression
hw.module @Expression(in %a: i8, in %b: i8, out forwarded: i8,
                      out observed: i8) {
  %value = comb.xor %a, %b : i8
  // CHECK: %[[F:.+]], %[[P:.+]] = probe.send %{{.+}} : i8
  %forwarded, %p = probe.send %value : i8
  %v = probe.read %p : <i8>
  hw.output %forwarded, %v : i8, i8
}

// CHECK-LABEL: hw.module @ProbeProducer
// CHECK-SAME: out p : !probe.ref<i8>
hw.module @ProbeProducer(in %in: i8, out p: !probe.ref<i8>) {
  // The unused forwarded result expresses a probe-only tap.
  %forwarded, %p = probe.send %in : i8
  hw.output %p : !probe.ref<i8>
}

// CHECK-LABEL: hw.module @ProbeInstanceRead
hw.module @ProbeInstanceRead(in %in: i8, out out: i8) {
  // CHECK: %[[P:.+]] = hw.instance "producer" @ProbeProducer(in: %in: i8) -> (p: !probe.ref<i8>)
  %p = hw.instance "producer" @ProbeProducer(in: %in: i8) -> (p: !probe.ref<i8>)
  // CHECK: %[[V:.+]] = probe.read %[[P]] : <i8>
  %v = probe.read %p : <i8>
  hw.output %v : i8
}

// OPT-LABEL: hw.module @OptimizationBarrier
hw.module @OptimizationBarrier(in %a: i8, in %b: i8, out forwarded: i8,
                               out observed: i8) {
  // OPT: %[[VALUE:.+]] = comb.xor %a, %b : i8
  %value = comb.xor %a, %b : i8
  // OPT-NEXT: %[[FORWARDED:.+]], %[[REF:.+]] = probe.send %[[VALUE]] : i8
  %forwarded, %ref = probe.send %value : i8
  // OPT-NEXT: %[[OBSERVED:.+]] = probe.read %[[REF]] : <i8>
  %observed = probe.read %ref : <i8>
  // OPT-NEXT: hw.output %[[FORWARDED]], %[[OBSERVED]] : i8, i8
  hw.output %forwarded, %observed : i8, i8
}
