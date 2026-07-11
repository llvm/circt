// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: hw.module @Basic
hw.module @Basic(in %in: i8, out out: i8) {
  // CHECK: %[[P:.+]] = probe.send %in : i8
  %p = probe.send %in : i8
  // CHECK: %[[V:.+]] = probe.read %[[P]] : <i8>
  %v = probe.read %p : <i8>
  hw.output %v : i8
}

// CHECK-LABEL: hw.module @Clock
// CHECK-SAME: in %clock : !seq.clock
// CHECK-SAME: out out : !seq.clock
hw.module @Clock(in %clock: !seq.clock, out out: !seq.clock) {
  // CHECK: %[[P:.+]] = probe.send %clock : !seq.clock
  %p = probe.send %clock : !seq.clock
  // CHECK: %[[V:.+]] = probe.read %[[P]] : <!seq.clock>
  %v = probe.read %p : <!seq.clock>
  hw.output %v : !seq.clock
}

// CHECK-LABEL: hw.module @ProbeProducer
// CHECK-SAME: out p : !probe.ref<i8>
hw.module @ProbeProducer(in %in: i8, out p: !probe.ref<i8>) {
  %p = probe.send %in : i8
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
