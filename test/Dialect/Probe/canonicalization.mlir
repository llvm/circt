// RUN: circt-opt %s --canonicalize --cse | FileCheck %s

// CHECK-LABEL: hw.module @DirectSendRead
// CHECK-NEXT: hw.output %in : i8
hw.module @DirectSendRead(in %in: i8, out out: i8) {
  %ref = probe.send %in : i8
  %value = probe.read %ref : <i8>
  hw.output %value : i8
}

// CHECK-LABEL: hw.module @UnusedSend
// CHECK-NOT: probe.send
hw.module @UnusedSend(in %in: i8, out out: i8) {
  %ref = probe.send %in : i8
  hw.output %in : i8
}
