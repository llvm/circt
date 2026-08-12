// RUN: circt-opt %s --canonicalize --cse | FileCheck %s

// CHECK-LABEL: hw.module @UnusedSend
// CHECK-NOT: probe.send
hw.module @UnusedSend(in %in: i8, out out: i8) {
  %ref = probe.send %in : i8
  hw.output %in : i8
}
