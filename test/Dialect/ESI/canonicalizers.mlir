// RUN: circt-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL:  hw.module @TestDanglingFIFO() {
// CHECK-NEXT:     hw.output
hw.module @TestDanglingFIFO() {
  %c0 = hw.constant 0 : i1
  %chanOutput, %ready = esi.wrap.fifo %c0, %c0 : !esi.channel<i1, FIFO>
}

// CHECK-LABEL:  hw.module @TestDanglingValidReady() {
// CHECK-NEXT:     hw.output
hw.module @TestDanglingValidReady() {
  %c0 = hw.constant 0 : i1
  %chanOutput, %ready = esi.wrap.vr %c0, %c0 : i1
}
