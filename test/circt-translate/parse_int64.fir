; RUN: circt-translate %s --import-firrtl | FileCheck %s --strict-whitespace

circuit Foo:
  module Foo:
    mem memory:
      data-type => UInt<8>
      depth => 2147483648
      reader => r
      read-latency => 0
      write-latency => 1
      read-under-write => undefined
; CHECK: module  {
; CHECK:   firrtl.circuit "Foo" {
; CHECK:     firrtl.module @Foo() {
; CHECK:       %memory_r = firrtl.mem Undefined {depth = 2147483648 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<31>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<8>>
; CHECK:     }
; CHECK:   }
; CHECK: }
