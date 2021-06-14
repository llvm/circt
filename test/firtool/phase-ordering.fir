; RUN: firtool %s --format=fir | FileCheck %s

; Temporary wires should not be introduced by type lowering, and if they are,
; they should be cleaned up by canonicalize.

; CHECK-LABEL: firrtl.module @Issue794(in %clock: !firrtl.clock,
; CHECK-NOT: firrtl.wire
; CHECK: }
circuit Issue794:
  module Issue794:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wAddr: UInt<4>[2]
    input wEn: UInt<1>[2]
    input wMask: UInt<1>[2]
    input wData: UInt<8>[2]

    mem memory:
      data-type => UInt<8>
      depth => 16
      reader => r
      writer => w0,w1
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    rData <= memory.r.data

    memory.w0.clk <= clock
    memory.w0.en <= wEn[0]
    memory.w0.addr <= wAddr[0]
    memory.w0.mask <= wMask[0]
    memory.w0.data <= wData[0]

    memory.w1.clk <= clock
    memory.w1.en <= wEn[1]
    memory.w1.addr <= wAddr[1]
    memory.w1.mask <= wMask[1]
    memory.w1.data <= wData[1]
