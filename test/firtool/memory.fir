; RUN: firtool %s --format=fir --ir-sv | FileCheck %s

circuit Qux: %[[{
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~Qux|Qux"
  }, 
  {
    "class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"
  }]]
  module Qux: 
    input clock: Clock
    input rAddr: UInt<2>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wAddr: UInt<2>
    input wEn: UInt<1>
    input wMask: UInt<1>
    input wData: UInt<8>
    input rwEn: UInt<1>
    input rwMode: UInt<1>
    input rwAddr: UInt<2>
    input rwMask: UInt<1>
    input rwDataIn: UInt<8>
    output rwDataOut: UInt<8>

    mem memory:
      data-type => UInt<8>
      depth => 4
      reader => r
      writer => w
      readwriter => rw
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    ; All of these are unified together
    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    ; This is split
    rData <= memory.r.data

    ; All of these are unified together
    memory.w.clk <= clock
    memory.w.en <= wEn
    memory.w.addr <= wAddr
    ; These two are split
    memory.w.mask <= wMask
    memory.w.data <= wData

    ; All of these are unified together
    memory.rw.clk <= clock
    memory.rw.en <= rwEn
    memory.rw.addr <= rwAddr
    memory.rw.wmode <= rwMode
    ; All of these are split
    memory.rw.wmask <= rwMask
    memory.rw.wdata <= rwDataIn
    rwDataOut <= memory.rw.rdata



;CHECK-LABEL: hw.module @Qux
;CHECK:    %[[memory_0:.+]] = sv.reg
;CHECK:    %[[memory_1:.+]] = sv.reg
;CHECK:    %[[memory_2:.+]] = sv.reg
;CHECK:    %[[memory_3:.+]] = sv.reg
;CHECK:    %[[addr:.+]] = sv.reg
;CHECK:    %[[v4:.+]] = sv.read_inout %[[addr]]
;CHECK:    %[[v5:.+]] = hw.array_create
;CHECK:    %[[v6:.+]] = hw.array_get %[[v5]][%[[v4]]]
;CHECK:    %[[v7:.+]] = hw.array_get %[[v5]][%rwAddr] : !hw.array<4xi8>
;CHECK:    sv.always posedge %clock {
;CHECK-NEXT:      sv.passign %[[addr]], %rAddr : i2
;CHECK-NEXT:      sv.passign %[[memory_0]]
;CHECK-NEXT:      sv.passign %[[memory_1]]
;CHECK-NEXT:      sv.passign %[[memory_2]]
;CHECK-NEXT:      sv.passign %[[memory_3]]
;CHECK-NEXT:    }
;CHECK:    hw.output %[[v6]], %[[v7]]
