; RUN: firtool %s --format=fir  --ir-sv | FileCheck %s

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
;CHECK:    %[[memory_r_data:.+]] = sv.wire sym @__Qux__memory_r_data
;CHECK:    %[[v0:.+]] = sv.read_inout %[[memory_r_data]]
;CHECK:    %[[memory_0:.+]] = sv.reg
;CHECK:    %[[memory_1:.+]] = sv.reg
;CHECK:    %[[memory_2:.+]] = sv.reg
;CHECK:    %[[memory_3:.+]] = sv.reg
;CHECK:    %[[en:.+]] = sv.reg
;CHECK:    %[[addr:.+]] = sv.reg
;CHECK:    %[[v5:.+]] = sv.read_inout %[[addr]]
;CHECK:    %[[v6:.+]] = hw.array_create
;CHECK:    %[[v7:.+]] = hw.array_get %[[v6]][%[[v5]]]
;CHECK:    sv.assign %[[memory_r_data]], %[[v7]] : i8
;CHECK:    %[[memory_rw_rdata:.+]] = sv.wire sym @__Qux__memory_rw_rdata
;CHECK:    %[[v8:.+]] = sv.read_inout %memory_rw_rdata : !hw.inout<i8>
;CHECK:    %[[v9:.+]] = hw.array_get %[[v6]][%rwAddr] : !hw.array<4xi8>
;CHECK:    sv.assign %[[memory_rw_rdata]], %[[v9]] : i8
;CHECK:    sv.always posedge %clock {
;CHECK-NEXT:      sv.passign %[[en]], %rEn : i1
;CHECK-NEXT:      sv.passign %[[addr]], %rAddr : i2
;CHECK-NEXT:      sv.passign %[[memory_0]]
;CHECK-NEXT:      sv.passign %[[memory_1]]
;CHECK-NEXT:      sv.passign %[[memory_2]]
;CHECK-NEXT:      sv.passign %[[memory_3]]
;CHECK-NEXT:    }
;CHECK:    hw.output %[[v0]], %[[v8]]
