; RUN: firtool %s --format=fir  --ir-hw | FileCheck %s 
; RUN: firtool %s --format=fir  --ir-hw --ignore-read-enable-mem | FileCheck --check-prefix=READ %s 

circuit Qux:
  module Qux:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wAddr: UInt<4>
    input wEn: UInt<1>
    input wMask: UInt<1>
    input wData: UInt<8>
    input rwEn: UInt<1>
    input rwMode: UInt<1>
    input rwAddr: UInt<4>
    input rwMask: UInt<1>
    input rwDataIn: UInt<8>
    output rwDataOut: UInt<8>

    mem memory:
      data-type => UInt<8>
      depth => 16
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


; READ-LABEL:  hw.module @memory_ext
; READ:    %[[vtrue:.+]] = hw.constant true
; READ:    %[[vMemory:.+]] = sv.reg  : !hw.inout<uarray<16xi8>>
; READ-NEXT:    %[[v2:.+]] = sv.reg  : !hw.inout<i4>
; READ-NEXT:    sv.always posedge %R0_clk {
; READ-NEXT:      sv.if %R0_en {
; READ-NEXT:        sv.passign %[[v2]], %R0_addr : i4
; READ-NEXT:      }
; READ-NEXT:    }
; READ-NEXT:    %[[v3:.+]] = sv.read_inout %[[v2]] : !hw.inout<i4>
; READ-NEXT:    %[[v4:.+]] = sv.array_index_inout %[[vMemory]][%[[v3]]] : !hw.inout<uarray<16xi8>>, i4
; READ-NEXT:    %[[v6:.+]] = sv.read_inout %[[v4]] : !hw.inout<i8>
; READ:    hw.output %[[v6]]

; CHECK:  hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "maskGran", "readUnderWrite", "writeUnderWrite", "writeClockIDs"]
; CHECK:  hw.module @memory_ext(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %RW0_addr: i4, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i8, %RW0_wmask: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W0_mask: i1) -> (R0_data: i8, RW0_rdata: i8) {
; CHECK:    %[[vtrue:.+]] = hw.constant true
; CHECK-NEXT:    %[[vMemory:.+]] = sv.reg  : !hw.inout<uarray<16xi8>>
; CHECK-NEXT:    %[[v0:.+]] = sv.reg  : !hw.inout<i1>
; CHECK-NEXT:    %[[v1:.+]] = sv.read_inout %[[v0]] : !hw.inout<i1>
; CHECK-NEXT:    %[[v2:.+]] = sv.reg  : !hw.inout<i4>
; CHECK-NEXT:    sv.always posedge %R0_clk {
; CHECK-NEXT:      sv.passign %[[v0]], %R0_en : i1
; CHECK-NEXT:      sv.passign %[[v2]], %R0_addr : i4
; CHECK-NEXT:    }
; CHECK-NEXT:    %[[v3:.+]] = sv.read_inout %[[v2]] : !hw.inout<i4>
; CHECK-NEXT:    %[[v4:.+]] = sv.array_index_inout %[[vMemory]][%[[v3]]] : !hw.inout<uarray<16xi8>>, i4
; CHECK-NEXT:    %[[v5:.+]] = sv.read_inout %[[v4]] : !hw.inout<i8>
; CHECK-NEXT:    %[[vx_i8:.+]] = sv.constantX : i8
; CHECK-NEXT:    %[[v6:.+]] = comb.mux %[[v1]], %[[v5]], %[[vx_i8]] : i8
; CHECK-NEXT:    %[[v7:.+]] = comb.xor %RW0_wmode, %true : i1
; CHECK-NEXT:    %[[v8:.+]] = comb.and %RW0_en, %[[v7]] : i1
; CHECK-NEXT:    %[[v9:.+]] = sv.array_index_inout %[[vMemory]][%RW0_addr] : !hw.inout<uarray<16xi8>>, i4
; CHECK-NEXT:    %[[v10:.+]] = sv.read_inout %[[v9]] : !hw.inout<i8>
; CHECK-NEXT:    %[[v11:.+]] = comb.mux %[[v8]], %[[v10]], %[[vx_i8]] : i8
; CHECK-NEXT:    sv.always posedge %RW0_clk {
; CHECK-NEXT:      %[[vv12:.+]] = comb.and %RW0_en, %RW0_wmask, %RW0_wmode : i1
; CHECK-NEXT:      sv.if %[[vv12]]  {
; CHECK-NEXT:        sv.passign %[[v9]], %RW0_wdata : i8
; CHECK-NEXT:      }
; CHECK-NEXT:    }
; CHECK-NEXT:    sv.always posedge %W0_clk {
; CHECK-NEXT:      %[[v12:.+]] = comb.and %W0_en, %W0_mask : i1
; CHECK-NEXT:      sv.if %[[v12]]  {
; CHECK-NEXT:        %[[v13:.+]] = sv.array_index_inout %[[vMemory]][%W0_addr] : !hw.inout<uarray<16xi8>>, i4
; CHECK-NEXT:        sv.passign %[[v13]], %W0_data : i8
; CHECK-NEXT:      }
; CHECK-NEXT:    }
; CHECK-NEXT:    hw.output %[[v6]], %[[v11]] : i8, i8
; CHECK-NEXT:  }
; CHECK-NEXT:  hw.module @Qux(%clock: i1, %rAddr: i4, %rEn: i1, %wAddr: i4, %wEn: i1, %wMask: i1, %wData: i8, %rwEn: i1, %rwMode: i1, %rwAddr: i4, %rwMask: i1, %rwDataIn: i8) -> (rData: i8, rwDataOut: i8) {
; CHECK-NEXT:    %memory.R0_data, %memory.RW0_rdata = hw.instance "memory" @memory_ext(R0_addr: %rAddr: i4, R0_en: %rEn: i1, R0_clk: %clock: i1, RW0_addr: %rwAddr: i4, RW0_en: %rwEn: i1, RW0_clk: %clock: i1, RW0_wmode: %rwMode: i1, RW0_wdata: %rwDataIn: i8, RW0_wmask: %rwMask: i1, W0_addr: %wAddr: i4, W0_en: %wEn: i1, W0_clk: %clock: i1, W0_data: %wData: i8, W0_mask: %wMask: i1) -> (R0_data: i8, RW0_rdata: i8)
; CHECK-NEXT:    hw.output %memory.R0_data, %memory.RW0_rdata : i8, i8
; CHECK-NEXT:  }
; CHECK-NEXT:}
