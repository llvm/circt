; RUN: firtool %s --format=fir  --lower-to-hw | FileCheck %s 

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
; CHECK:  hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "maskGran", "readUnderWrite", "writeUnderWrite", "writeClockIDs"]
; CHECK:  hw.module @FIRRTLMem_1_1_1_8_16_1_1_0_1_a(%ro_clock_0: i1, %ro_en_0: i1, %ro_addr_0: i4, %rw_clock_0: i1, %rw_en_0: i1, %rw_addr_0: i4, %rw_wmode_0: i1, %rw_wmask_0: i1, %rw_wdata_0: i8, %wo_clock_0: i1, %wo_en_0: i1, %wo_addr_0: i4, %wo_mask_0: i1, %wo_data_0: i8) -> (ro_data_0: i8, rw_rdata_0: i8) {
; CHECK-NEXT:    %[[vtrue:.+]] = hw.constant true
; CHECK-NEXT:    %[[vMemory:.+]] = sv.reg  : !hw.inout<uarray<16xi8>>
; CHECK-NEXT:    %[[v0:.+]] = sv.reg  : !hw.inout<i1>
; CHECK-NEXT:    %[[v1:.+]] = sv.read_inout %[[v0]] : !hw.inout<i1>
; CHECK-NEXT:    %[[v2:.+]] = sv.reg  : !hw.inout<i4>
; CHECK-NEXT:    sv.alwaysff(posedge %ro_clock_0)  {
; CHECK-NEXT:      sv.passign %[[v0]], %ro_en_0 : i1
; CHECK-NEXT:      sv.passign %[[v2]], %ro_addr_0 : i4
; CHECK-NEXT:    }
; CHECK-NEXT:    %[[v3:.+]] = sv.read_inout %[[v2]] : !hw.inout<i4>
; CHECK-NEXT:    %[[v4:.+]] = sv.array_index_inout %[[vMemory]][%[[v3]]] : !hw.inout<uarray<16xi8>>, i4
; CHECK-NEXT:    %[[v5:.+]] = sv.read_inout %[[v4]] : !hw.inout<i8>
; CHECK-NEXT:    %[[vx_i8:.+]] = sv.constantX : i8
; CHECK-NEXT:    %[[v6:.+]] = comb.mux %[[v1]], %[[v5]], %[[vx_i8]] : i8
; CHECK-NEXT:    %[[v7:.+]] = comb.xor %rw_wmode_0, %true : i1
; CHECK-NEXT:    %[[v8:.+]] = comb.and %rw_en_0, %[[v7]] : i1
; CHECK-NEXT:    %[[v9:.+]] = sv.array_index_inout %[[vMemory]][%rw_addr_0] : !hw.inout<uarray<16xi8>>, i4
; CHECK-NEXT:    %[[v10:.+]] = sv.read_inout %[[v9]] : !hw.inout<i8>
; CHECK-NEXT:    %[[v11:.+]] = comb.mux %[[v8]], %[[v10]], %[[vx_i8]] : i8
; CHECK-NEXT:    sv.alwaysff(posedge %rw_clock_0)  {
; CHECK-NEXT:      %[[vv12:.+]] = comb.and %rw_en_0, %rw_wmask_0, %rw_wmode_0 : i1
; CHECK-NEXT:      sv.if %[[vv12]]  {
; CHECK-NEXT:        sv.passign %[[v9]], %rw_wdata_0 : i8
; CHECK-NEXT:      }
; CHECK-NEXT:    }
; CHECK-NEXT:    sv.alwaysff(posedge %wo_clock_0)  {
; CHECK-NEXT:      %[[v12:.+]] = comb.and %wo_en_0, %wo_mask_0 : i1
; CHECK-NEXT:      sv.if %[[v12]]  {
; CHECK-NEXT:        %[[v13:.+]] = sv.array_index_inout %[[vMemory]][%wo_addr_0] : !hw.inout<uarray<16xi8>>, i4
; CHECK-NEXT:        sv.passign %[[v13]], %wo_data_0 : i8
; CHECK-NEXT:      }
; CHECK-NEXT:    }
; CHECK-NEXT:    hw.output %[[v6]], %[[v11]] : i8, i8
; CHECK-NEXT:  }
; CHECK-NEXT:  hw.module @Qux(%clock: i1, %rAddr: i4, %rEn: i1, %wAddr: i4, %wEn: i1, %wMask: i1, %wData: i8, %rwEn: i1, %rwMode: i1, %rwAddr: i4, %rwMask: i1, %rwDataIn: i8) -> (rData: i8, rwDataOut: i8) attributes {firrtl.moduleHierarchyFile = #hw.output_file<"testharness_hier.json", excludeFromFileList>} {
; CHECK-NEXT:    %memory.ro_data_0, %memory.rw_rdata_0 = hw.instance "memory" @FIRRTLMem_1_1_1_8_16_1_1_0_1_a(ro_clock_0: %clock: i1, ro_en_0: %rEn: i1, ro_addr_0: %rAddr: i4, rw_clock_0: %clock: i1, rw_en_0: %rwEn: i1, rw_addr_0: %rwAddr: i4, rw_wmode_0: %rwMode: i1, rw_wmask_0: %rwMask: i1, rw_wdata_0: %rwDataIn: i8, wo_clock_0: %clock: i1, wo_en_0: %wEn: i1, wo_addr_0: %wAddr: i4, wo_mask_0: %wMask: i1, wo_data_0: %wData: i8) -> (ro_data_0: i8, rw_rdata_0: i8)
; CHECK-NEXT:    hw.output %memory.ro_data_0, %memory.rw_rdata_0 : i8, i8
; CHECK-NEXT:  }
; CHECK-NEXT:}
