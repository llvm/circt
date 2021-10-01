// RUN: circt-opt -hw-memory-sim %s | FileCheck %s

hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "readUnderWrite", "writeUnderWrite", "writeClockIDs"]

//CHECK-LABEL: @complex
hw.module @complex(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant true
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_2_4_0_0(ro_addr_0: %c0_i4: i4, ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1, 
     rw_wdata_0: %data0: i16,rw_wmask_0: %true: i1,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16, wo_mask_0: %true: i1) -> (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}

hw.module @complexMultiBit(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant 1 : i4
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi(ro_addr_0: %c0_i4: i4, ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1, 
     rw_wdata_0: %data0: i16,rw_wmask_0: %true: i4,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16, wo_mask_0: %true: i4) -> (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}
//CHECK-LABEL: @simple
hw.module @simple(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant true
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_0_1_0_0( ro_addr_0: %c0_i4: i4,ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1, 
     rw_wdata_0: %data0: i16, rw_wmask_0: %true: i1,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16, wo_mask_0: %true: i1) -> 
     (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}

//CHECK-LABEL: @WriteOrderedSameClock
hw.module @WriteOrderedSameClock(%clock: i1, %w0_addr: i4, %w0_en: i1, %w0_data: i8, %w0_mask: i1, %w1_addr: i4, %w1_en: i1, %w1_data: i8, %w1_mask: i1) {
  hw.instance "memory"
    @FIRRTLMemOneAlways(wo_addr_0: %w0_addr: i4, wo_en_0: %w0_en: i1,
      wo_clock_0: %clock: i1, wo_data_0: %w0_data: i8, wo_mask_0: %w0_mask: i1,
      wo_addr_1: %w1_addr: i4, wo_en_1: %w1_en: i1, wo_clock_1: %clock: i1,
       wo_data_1: %w1_data: i8,wo_mask_1: %w1_mask: i1) -> ()
  hw.output
}

//CHECK-LABEL: @WriteOrderedDifferentClock
hw.module @WriteOrderedDifferentClock(%clock: i1, %clock2: i1, %w0_addr: i4, %w0_en: i1, %w0_data: i8, %w0_mask: i1, %w1_addr: i4, %w1_en: i1, %w1_data: i8, %w1_mask: i1) {
  hw.instance "memory"
    @FIRRTLMemTwoAlways(wo_addr_0: %w0_addr: i4, wo_en_0: %w0_en: i1,
      wo_clock_0: %clock: i1, wo_data_0: %w0_data: i8, wo_mask_0: %w0_mask: i1,
      wo_addr_1: %w1_addr: i4, wo_en_1: %w1_en: i1, wo_clock_1: %clock2: i1,
      wo_data_1: %w1_data: i8, wo_mask_1: %w1_mask: i1) -> ()
  hw.output
}

hw.module.generated @FIRRTLMem_1_1_1_16_10_0_1_0_0, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %rw_wmask_0: i1,  %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16, %wo_mask_0: i1) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : ui32, width = 16 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 0 : i32}

//CHECK-LABEL: @FIRRTLMem_1_1_1_16_10_0_1_0_0
//CHECK:       %Memory0 = sv.reg  : !hw.inout<uarray<10xi16>>
//CHECK-NEXT:  %[[rslot:.+]] = sv.array_index_inout %Memory0[%ro_addr_0]
//CHECK-NEXT:  %[[read1:.+]] = sv.read_inout %[[rslot]]
//CHECK-NEXT:  %[[read:.+]] = comb.concat %[[read1]] : (i16) -> i16
//CHECK-NEXT:  %[[x:.+]] = sv.constantX
//CHECK-NEXT:  %[[readres:.+]] = comb.mux %ro_en_0, %[[read]], %[[x]]
//CHECK-NEXT:  %[[rw_wmask_0:.+]] = comb.extract %rw_wmask_0 from 0 : (i1) -> i1
//CHECK-NEXT:  %[[rw_wdata_0:.+]] = comb.extract %rw_wdata_0 from 0 : (i16) -> i16
//CHECK-NEXT:  %[[rwtmp:.+]] = sv.wire
//CHECK-NEXT:  %[[rwres:.+]] = sv.read_inout %[[rwtmp]]
//CHECK-NEXT:  %false = hw.constant false
//CHECK-NEXT:  %[[rwrcondpre:.+]] = comb.icmp eq %rw_wmode_0, %false
//CHECK-NEXT:  %[[rwrcond:.+]] = comb.and %rw_en_0, %[[rwrcondpre]]
//CHECK-NEXT:  %[[rwslot:.+]] = sv.array_index_inout %Memory0[%rw_addr_0]
//CHECK-NEXT:  %[[v11:.+]] = sv.read_inout %10 : !hw.inout<i16>
//CHECK-NEXT:  %[[rwdata:.+]] = comb.concat %[[v11]] : (i16) -> i16
//CHECK-NEXT:  %[[x2:.+]] = sv.constantX
//CHECK-NEXT:  %[[rwdata2:.+]] = comb.mux %[[rwrcond]], %[[rwdata]], %[[x2]]
//CHECK-NEXT:  sv.assign %[[rwtmp]], %[[rwdata2:.+]]
//CHECK-NEXT:    sv.alwaysff(posedge %rw_clock_0)  {
//CHECK-NEXT:      %[[rwwcondpre:.+]] = comb.and %[[rw_wmask_0]], %rw_wmode_0
//CHECK-NEXT:      %[[rwwcond:.+]] = comb.and %rw_en_0, %[[rwwcondpre]]
//CHECK-NEXT:      sv.if %[[rwwcond]]  {
//CHECK-NEXT:        sv.passign %[[rwslot]], %[[rw_wdata_0]]
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:  %[[v14:.+]] = comb.extract %wo_mask_0 from 0 : (i1) -> i1
//CHECK-NEXT:  %[[v15:.+]] = comb.extract %wo_data_0 from 0 : (i16) -> i16
//CHECK-NEXT:  sv.alwaysff(posedge %wo_clock_0)  {
//CHECK-NEXT:    %[[wcond:.+]] = comb.and %wo_en_0, %[[v14]]
//CHECK-NEXT:    sv.if %[[wcond]]  {
//CHECK-NEXT:      %[[wslot:.+]] = sv.array_index_inout %Memory0[%wo_addr_0]
//CHECK-NEXT:      sv.passign %[[wslot]], %[[v15]]
//CHECK-NEXT:    }
//CHECK-NEXT:  }
//CHECK-NEXT:  hw.output %[[readres]], %[[rwres]]

hw.module.generated @FIRRTLMem_1_1_1_16_10_2_4_0_0, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %rw_wmask_0: i1,  %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16, %wo_mask_0: i1) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : ui32, width = 16 : ui32, writeClockIDs = [], writeLatency = 4 : ui32, writeUnderWrite = 0 : i32}

//CHECK-LABEL: @FIRRTLMem_1_1_1_16_10_2_4_0_0
//COM: This produces a lot of output, we check one field's pipeline
//CHECK:         %Memory0 = sv.reg  : !hw.inout<uarray<10xi16>>
//CHECK:         sv.alwaysff(posedge %ro_clock_0)  {
//CHECK-NEXT:      sv.passign %0, %ro_en_0 : i1
//CHECK-NEXT:    }
//CHECK-NEXT:    %1 = sv.read_inout %0 : !hw.inout<i1>
//CHECK-NEXT:    %2 = sv.reg  : !hw.inout<i1>
//CHECK-NEXT:    sv.alwaysff(posedge %ro_clock_0)  {
//CHECK-NEXT:      sv.passign %2, %1 : i1
//CHECK-NEXT:    }
//CHECK-NEXT:    %3 = sv.read_inout %2 : !hw.inout<i1>
//CHECK-NEXT:    %4 = sv.reg  : !hw.inout<i4>
//CHECK-NEXT:    sv.alwaysff(posedge %ro_clock_0)  {
//CHECK-NEXT:      sv.passign %4, %ro_addr_0 : i4
//CHECK-NEXT:    }
//CHECK-NEXT:    %5 = sv.read_inout %4 : !hw.inout<i4>
//CHECK-NEXT:    %6 = sv.reg  : !hw.inout<i4>
//CHECK-NEXT:    sv.alwaysff(posedge %ro_clock_0)  {
//CHECK-NEXT:      sv.passign %6, %5 : i4
//CHECK-NEXT:    }
//CHECK-NEXT:    %7 = sv.read_inout %6 : !hw.inout<i4>
//CHECK-NEXT:    %8 = sv.array_index_inout %Memory0[%7] : !hw.inout<uarray<10xi16>>, i4

hw.module.generated @FIRRTLMemOneAlways, @FIRRTLMem( %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1,%wo_data_0: i8, %wo_mask_0: i1, %wo_addr_1: i4,  %wo_en_1: i1, %wo_clock_1: i1, %wo_data_1: i8, %wo_mask_1: i1) attributes {depth = 16 : i64, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [0 : i32, 0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

//CHECK-LABEL: @FIRRTLMemOneAlways
//CHECK-COUNT-1:  sv.alwaysff
//CHECK-NOT:      sv.alwaysff

hw.module.generated @FIRRTLMemTwoAlways, @FIRRTLMem( %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1,%wo_data_0: i8, %wo_mask_0: i1, %wo_addr_1: i4,  %wo_en_1: i1, %wo_clock_1: i1, %wo_data_1: i8, %wo_mask_1: i1) attributes {depth = 16 : i64, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [0 : i32, 1 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

//CHECK-LABEL: @FIRRTLMemTwoAlways
//CHECK-COUNT-2:  sv.alwaysff
//CHECK-NOT:      sv.alwaysff

hw.module.generated @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0:
i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %rw_wmask_0: i4,  %wo_addr_0: i4, %wo_en_0: i1,
%wo_clock_0: i1, %wo_data_0: i16, %wo_mask_0: i4) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64,
numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32,maskGran = 4 :ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : ui32, width = 16 : ui32, writeClockIDs = [], writeLatency = 4 : ui32, writeUnderWrite = 0 : i32}

// CHECK-LABEL:  hw.module @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1, %rw_addr_0: i4, %rw_en_0: i1, %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16, %rw_wmask_0: i4, %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16, %wo_mask_0: i4) -> (ro_data_0: i16, rw_rdata_0: i16) {
// CHECK-NEXT:    %Memory0 = sv.reg  : !hw.inout<uarray<10xi4>>
// CHECK-NEXT:    %Memory1 = sv.reg  : !hw.inout<uarray<10xi4>>
// CHECK-NEXT:    %Memory2 = sv.reg  : !hw.inout<uarray<10xi4>>
// CHECK-NEXT:    %Memory3 = sv.reg  : !hw.inout<uarray<10xi4>>
// CHECK:    %8 = sv.array_index_inout %Memory0[%7] : !hw.inout<uarray<10xi4>>, i4
// CHECK-NEXT:    %9 = sv.read_inout %8 : !hw.inout<i4>
// CHECK-NEXT:    %10 = sv.array_index_inout %Memory1[%7] : !hw.inout<uarray<10xi4>>, i4
// CHECK-NEXT:    %11 = sv.read_inout %10 : !hw.inout<i4>
// CHECK-NEXT:    %12 = sv.array_index_inout %Memory2[%7] : !hw.inout<uarray<10xi4>>, i4
// CHECK-NEXT:    %13 = sv.read_inout %12 : !hw.inout<i4>
// CHECK-NEXT:    %14 = sv.array_index_inout %Memory3[%7] : !hw.inout<uarray<10xi4>>, i4
// CHECK-NEXT:    %15 = sv.read_inout %14 : !hw.inout<i4>
// CHECK-NEXT:    %16 = comb.concat %9, %11, %13, %15 : (i4, i4, i4, i4) -> i16
// CHECK:   sv.alwaysff(posedge %rw_clock_0)  {
// CHECK:     sv.passign %42, %rw_wmask_0 : i4
// CHECK:   }
// CHECK:   %43 = sv.read_inout %42 : !hw.inout<i4>
// CHECK:   %44 = sv.reg  : !hw.inout<i4>
// CHECK:   sv.alwaysff(posedge %rw_clock_0)  {
// CHECK:     sv.passign %44, %43 : i4
// CHECK:   }
// CHECK:   %45 = sv.read_inout %44 : !hw.inout<i4>
// CHECK:   %46 = sv.reg  : !hw.inout<i4>
// CHECK:   sv.alwaysff(posedge %rw_clock_0)  {
// CHECK:     sv.passign %46, %45 : i4
// CHECK:   }
// CHECK:   %47 = sv.read_inout %46 : !hw.inout<i4>
// CHECK:   %48 = comb.extract %47 from 0 : (i4) -> i1
// CHECK:   %49 = comb.extract %41 from 0 : (i16) -> i4
// CHECK:   %50 = comb.extract %47 from 1 : (i4) -> i1
// CHECK:   %51 = comb.extract %41 from 4 : (i16) -> i4
// CHECK:   %52 = comb.extract %47 from 2 : (i4) -> i1
// CHECK:   %53 = comb.extract %41 from 8 : (i16) -> i4
// CHECK:   %54 = comb.extract %47 from 3 : (i4) -> i1
// CHECK:   %55 = comb.extract %41 from 12 : (i16) -> i4
// CHECK:   %60 = sv.array_index_inout %Memory0[%23] : !hw.inout<uarray<10xi4>>, i4
// CHECK:   %61 = sv.read_inout %60 : !hw.inout<i4>
// CHECK:   %62 = sv.array_index_inout %Memory1[%23] : !hw.inout<uarray<10xi4>>, i4
// CHECK:   %63 = sv.read_inout %62 : !hw.inout<i4>
// CHECK:   %64 = sv.array_index_inout %Memory2[%23] : !hw.inout<uarray<10xi4>>, i4
// CHECK:   %65 = sv.read_inout %64 : !hw.inout<i4>
// CHECK:   %66 = sv.array_index_inout %Memory3[%23] : !hw.inout<uarray<10xi4>>, i4
// CHECK:   %67 = sv.read_inout %66 : !hw.inout<i4>
// CHECK:   %68 = comb.concat %61, %63, %65, %67 : (i4, i4, i4, i4) -> i16
// CHECK:   %94 = comb.extract %93 from 0 : (i4) -> i1
// CHECK:   %95 = comb.extract %87 from 0 : (i16) -> i4
// CHECK:   %96 = comb.extract %93 from 1 : (i4) -> i1
// CHECK:   %97 = comb.extract %87 from 4 : (i16) -> i4
// CHECK:   %98 = comb.extract %93 from 2 : (i4) -> i1
// CHECK:   %99 = comb.extract %87 from 8 : (i16) -> i4
// CHECK:   %100 = comb.extract %93 from 3 : (i4) -> i1
// CHECK:   %101 = comb.extract %87 from 12 : (i16) -> i4
// CHECK:   sv.alwaysff(posedge %wo_clock_0)  {
// CHECK:     %102 = comb.and %81, %94 : i1
// CHECK:     sv.if %102  {
// CHECK:       %106 = sv.array_index_inout %Memory0[%75] : !hw.inout<uarray<10xi4>>, i4
// CHECK:       sv.passign %106, %95 : i4
// CHECK:     }
// CHECK:     %103 = comb.and %81, %96 : i1
// CHECK:     sv.if %103  {
// CHECK:       %106 = sv.array_index_inout %Memory1[%75] : !hw.inout<uarray<10xi4>>, i4
// CHECK:       sv.passign %106, %97 : i4
// CHECK:     }
// CHECK:     %104 = comb.and %81, %98 : i1
// CHECK:     sv.if %104  {
// CHECK:       %106 = sv.array_index_inout %Memory2[%75] : !hw.inout<uarray<10xi4>>, i4
// CHECK:       sv.passign %106, %99 : i4
// CHECK:     }
// CHECK:     %105 = comb.and %81, %100 : i1
// CHECK:     sv.if %105  {
// CHECK:       %106 = sv.array_index_inout %Memory3[%75] : !hw.inout<uarray<10xi4>>, i4
// CHECK:       sv.passign %106, %101 : i4
