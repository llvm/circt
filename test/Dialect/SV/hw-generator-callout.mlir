// RUN:  circt-opt -hw-generator-callout='schema-name=FIRRTL_Memory generator-executable=echo generator-executable-arguments=file1.v,file2.v,file3.v,file4.v ' %s | FileCheck %s

builtin.module attributes {firrtl.mainModule = "Top"}  {

// CHECK:   hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "readUnderWrite"]
// CHECK:   hw.module.extern @FIRRTLMem_1_1_1_42_12_0_1_0(%ro_clock_0: i1, %ro_en_0: i1, %ro_addr_0: i4, %rw_clock_0: i1, %rw_en_0: i1, %rw_addr_0: i4, %rw_wmode_0: i1, %rw_wmask_0: i1, %rw_wdata_0: i42, %wo_clock_0: i1, %wo_en_0: i1, %wo_addr_0: i4, %wo_mask_0: i1, %wo_data_0: i42) -> (%ro_data_0: i42, %rw_rdata_0: i42) attributes {filenames = "file1.v,file2.v,file3.v,file4.v --moduleName FIRRTLMem_1_1_1_42_12_0_1_0 --hierarchichalName Top._A._M.mem --depth 12 --numReadPorts 1 --numWritePorts 0 --numReadWritePorts 1 --readLatency 1 --writeLatency 1 --width 42 --readUnderWrite 0"}
  hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "readUnderWrite"]
  hw.module.generated @FIRRTLMem_1_1_1_42_12_0_1_0, @FIRRTLMem(%ro_clock_0: i1, %ro_en_0: i1, %ro_addr_0: i4,
  %rw_clock_0: i1, %rw_en_0: i1, %rw_addr_0: i4, %rw_wmode_0: i1, %rw_wmask_0: i1, %rw_wdata_0: i42, %wo_clock_0: i1,
  %wo_en_0: i1, %wo_addr_0: i4, %wo_mask_0: i1, %wo_data_0: i42) -> (%ro_data_0: i42, %rw_rdata_0: i42) attributes
  {depth = 12 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 1 :
  ui32, readUnderWrite = 0 : ui32, verificationData = {baseAddress = 2147516416 : i64, dataBits = 8 : i64, eccBits = 0 :
  i64, eccIndices = [ 32, 33, 34, 35, 36, 37, 38], eccScheme = "none"}, width = 42 : ui32, writeLatency = 1 : ui32}

  hw.module @Top(%clock1: i1, %clock2: i1, %inpred: i1, %indata: i42) -> (%result: i42, %result2: i42) {
    %1, %2 = hw.instance "_A" @A(%clock1, %clock2, %inpred, %indata) : (i1, i1, i1, i42) -> (i42, i42)
    hw.output %1, %2 : i42, i42
  }
  hw.module @A(%clock1: i1, %clock2: i1, %inpred: i1, %indata: i42) -> (%result: i42, %result2: i42) {
    %1, %2 = hw.instance "_M" @MemSimple(%clock1, %clock2, %inpred, %indata) : (i1, i1, i1, i42) -> (i42, i42)
    hw.output %1, %2 : i42, i42
  }
  hw.module @MemSimple(%clock1: i1, %clock2: i1, %inpred: i1, %indata: i42) -> (%result: i42, %result2: i42) {
    %c0_i3 = hw.constant 0 : i3
    %true = hw.constant true
    %false = hw.constant false
    %.rw.wdata.wire = sv.wire  : !hw.inout<i42>
    %0 = sv.read_inout %.rw.wdata.wire : !hw.inout<i42>
    %_M.ro_data_0, %_M.rw_rdata_0 = hw.instance "mem" @FIRRTLMem_1_1_1_42_12_0_1_0(%clock1, %true, %c0_i4, %clock1, %true, %c0_i4_0, %true, %true, %0, %clock2, %inpred, %c0_i4_1, %true, %indata) : (i1, i1, i4, i1, i1, i4, i1, i1, i42, i1, i1, i4, i1, i42) -> (i42, i42)
    %c0_i4 = hw.constant 0 : i4
    %c0_i4_0 = hw.constant 0 : i4
    %c0_i4_1 = hw.constant 0 : i4
    hw.output %_M.ro_data_0, %_M.rw_rdata_0 : i42, i42
  }
}
