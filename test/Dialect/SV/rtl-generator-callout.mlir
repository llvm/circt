// RUN:  circt-opt -rtl-generator-callout='generator-executable=touch;FIRRTLMem_2_0_0_16_10_0_1_0.v'  %s | FileCheck %s
// We use a dummy generator executable, which creates an empty file. 
// If duh-mem is installed then:: circt-opt -rtl-generator-callout='generator-executable=./node_modules/.bin/duh-mem;fir' %s

module attributes {firrtl.mainModule = "top_mod"}  {
  rtl.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "readLatency", "writeLatency", "width", "readUnderWrite", "numReadWritePorts"]
  rtl.module.generated @FIRRTLMem_2_0_0_16_10_0_1_0, @FIRRTLMem(%rd_clock_0: i1, %rd_en_0: i1, %rd_addr_0: i4,
  %rd_clock_1: i1, %rd_en_1: i1, %rd_addr_1: i4) -> (%rd_data_0: i16, %rd_data_1: i16) attributes {depth = 10 : i64,
  numReadPorts = 2 : i32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : ui32, width = 16 : ui32, writeLatency = 1 : ui32, numReadWritePorts = 1 : ui32}

  // CHECK:  rtl.module.extern @FIRRTLMem_2_0_0_16_10_0_1_0(%rd_clock_0: i1, %rd_en_0: i1, %rd_addr_0: i4, %rd_clock_1: i1, %rd_en_1: i1, %rd_addr_1: i4) -> (%rd_data_0: i16, %rd_data_1: i16) attributes {Filename = "FIRRTLMem_2_0_0_16_10_0_1_0.v"}

  sv.verbatim "// Standard header to adapt well known macros to our needs."
  sv.verbatim ""
  rtl.module @top_mod(%clock: i1, %reset: i1, %r0en: i1) -> (%data: i16) {
    %c0_i16 = rtl.constant 0 : i16
    %false = rtl.constant false
    %c0_i4 = rtl.constant 0 : i4
    %tmp41.rd_data_0 = rtl.instance "tmp41" @FIRRTLMem_2_0_0_16_10_0_1_0(%clock, %r0en, %c0_i4, %clock, %r0en, %c0_i4, %false, %c0_i16) : (i1, i1, i4, i1, i1, i4, i1, i16) -> i16
    rtl.output %tmp41.rd_data_0 : i16
  }
}
