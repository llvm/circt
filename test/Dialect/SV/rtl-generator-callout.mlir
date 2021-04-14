// RUN:  circt-opt -rtl-generator-callout='generator-executable=echo generator-executable-arguments=file1.v,file2.v,file3.v,file4.v;-n generator-executable-output-filename=litTestout' %s | FileCheck %s

module attributes {firrtl.mainModule = "top_mod"}  {
  rtl.generator.schema @SchemaVar, "Schema_Name", ["port1", "port2"]
  rtl.module.generated @sampleModuleName, @SchemaVar(%rd_clock_0: i1, %rd_en_0: i1, %rd_addr_0: i4,
  %rd_clock_1: i1, %rd_en_1: i1, %rd_addr_1: i4) -> (%rd_data_0: i16, %rd_data_1: i16) attributes {port1 = 10 : i64, port2 = 2 : i32}

  // CHECK:  rtl.module.extern @sampleModuleName(%rd_clock_0: i1, %rd_en_0: i1, %rd_addr_0: i4, %rd_clock_1: i1, %rd_en_1: i1, %rd_addr_1: i4) -> (%rd_data_0: i16, %rd_data_1: i16) attributes {filenames =  "file1.v,file2.v,file3.v,file4.v --moduleName sampleModuleName --port1 10 --port2 2  "}

  sv.verbatim "// Standard header to adapt well known macros to our needs."
  sv.verbatim ""
  rtl.module @top_mod(%clock: i1, %reset: i1, %r0en: i1) -> (%data: i16) {
    %c0_i16 = rtl.constant 0 : i16
    %false = rtl.constant false
    %c0_i4 = rtl.constant 0 : i4
    %tmp41.rd_data_0 = rtl.instance "tmp41" @sampleModuleName(%clock, %r0en, %c0_i4, %clock, %r0en, %c0_i4, %false, %c0_i16) : (i1, i1, i4, i1, i1, i4, i1, i16) -> i16
    rtl.output %tmp41.rd_data_0 : i16
  }
}
