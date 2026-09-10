// RUN:  circt-opt -hw-generator-callout='schema-name=Schema_Name generator-executable=echo generator-executable-arguments=file1.v,file2.v,file3.v,file4.v ' %s | FileCheck %s

module attributes {firrtl.mainModule = "top_mod"}  {
  hw.generator.schema @SchemaVar, "Schema_Name", ["port1", "port2"]
  hw.module.generated @sampleModuleName, @SchemaVar(in %rd_clock_0: i1, in %rd_en_0: i1, in %rd_addr_0: i4,
  in %rd_clock_1: i1, in %rd_en_1: i1, in %rd_addr_1: i4, out rd_data_0: i16, out rd_data_1: i16) attributes {port1 = 10 : i64, port2 = 2 : i32}

  // CHECK:  hw.module.extern @sampleModuleName(
  // CHECK-SAME: in %rd_clock_0 : i1, 
  // CHECK-SAME: in %rd_en_0 : i1, 
  // CHECK-SAME: in %rd_addr_0 : i4, 
  // CHECK-SAME: in %rd_clock_1 : i1, 
  // CHECK-SAME: in %rd_en_1 : i1, 
  // CHECK-SAME: in %rd_addr_1 : i4,
  // CHECK-SAME: out rd_data_0 : i16, 
  // CHECK-SAME: out rd_data_1 : i16
  // CHECK-SAME: ) attributes {filenames =  "file1.v,file2.v,file3.v,file4.v --moduleName sampleModuleName --port1 10 --port2 2"}

  hw.generator.schema @SchemaVar1, "Schema_Name1", ["port1", "port2"]
  hw.module.generated @sampleModuleName1, @SchemaVar1(in %rd_clock_0: i1, in %rd_en_0: i1, in %rd_addr_0: i4, in %rd_clock_1: i1, in %rd_en_1: i1, in %rd_addr_1: i4, out rd_data_0: i16, out rd_data_1: i16) attributes {port1 = 10 : i64, port2 = 2 : i32}
  // CHECK: hw.module.generated @sampleModuleName1, @SchemaVar1(
  // CHECK-SAME: in %rd_clock_0 : i1, 
  // CHECK-SAME: in %rd_en_0 : i1, 
  // CHECK-SAME: in %rd_addr_0 : i4, 
  // CHECK-SAME: in %rd_clock_1 : i1, 
  // CHECK-SAME: in %rd_en_1 : i1, 
  // CHECK-SAME: in %rd_addr_1 : i4,
  // CHECK-SAME: out rd_data_0 : i16, 
  // CHECK-SAME: out rd_data_1 : i16
  // CHECK-SAME: ) attributes {port1 = 10 : i64, port2 = 2 : i32}
}
