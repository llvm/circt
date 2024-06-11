// RUN: circt-opt --firrtl-emit-metadata="repl-seq-mem=true repl-seq-mem-file=dut.conf" -split-input-file %s | FileCheck %s

firrtl.circuit "empty" {
  firrtl.module @empty() {
  }
}
// CHECK-LABEL: firrtl.circuit "empty"   {
// CHECK-NEXT:    firrtl.module @empty() {
// CHECK-NEXT:    }
// CHECK-NEXT:    emit.file "metadata{{/|\\\\}}seq_mems.json" {
// CHECK-NEXT:      sv.verbatim "[]"
// CHECK-NEXT:    }
// CHECK-NEXT:    emit.file "dut.conf" {
// CHECK-NEXT:      sv.verbatim ""
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// Memory metadata om class should not be created.
// CHECK-NOT: om.class @MemorySchema

// -----

//===----------------------------------------------------------------------===//
// RetimeModules
//===----------------------------------------------------------------------===//

firrtl.circuit "retime0" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = "retime_modules.json"
}]} {

  firrtl.module @retime0() attributes { annotations = [{
      class = "freechips.rocketchip.util.RetimeModuleAnnotation"
  }]} { }

  firrtl.module @retime1() { }

  firrtl.module @retime2() attributes { annotations = [{
      class = "freechips.rocketchip.util.RetimeModuleAnnotation"
  }]} { }
}
// CHECK-LABEL: firrtl.circuit "retime0"   {
// CHECK:         firrtl.module @retime0(out %metadataObj: !firrtl.any
// CHECK:           [[SIFIVE_METADATA:%.+]] = firrtl.object @SiFive_Metadata
// CHECK:           [[METADATA_OBJ:%.+]] = firrtl.object.anyref_cast [[SIFIVE_METADATA]]
// CHECK:           propassign %metadataObj, [[METADATA_OBJ]]
// CHECK:         firrtl.module @retime1() {
// CHECK:         firrtl.module @retime2()

// CHECK:    firrtl.class @RetimeModulesSchema(in %[[moduleName_in:.+]]: !firrtl.string, out %moduleName: !firrtl.string) {
// CHECK:    firrtl.propassign %moduleName, %[[moduleName_in]]

// CHECK:  firrtl.class @RetimeModulesMetadata
// CHECK-SAME: (out %[[retime0_field:[a-zA-Z][a-zA-Z0-9_]*]]: !firrtl.class<@RetimeModulesSchema
// CHECK:    %[[v0:.+]] = firrtl.string "retime0"
// CHECK:    %retime0 = firrtl.object @RetimeModulesSchema
// CHECK-NEXT:    %[[v1:.+]] = firrtl.object.subfield %retime0
// CHECK-NEXT:    firrtl.propassign %[[v1]], %[[v0]] : !firrtl.string
// CHECK-NEXT:    firrtl.propassign %[[retime0_field]], %retime0
// CHECK:    %2 = firrtl.string "retime2"
// CHECK:    %retime2 = firrtl.object @RetimeModulesSchema
// CHECK:    firrtl.object.subfield %retime2
// CHECK:    firrtl.propassign %[[retime2field:.+]], %retime2

// CHECK:               emit.file "retime_modules.json" {
// CHECK-NEXT{LITERAL}:   sv.verbatim "[\0A \22{{0}}\22,\0A \22{{1}}\22\0A]" {symbols = [@retime0, @retime2]}
// CHECK-NEXT:          }

// -----

//===----------------------------------------------------------------------===//
// SitestBlackbox
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "DUTBlackboxes" {
firrtl.circuit "DUTBlackboxes" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
    filename = "dut_blackboxes.json"
  }]} {
  firrtl.module @DUTBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  }
// CHECK-NOT:  emit.file ""
// CHECK:      emit.file "dut_blackboxes.json" {
// CHECK-NEXT:   emit.verbatim "[]"
// CHECK-NEXT: }
// CHECK-NOT:  emit.file ""
}

// -----

// CHECK-LABEL: firrtl.circuit "TestBlackboxes"  {
firrtl.circuit "TestBlackboxes" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
    filename = "test_blackboxes.json"
  }]} {
  firrtl.module @TestBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  }
// CHECK-NOT:  emit.file ""
// CHECK:      emit.file "test_blackboxes.json" {
// CHECK-NEXT:   emit.verbatim "[]"
// CHECK-NEXT: }
// CHECK-NOT:  emit.file ""
}

// -----

// CHECK-LABEL: firrtl.circuit "BasicBlackboxes"   {
firrtl.circuit "BasicBlackboxes" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
    filename = "dut_blackboxes.json"
  }, {
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
    filename = "test_blackboxes.json"
  }]} {

  firrtl.module @BasicBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    firrtl.instance test @DUTBlackbox_0()
    firrtl.instance test @DUTBlackbox_1()
    firrtl.instance test @DUTBlackbox_2()
  }

  // These should all be ignored.
  firrtl.extmodule @ignored0() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno"}], defname = "ignored0"}
  firrtl.extmodule @ignored1() attributes {annotations = [{class = "firrtl.transforms.BlackBoxPathAnno"}], defname = "ignored1"}
  firrtl.extmodule @ignored2() attributes {annotations = [{class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}], defname = "ignored2"}
  firrtl.extmodule @ignored3() attributes {annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.blackbox", id = 4 : i64}], defname = "ignored3"}
  firrtl.extmodule @ignored4() attributes {annotations = [{class = "firrtl.transforms.BlackBox"}], defname = "ignored4"}

// CHECK:    firrtl.class @SitestBlackBoxModulesSchema(in %[[moduleName_in:.+]]: !firrtl.string, out %moduleName: !firrtl.string) {
// CHECK:      firrtl.propassign %moduleName, %[[moduleName_in]]
// CHECK:    }

// CHECK:    firrtl.class @SitestBlackBoxMetadata(out %TestBlackbox_field: !firrtl.class<@SitestBlackBoxModulesSchema(in moduleName_in: !firrtl.string, out moduleName: !firrtl.string)>, out %DUTBlackbox_0_field: !firrtl.class<@SitestBlackBoxModulesSchema(in moduleName_in: !firrtl.string, out moduleName: !firrtl.string)>, out %DUTBlackbox_1_field: !firrtl.class<@SitestBlackBoxModulesSchema(in moduleName_in: !firrtl.string, out moduleName: !firrtl.string)>) attributes {portAnnotations = []} {
// CHECK:      firrtl.string "TestBlackbox"
// CHECK:      %TestBlackbox = firrtl.object @SitestBlackBoxModulesSchema
// CHECK:      firrtl.propassign %TestBlackbox_field, %TestBlackbox
// CHECK:      firrtl.string "DUTBlackbox2"
// CHECK:      %DUTBlackbox_0 = firrtl.object @SitestBlackBoxModulesSchema
// CHECK:      firrtl.propassign %DUTBlackbox_0_field, %DUTBlackbox_0
// CHECK:      firrtl.string "DUTBlackbox1"
// CHECK:      %DUTBlackbox_1 = firrtl.object @SitestBlackBoxModulesSchema
// CHECK:      firrtl.propassign %DUTBlackbox_1_field, %DUTBlackbox_1
// CHECK:    }
  // Gracefully handle missing defnames.
  firrtl.extmodule @NoDefName()

  firrtl.extmodule @TestBlackbox() attributes {defname = "TestBlackbox"}

  // CHECK:               emit.file "test_blackboxes.json" {
  // CHECK-NEXT{LITERAL}:   emit.verbatim "[\0A \22TestBlackbox\22\0A]"
  // CHECK-NEXT:          }

  // Should be de-duplicated and sorted.
  firrtl.extmodule @DUTBlackbox_0() attributes {defname = "DUTBlackbox2"}
  firrtl.extmodule @DUTBlackbox_1() attributes {defname = "DUTBlackbox1"}
  firrtl.extmodule @DUTBlackbox_2() attributes {defname = "DUTBlackbox1"}
  // CHECK:               emit.file "dut_blackboxes.json" {
  // CHECK-NEXT{LITERAL}:   emit.verbatim "[\0A \22DUTBlackbox1\22,\0A \22DUTBlackbox2\22\0A]"
  // CHECK-NEXT:          }
}

// -----

//===----------------------------------------------------------------------===//
// MemoryMetadata
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top"
{
  firrtl.module @top() { }
  // When there are no memories, we still need to emit the memory metadata.

  // CHECK:      emit.file "metadata{{/|\\\\}}seq_mems.json" {
  // CHECK-NEXT:   sv.verbatim "[]"
  // CHECK-NEXT: }

  // CHECK:      emit.file "dut.conf" {
  // CHECK-NEXT:   sv.verbatim ""
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: firrtl.circuit "OneMemory"
firrtl.circuit "OneMemory" {
  firrtl.module @OneMemory() {
    %0:5= firrtl.instance MWrite_ext sym @MWrite_ext_0  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>, in user_input: !firrtl.uint<5>)
  }
  firrtl.memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>, in user_input: !firrtl.uint<5>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [{direction = "input", name = "user_input", width = 5 : ui32}], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}


  // CHECK:  firrtl.class @MemorySchema
  // CHECK:    firrtl.propassign %name, %name_in : !firrtl.string
  // CHECK:    firrtl.propassign %depth, %depth_in : !firrtl.integer
  // CHECK:    firrtl.propassign %width, %width_in : !firrtl.integer
  // CHECK:    firrtl.propassign %maskBits, %maskBits_in : !firrtl.integer
  // CHECK:    firrtl.propassign %readPorts, %readPorts_in : !firrtl.integer
  // CHECK:    firrtl.propassign %writePorts, %writePorts_in : !firrtl.integer
  // CHECK:    firrtl.propassign %readwritePorts, %readwritePorts_in : !firrtl.integer
  // CHECK:    firrtl.propassign %writeLatency, %writeLatency_in : !firrtl.integer
  // CHECK:    firrtl.propassign %readLatency, %readLatency_in : !firrtl.integer
  // CHECK:    firrtl.propassign %hierarchy, %hierarchy_in : !firrtl.list<path>
  
  // CHECK: firrtl.class @MemoryMetadata
  // CHECK:   firrtl.path instance distinct[0]<>
  // CHECK:   %MWrite_ext = firrtl.object @MemorySchema
  // CHECK:   firrtl.string "MWrite_ext"
  // CHECK:   firrtl.object.subfield %MWrite_ext[name_in]
  // CHECK:   firrtl.integer 12
  // CHECK:   firrtl.object.subfield %MWrite_ext[depth_in]
  // CHECK:   firrtl.integer 42
  // CHECK:   firrtl.object.subfield %MWrite_ext[width_in]
  // CHECK:   firrtl.integer 1
  // CHECK:   firrtl.object.subfield %MWrite_ext[maskBits_in]
  // CHECK:   firrtl.integer 0
  // CHECK:   firrtl.object.subfield %MWrite_ext[readPorts_in]
  // CHECK:   firrtl.integer 1
  // CHECK:   firrtl.object.subfield %MWrite_ext[writePorts_in]
  // CHECK:   firrtl.integer 0
  // CHECK:   firrtl.object.subfield %MWrite_ext[readwritePorts_in]
  // CHECK:   firrtl.integer 1
  // CHECK:   firrtl.object.subfield %MWrite_ext[writeLatency_in]
  // CHECK:   firrtl.integer 1
  // CHECK:   firrtl.object.subfield %MWrite_ext[readLatency_in]
  // CHECK:   firrtl.object.subfield %MWrite_ext[hierarchy_in]
  // CHECK:   firrtl.propassign %MWrite_ext_field, %MWrite_ext
  // CHECK: }

  // CHECK:               emit.file "metadata{{/|\\\\}}seq_mems.json" {
  // CHECK-NEXT{LITERAL}:   sv.verbatim  "[\0A {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 12,\0A \22width\22: 42,\0A \22masked\22: false,\0A \22read\22: 0,\0A \22write\22: 1,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [\0A {\0A \22name\22: \22user_input\22,\0A \22direction\22: \22input\22,\0A \22width\22: 5\0A }\0A ],\0A \22hierarchy\22: [\0A \22{{1}}.MWrite_ext\22\0A ]\0A }\0A]"
  // CHECK-SAME:              {symbols = [@MWrite_ext, @OneMemory]}
  // CHECK-NEXT:          }

  // CHECK:               emit.file "dut.conf" {
  // CHECK-NEXT{LITERAL}:   sv.verbatim "name {{0}} depth 12 width 42 ports write\0A"
  // CHECK-SAME:              {symbols = [@MWrite_ext]}
  // CHECK-NEXT:          }
}

// -----

// CHECK-LABEL: firrtl.circuit "DualReadsSMem"
firrtl.circuit "DualReadsSMem" {
  firrtl.module @DualReadsSMem() {
    %0:12 = firrtl.instance DualReads_ext {annotations = [{class = "sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation", data = {baseAddress = 2147483648 : i64, dataBits = 8 : i64, eccBits = 0 : i64, eccIndices = [], eccScheme = "none"}}]}  @DualReads_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, in R0_data: !firrtl.uint<42>, in R1_addr: !firrtl.uint<4>, in R1_en: !firrtl.uint<1>, in R1_clk: !firrtl.clock, in R1_data: !firrtl.uint<42>, in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
  firrtl.memmodule @DualReads_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, in R0_data: !firrtl.uint<42>, in R1_addr: !firrtl.uint<4>, in R1_en: !firrtl.uint<1>, in R1_clk: !firrtl.clock, in R1_data: !firrtl.uint<42>, in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 2 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // CHECK{LITERAL}: sv.verbatim "[\0A {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 12,\0A \22width\22: 42,\0A \22masked\22: false,\0A \22read\22: 2,\0A \22write\22: 1,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{1}}.DualReads_ext\22\0A ]\0A }\0A]"
  // CHECK: {symbols = [@DualReads_ext, @DualReadsSMem]}
  // CHECK{LITERAL}: sv.verbatim "name {{0}} depth 12 width 42 ports write,read,read\0A" {symbols = [@DualReads_ext]}

}

// -----

// CHECK-LABEL: firrtl.circuit "ReadOnlyMemory"
firrtl.circuit "ReadOnlyMemory" {
  firrtl.module @ReadOnlyMemory() {
    %0:4 = firrtl.instance rom_ext sym @rom_ext_0 @rom_ext(in R0_addr: !firrtl.uint<9>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<32>)
  }
  firrtl.memmodule @rom_ext(in R0_addr: !firrtl.uint<9>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<32>) attributes {dataWidth = 32 : ui32, depth = 512 : ui64, extraPorts = [], maskBits = 0 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // CHECK{LITERAL}: sv.verbatim "[\0A  {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 512,\0A \22width\22: 32,\0A \22masked\22: false,\0A \22read\22: 1,\0A \22write\22: 0,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{1}}.rom_ext\22\0A ]\0A }\0A]"
  // CHECK: symbols = [@rom_ext, @ReadOnlyMemory]}
  // CHECK{LITERAL}: sv.verbatim "name {{0}} depth 512 width 32 ports read\0A" {symbols = [@rom_ext]}
}

// -----

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top" {
    // CHECK: hw.hierpath @[[DUTNLA:.+]] [@top::@sym]
    firrtl.module @top()  {
      // CHECK: firrtl.instance dut sym @[[DUT_SYM:.+]] {annotations = [{circt.nonlocal = @dutNLA, class = "circt.tracker", id = distinct[0]<>}]} @DUT() 
      firrtl.instance dut @DUT()
      firrtl.instance mem1 @Mem1()
      firrtl.instance mem2 @Mem2()
    }
    firrtl.module private @Mem1() {
      %0:4 = firrtl.instance head_ext  @head_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.module private @Mem2() {
      %0:4 =  firrtl.instance head_0_ext  @head_0_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.module private @DUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
      // CHECK: firrtl.instance mem1 sym @[[MEM1_SYM:.+]] @Mem(
      firrtl.instance mem1 @Mem()
    }
    firrtl.module private @Mem() {
      %0:10 = firrtl.instance memory_ext {annotations = [{class = "sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation", data = {baseAddress = 2147483648 : i64, dataBits = 8 : i64, eccBits = 0 : i64, eccIndices = [], eccScheme = "none"}}]} @memory_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<8>, in RW0_addr: !firrtl.uint<4>, in RW0_en: !firrtl.uint<1>, in RW0_clk: !firrtl.clock, in RW0_wmode: !firrtl.uint<1>, in RW0_wdata: !firrtl.uint<8>, out RW0_rdata: !firrtl.uint<8>)
      %1:8 = firrtl.instance dumm_ext @dumm_ext(in R0_addr: !firrtl.uint<5>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<5>, in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.memmodule private @head_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule private @head_0_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule private @memory_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<8>, in RW0_addr: !firrtl.uint<4>, in RW0_en: !firrtl.uint<1>, in RW0_clk: !firrtl.clock, in RW0_wmode: !firrtl.uint<1>, in RW0_wdata: !firrtl.uint<8>, out RW0_rdata: !firrtl.uint<8>) attributes {dataWidth = 8 : ui32, depth = 16 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule private @dumm_ext(in R0_addr: !firrtl.uint<5>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<5>, in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}

  // CHECK:               emit.file "metadata{{/|\\\\}}seq_mems.json" {
  // CHECK-NEXT{LITERAL}:   sv.verbatim "[\0A {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 16,\0A \22width\22: 8,\0A \22masked\22: false,\0A \22read\22: 1,\0A \22write\22: 0,\0A \22readwrite\22: 1,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{3}}.{{4}}.memory_ext\22\0A ]\0A },\0A {\0A \22module_name\22: \22{{5}}\22,\0A \22depth\22: 20,\0A \22width\22: 5,\0A \22masked\22: false,\0A \22read\22: 1,\0A \22write\22: 1,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{3}}.{{4}}.dumm_ext\22\0A ]\0A }\0A]"
  // CHECK-SAME:              {symbols = [@memory_ext, @top, #hw.innerNameRef<@top::@[[DUT_SYM]]>, @DUT, #hw.innerNameRef<@DUT::@[[MEM1_SYM]]>, @dumm_ext]}
  // CHECK-NEXT:          }

  // CHECK:               emit.file "dut.conf" {
  // CHECK-NEXT{LITERAL}:   sv.verbatim "name {{0}} depth 20 width 5 ports write\0Aname {{1}} depth 20 width 5 ports write\0Aname {{2}} depth 16 width 8 ports read,rw\0Aname {{3}} depth 20 width 5 ports write,read\0A"
  // CHECK-SAME:              {symbols = [@head_ext, @head_0_ext, @memory_ext, @dumm_ext]}
  // CHECK-NEXT:          }

  // CHECK:  firrtl.class @SiFive_Metadata
  // CHECK:    %[[V0:.+]] = firrtl.path instance distinct[0]<>
  // CHECK-NEXT:    %[[V1:.+]] = firrtl.list.create %[[V0]] : !firrtl.list<path>
  // CHECK-NEXT:    firrtl.propassign %dutModulePath_field_1, %[[V1]] : !firrtl.list<path>
}
