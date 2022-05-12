// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-emit-metadata{repl-seq-mem=true repl-seq-mem-file="metadata/dut.conf"})' %s | FileCheck %s

firrtl.circuit "empty" {
  firrtl.module @empty() {
  }
}
// CHECK-LABEL: firrtl.circuit "empty"   {
// CHECK-NEXT:    firrtl.module @empty() {
// CHECK-NEXT:    }
// CHECK-NEXT:  }

//===----------------------------------------------------------------------===//
// RetimeModules
//===----------------------------------------------------------------------===//

firrtl.circuit "retime0" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = "tmp/retime_modules.json"
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
// CHECK:         firrtl.module @retime0() {
// CHECK:         firrtl.module @retime1() {
// CHECK:         firrtl.module @retime2() {
// CHECK{LITERAL}:  sv.verbatim "[\22{{0}}\22,\22{{1}}\22]"
// CHECK-SAME:        output_file = #hw.output_file<"tmp/retime_modules.json", excludeFromFileList>
// CHECK-SAME:        symbols = [@retime0, @retime2]

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
// CHECK-NOT: sv.verbatim "[]" {output_file = #hw.output_file<"", excludeFromFileList>, symbols = []}
// CHECK:     sv.verbatim "[]" {output_file = #hw.output_file<"dut_blackboxes.json", excludeFromFileList>, symbols = []}
// CHECK-NOT: sv.verbatim "[]" {output_file = #hw.output_file<"", excludeFromFileList>, symbols = []}
}

// CHECK-LABEL: firrtl.circuit "TestBlackboxes"  {
firrtl.circuit "TestBlackboxes" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
    filename = "test_blackboxes.json"
  }]} {
  firrtl.module @TestBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  }
// CHECK-NOT: sv.verbatim "[]" {output_file = #hw.output_file<"", excludeFromFileList>, symbols = []}
// CHECK:     sv.verbatim "[]" {output_file = #hw.output_file<"test_blackboxes.json", excludeFromFileList>, symbols = []}
// CHECK-NOT: sv.verbatim "[]" {output_file = #hw.output_file<"", excludeFromFileList>, symbols = []}
}

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
  firrtl.extmodule @ignored2() attributes {annotations = [{class = "sifive.enterprise.firrtl.ScalaClassAnnotation", className = "freechips.rocketchip.util.BlackBoxedROM"}], defname = "ignored2"}
  firrtl.extmodule @ignored3() attributes {annotations = [{class = "sifive.enterprise.grandcentral.DataTapsAnnotation"}], defname = "ignored3"}
  firrtl.extmodule @ignored4() attributes {annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation", id = 4 : i64}], defname = "ignored4"}
  firrtl.extmodule @ignored5() attributes {annotations = [{class = "sifive.enterprise.grandcentral.transforms.SignalMappingAnnotation"}], defname = "ignored5"}
  firrtl.extmodule @ignored6() attributes {annotations = [{class = "firrtl.transforms.BlackBox"}], defname = "ignored6"}

  // Gracefully handle missing defnames.
  firrtl.extmodule @NoDefName()

  firrtl.extmodule @TestBlackbox() attributes {defname = "TestBlackbox"}
  // CHECK: sv.verbatim "[\22TestBlackbox\22]" {output_file = #hw.output_file<"test_blackboxes.json", excludeFromFileList>, symbols = []}

  // Should be de-duplicated and sorted.
  firrtl.extmodule @DUTBlackbox_0() attributes {defname = "DUTBlackbox2"}
  firrtl.extmodule @DUTBlackbox_1() attributes {defname = "DUTBlackbox1"}
  firrtl.extmodule @DUTBlackbox_2() attributes {defname = "DUTBlackbox1"}
  // CHECK: sv.verbatim "[\22DUTBlackbox1\22,\22DUTBlackbox2\22]" {output_file = #hw.output_file<"dut_blackboxes.json", excludeFromFileList>, symbols = []}
}

//===----------------------------------------------------------------------===//
// MemoryMetadata
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top"
{
  firrtl.module @top() { }
  // When there are no memories, we still need to emit the memory metadata.
  // CHECK: sv.verbatim "[]" {output_file = #hw.output_file<"metadata/tb_seq_mems.json", excludeFromFileList>, symbols = []}
  // CHECK: sv.verbatim "[]" {output_file = #hw.output_file<"metadata/seq_mems.json", excludeFromFileList>, symbols = []}
  // CHECK: sv.verbatim "" {output_file = #hw.output_file<"\22metadata/dut.conf\22", excludeFromFileList>, symbols = []}
}

// CHECK-LABEL: firrtl.circuit "OneMemory"
firrtl.circuit "OneMemory" {
  firrtl.module @OneMemory() {
    %0:5= firrtl.instance MWrite_ext sym @MWrite_ext_0  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>, in user_input: !firrtl.uint<5>)
  }
  firrtl.memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>, in user_input: !firrtl.uint<5>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [{direction = "input", name = "user_input", width = 5 : ui32}], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    // CHECK: sv.verbatim "[{\22module_name\22:\22{{[{][{]0[}][}]}}
    // CHECK-SAME: \22,\22depth\22:12,\22width\22:42,\22masked\22:false,\22read\22:false,\22write\22:true,\22readwrite\22:false,
    // CHECK-SAME: \22extra_ports\22:[{\22name\22:\22user_input\22,\22direction\22:\22input\22,\22width\22:5}], 
    // CHECK-SAME: \22hierarchy\22:[\22{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}\22]}]"
    // CHECK-SAME: {output_file = #hw.output_file<"metadata/tb_seq_mems.json", excludeFromFileList>, symbols = [@MWrite_ext, @OneMemory, #hw.innerNameRef<@OneMemory::@MWrite_ext_0>]}
    // CHECK: sv.verbatim "[]" {output_file = #hw.output_file<"metadata/seq_mems.json", excludeFromFileList>, symbols = []}
    // CHECK: sv.verbatim "name {{[{][{]0[}][}]}} depth 12 width 42 ports write\0A" {output_file = #hw.output_file<"\22metadata/dut.conf\22", excludeFromFileList>, symbols = [@MWrite_ext]}

}

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top" {
    firrtl.module @top()  {
      firrtl.instance dut @DUT()
      firrtl.instance mem1 @Mem1()
      firrtl.instance mem2 @Mem2()
    }
    firrtl.module @Mem1() {
      %0:4 = firrtl.instance head_ext  @head_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.module @Mem2() {
      %0:4 =  firrtl.instance head_0_ext  @head_0_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.module @DUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
      firrtl.instance mem1 @Mem()
    }
    firrtl.module @Mem() {
      %0:10 = firrtl.instance memory_ext {annotations = [{class = "sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation", data = {baseAddress = 2147483648 : i64, dataBits = 8 : i64, eccBits = 0 : i64, eccIndices = [], eccScheme = "none"}}]} @memory_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<8>, in RW0_addr: !firrtl.uint<4>, in RW0_en: !firrtl.uint<1>, in RW0_clk: !firrtl.clock, in RW0_wmode: !firrtl.uint<1>, in RW0_wdata: !firrtl.uint<8>, out RW0_rdata: !firrtl.uint<8>)
      %1:8 = firrtl.instance dumm_ext @dumm_ext(in R0_addr: !firrtl.uint<5>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<5>, in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.memmodule @head_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule @head_0_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule @memory_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<8>, in RW0_addr: !firrtl.uint<4>, in RW0_en: !firrtl.uint<1>, in RW0_clk: !firrtl.clock, in RW0_wmode: !firrtl.uint<1>, in RW0_wdata: !firrtl.uint<8>, out RW0_rdata: !firrtl.uint<8>) attributes {dataWidth = 8 : ui32, depth = 16 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule @dumm_ext(in R0_addr: !firrtl.uint<5>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<5>, in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    // CHECK: sv.verbatim "[{\22module_name\22:\22{{[{][{]0[}][}]}}\22,\22depth\22:20,\22width\22:5,\22masked\22:false,\22read\22:false,\22write\22:true,\22readwrite\22:false,\22extra_ports\22:[]
    // CHECK-SAME: \22hierarchy\22:[\22{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}\22]},
    // CHECK-SAME: {\22module_name\22:\22{{[{][{]4[}][}]}}\22,\22depth\22:20,\22width\22:5,\22masked\22:false,\22read\22:false,\22write\22:true,\22readwrite\22:false,\22extra_ports\22:[]
    // CHECK-SAME: \22hierarchy\22:[\22{{[{][{]5[}][}]}}.{{[{][{]6[}][}]}}.{{[{][{]7[}][}]}}\22]}]" 
    // CHECK-SAME: {output_file = #hw.output_file<"metadata/tb_seq_mems.json", excludeFromFileList>
    // CHECK-SAME: symbols = [@head_ext, @top, #hw.innerNameRef<@top::@metadata_sym_0>, #hw.innerNameRef<@Mem1::@metadata_sym>, @head_0_ext, @top, #hw.innerNameRef<@top::@metadata_sym_1>, #hw.innerNameRef<@Mem2::@metadata_sym>]}
    // CHECK: sv.verbatim "[{\22module_name\22:\22{{[{][{]0[}][}]}}\22,\22depth\22:16,\22width\22:8,\22masked\22:false,\22read\22:true,\22write\22:false,\22readwrite\22:true,\22extra_ports\22:[]
    // CHECK-SAME: \22hierarchy\22:[\22{{[{][{]3[}][}]}}.{{[{][{]4[}][}]}}.{{[{][{]5[}][}]}}\22]},
    // CHECK-SAME: {\22module_name\22:\22{{[{][{]6[}][}]}}\22,\22depth\22:20,\22width\22:5,\22masked\22:false,\22read\22:true,\22write\22:true,\22readwrite\22:false,\22extra_ports\22:[],
    // CHECK-SAME: \22hierarchy\22:[\22{{[{][{]9[}][}]}}.{{[{][{]10[}][}]}}.{{[{][{]11[}][}]}}\22]}]"
    // CHECK-SAME: output_file = #hw.output_file<"metadata/seq_mems.json", excludeFromFileList>, symbols = [@memory_ext, @top, #hw.innerNameRef<@top::@metadata_sym>, @DUT, #hw.innerNameRef<@DUT::@metadata_sym>, #hw.innerNameRef<@Mem::@metadata_sym>, @dumm_ext, @top, #hw.innerNameRef<@top::@metadata_sym>, @DUT, #hw.innerNameRef<@DUT::@metadata_sym>, #hw.innerNameRef<@Mem::@metadata_sym_0>]}
    // CHECK: sv.verbatim "name {{[{][{]0[}][}]}} depth 16 width 8 ports rw\0Aname {{[{][{]1[}][}]}} depth 20 width 5 ports write,read
    // CHECK-SAME: \0Aname {{[{][{]2[}][}]}} depth 20 width 5 ports write\0Aname {{[{][{]3[}][}]}} depth 20 width 5 ports write\0A"
    // CHECK-SAME: {output_file = #hw.output_file<"\22metadata/dut.conf\22", excludeFromFileList>, symbols = [@memory_ext, @dumm_ext, @head_ext, @head_0_ext]}
}
