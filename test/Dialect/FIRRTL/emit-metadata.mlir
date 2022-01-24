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
      class = "sifive.enterprise.firrtl.RetimeModuleAnnotation"
  }]} { }

  firrtl.module @retime1() { }

  firrtl.module @retime2() attributes { annotations = [{
      class = "sifive.enterprise.firrtl.RetimeModuleAnnotation"
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
  firrtl.extmodule @ignored1() attributes {annotations = [{class = "firrtl.transforms.BlackBoxResourceAnno"}], defname = "ignored1"}
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

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top" 
{
    firrtl.module @top()  {
      firrtl.instance dut @dut()
      firrtl.instance mem1 @Mem1()
    }
    firrtl.module @Mem1() {
      %head_MPORT_2 = firrtl.mem Undefined  {depth = 20 : i64, name = "head", portNames = ["MPORT_2", "MPORT_6"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
    }
    firrtl.module @dut() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
      firrtl.instance mem1 @Mem()
    }
    firrtl.module @Mem() {
      %memory_rw, %memory_rw_r = firrtl.mem Undefined  {annotations = [{class = "sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation", data = {baseAddress = 2147483648 : i64, dataBits = 8 : i64, eccBits = 0 : i64, eccIndices = [], eccScheme = "none"}}], depth = 16 : i64, name = "memory", portNames = ["rw", "rw_r", "rw_w"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      %head_MPORT_2, %head_MPORT_6 = firrtl.mem Undefined  {depth = 20 : i64, name = "dumm", portNames = ["MPORT_2", "MPORT_6"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>, !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
    }
    // CHECK: sv.verbatim "[{\22module_name\22:\22head_ext\22,\22depth\22:20,\22width\22:5,\22masked\22:\22true\22,\22read\22:\22false\22,\22write\22:\22true\22,\22readwrite\22:\22false\22,\22mask_granularity\22:5,\22extra_ports\22:[],\22hierarchy\22:[\22top.mem1.head\22]}]" {output_file = #hw.output_file<"metadata/tb_seq_mems.json", excludeFromFileList>, symbols = []}
    // CHECK: sv.verbatim "[{\22module_name\22:\22memory_ext\22,\22depth\22:16,\22width\22:8,\22masked\22:\22true\22,\22read\22:\22true\22,\22write\22:\22false\22,\22readwrite\22:\22true\22,\22mask_granularity\22:8,\22extra_ports\22:[],\22hierarchy\22:[\22top.dut.mem1.memory\22]
    // CHECK-SAME: output_file = #hw.output_file<"metadata/seq_mems.json", excludeFromFileList>, symbols = []
    // CHECK: sv.verbatim "name memory_ext depth 16 width 8 ports mrw mask_gran 8\0Aname head_ext depth 20 width 5 ports mwrite mask_gran 5\0A"
    // CHECK-SAME: {output_file = #hw.output_file<"\22metadata/dut.conf\22", excludeFromFileList>, symbols = []}
}
