// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-emit-metadata)' %s | FileCheck %s

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
    firrtl.instance @DUTBlackbox_0 { name = "test" }
    firrtl.instance @DUTBlackbox_1 { name = "test" }
    firrtl.instance @DUTBlackbox_2 { name = "test" }
  }

  // These should all be ignored.
  firrtl.extmodule @ignored0() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno"}], defname = "ignored0"}
  firrtl.extmodule @ignored1() attributes {annotations = [{class = "firrtl.transforms.BlackBoxResourceAnno"}], defname = "ignored1"}
  firrtl.extmodule @ignored2() attributes {annotations = [{class = "sifive.enterprise.firrtl.ScalaClassAnnotation", className = "freechips.rocketchip.util.BlackBoxedROM"}], defname = "ignored2"}
  firrtl.extmodule @ignored3() attributes {annotations = [{class = "sifive.enterprise.grandcentral.DataTapsAnnotation"}], defname = "ignored3"}
  firrtl.extmodule @ignored4() attributes {annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation", id = 4 : i64}], defname = "ignored4"}
  firrtl.extmodule @ignored5() attributes {annotations = [{class = "sifive.enterprise.grandcentral.transforms.SignalMappingAnnotation"}], defname = "ignored5"}

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
