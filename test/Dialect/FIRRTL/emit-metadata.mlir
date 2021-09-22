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
    filename = "./tmp/retime_modules.json"
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


