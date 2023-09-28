// RUN: circt-opt -om-freeze-paths %s | FileCheck %s
hw.hierpath private @nla [@PathModule::@sym]
hw.hierpath private @nla_0 [@PathModule::@sym_0]
hw.hierpath private @nla_1 [@PathModule::@sym_1]
hw.hierpath private @nla_2 [@PathModule::@sym_2]
hw.hierpath private @nla_3 [@PathModule::@child]
hw.hierpath private @nla_4 [@Child]
hw.hierpath private @nla_5 [@PathModule::@child, @Child::@sym]
hw.module @PathModule(in %in: i1 {hw.exportPort = #hw<innerSym@sym>}) {
  %wire = hw.wire %wire sym @sym_0 {hw.verilogName = "wire"} : i8
  %array = hw.wire %array sym [<@sym_1,1,public>] {hw.verilogName = "array"}: !hw.array<1xi8>
  %struct = hw.wire %struct sym [<@sym_2,1,public>] {hw.verilogName = "struct"}: !hw.struct<a: i8>
  hw.instance "child" sym @child @Child() -> () {hw.verilogName = "child"}
  hw.output
}
hw.hierpath private @NonLocal [@PathModule::@child, @Child]
hw.module private @Child() {
  %z_i8 = sv.constantZ : i8
  %non_local = hw.wire %z_i8 sym @sym {hw.verilogName = "non_local"}  : i8
  hw.output
}
// CHECK-LABEL om.class @PathTest()
om.class @PathTest() {
  // CHECK: om.constant #om.path<"OMReferenceTarget:PathModule>in"> : !om.path
  %0 = om.path reference @nla

  // CHECK: om.constant #om.path<"OMReferenceTarget:PathModule>wire"> : !om.path
  %1 = om.path reference @nla_0

  // CHECK: om.constant #om.path<"OMMemberReferenceTarget:PathModule>wire"> : !om.path
  %2 = om.path member_reference @nla_0

  // CHECK: om.constant #om.path<"OMDontTouchedReferenceTarget:PathModule>wire"> : !om.path
  %4 = om.path dont_touch @nla_0

  // CHECK: om.constant #om.path<"OMReferenceTarget:PathModule>array[0]"> : !om.path
  %5 = om.path reference @nla_1

  // CHECK: om.constant #om.path<"OMReferenceTarget:PathModule>struct.a"> : !om.path
  %6 = om.path reference @nla_2

  // CHECK: om.constant #om.path<"OMReferenceTarget:PathModule/child:Child>non_local"> : !om.path
  %7 = om.path reference @nla_5

  // CHECK: om.constant #om.path<"OMInstanceTarget:PathModule/child:Child"> : !om.path
  %8 = om.path instance @nla_3

  // CHECK: om.constant #om.path<"OMMemberInstanceTarget:PathModule/child:Child"> : !om.path
  %9 = om.path member_instance @nla_3

  // CHECK: om.constant #om.path<"OMReferenceTarget:PathModule/child:Child"> : !om.path
  %10 = om.path reference @nla_4
}
