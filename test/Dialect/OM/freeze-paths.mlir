// RUN: circt-opt -om-freeze-paths %s | FileCheck %s
hw.hierpath private @nla [@PathModule::@sym]
hw.hierpath private @nla_0 [@PathModule::@sym_0]
hw.hierpath private @nla_1 [@PathModule::@sym_1]
hw.hierpath private @nla_2 [@PathModule::@sym_2]
hw.hierpath private @nla_3 [@PathModule::@child]
hw.hierpath private @nla_4 [@Child]
hw.hierpath private @nla_5 [@PathModule::@child, @Child::@sym]
hw.hierpath private @nla_6 [@PublicLeaf]
hw.module @PathModule(in %in: i1 {hw.exportPort = #hw<innerSym@sym>}) {
  %wire = hw.wire %wire sym @sym_0 {hw.verilogName = "wire"} : i8
  %array = hw.wire %array sym [<@sym_1,1,public>] {hw.verilogName = "array"}: !hw.array<1xi8>
  %struct = hw.wire %struct sym [<@sym_2,1,public>] {hw.verilogName = "struct"}: !hw.struct<a: i8>
  hw.instance "child" sym @child @Child() -> () {hw.verilogName = "child"}
  hw.instance "public_middle" @PublicMiddle() -> () {hw.verilogName = "public_middle"}
  hw.output
}
hw.hierpath private @NonLocal [@PathModule::@child, @Child]
hw.module private @Child() {
  %z_i8 = sv.constantZ : i8
  %non_local = hw.wire %z_i8 sym @sym {hw.verilogName = "non_local"}  : i8
  hw.output
}

hw.module @PublicMiddle() {
  hw.instance "leaf" @PublicLeaf() -> () {hw.verilogName = "leaf"}
}

hw.module private @PublicLeaf() {}

// CHECK-LABEL om.class @PathTest()
om.class @PathTest(%basepath : !om.basepath, %path : !om.path) {
  // CHECK: om.frozenpath_create reference %basepath "PathModule>in"
  %0 = om.path_create reference %basepath @nla

  // CHECK: om.frozenpath_create reference %basepath "PathModule>wire"
  %1 = om.path_create reference %basepath @nla_0

  // CHECK: om.frozenpath_create member_reference %basepath "PathModule>wire"
  %2 = om.path_create member_reference %basepath @nla_0

  // CHECK: om.frozenpath_create dont_touch %basepath "PathModule>wire"
  %4 = om.path_create dont_touch %basepath @nla_0

  // CHECK: om.frozenpath_create reference %basepath "PathModule>array[0]" 
  %5 = om.path_create reference %basepath @nla_1

  // CHECK: om.frozenpath_create reference %basepath "PathModule>struct.a"
  %6 = om.path_create reference %basepath @nla_2

  // CHECK: om.frozenpath_create reference %basepath "PathModule/child:Child>non_local"
  %7 = om.path_create reference %basepath @nla_5

  // CHECK: om.frozenpath_create instance %basepath "PathModule/child:Child"
  %8 = om.path_create instance %basepath @nla_3

  // CHECK: om.frozenpath_create member_instance %basepath "PathModule/child:Child"
  %9 = om.path_create member_instance %basepath @nla_3

  // CHECK: om.frozenpath_create reference %basepath "PathModule/child:Child"
  %10 = om.path_create reference %basepath @nla_4

  // CHECK: om.frozenpath_create reference %basepath "PublicMiddle/leaf:PublicLeaf"
  %11 = om.path_create reference %basepath @nla_6

  // CHECK: om.frozenpath_empty
  %12 = om.path_empty

  // CHECK: om.frozenbasepath_create %basepath "PathModule/child"
  %13 = om.basepath_create %basepath @nla_3
}
