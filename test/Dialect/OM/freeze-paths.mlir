// RUN: circt-opt -om-freeze-paths %s | FileCheck %s
hw.hierpath private @nla [@PathModule::@sym]
hw.hierpath private @nla_0 [@PathModule::@sym_0]
hw.hierpath private @nla_1 [@PathModule::@sym_1]
hw.hierpath private @nla_2 [@PathModule::@sym_2]
hw.hierpath private @nla_3 [@PathModule::@child]
hw.hierpath private @nla_4 [@PathModule::@child, @Child]
hw.hierpath private @nla_5 [@PathModule::@child, @Child::@sym]
hw.hierpath private @nla_6 [@PublicMiddle::@leaf, @PublicLeaf]
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
  hw.instance "leaf" sym @leaf @PublicLeaf() -> () {hw.verilogName = "leaf"}
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

  om.class.fields
}

// CHECK-LABEL: om.class @ListCreateTest
// CHECK-SAME: -> (notpath: !om.list<i1>, basepath: !om.list<!om.frozenbasepath>, path: !om.list<!om.frozenpath>, nestedpath: !om.list<!om.list<!om.frozenpath>>, concatpath: !om.list<!om.list<!om.frozenpath>>)
om.class @ListCreateTest(%notpath: i1, %basepath : !om.basepath, %path : !om.path) -> (notpath: !om.list<i1>, basepath: !om.list<!om.basepath>, path: !om.list<!om.path>, nestedpath: !om.list<!om.list<!om.path>>, concatpath: !om.list<!om.list<!om.path>>) {
  // CHECK: [[NOT_PATH_LIST:%.+]] = om.list_create %notpath : i1
  %0 = om.list_create %notpath : i1

  // CHECK: [[BASE_PATH_LIST:%.+]] = om.list_create %basepath : !om.frozenbasepath
  %1 = om.list_create %basepath : !om.basepath

  // CHECK: [[PATH_LIST:%.+]] = om.list_create %path : !om.frozenpath
  %2 = om.list_create %path : !om.path

  // CHECK: [[NESTED_PATH_LIST:%.+]] = om.list_create [[PATH_LIST]] : !om.list<!om.frozenpath>
  %3 = om.list_create %2 : !om.list<!om.path>

  // CHECK: [[CONCAT_PATH_LIST:%.+]] = om.list_concat [[NESTED_PATH_LIST]] : <!om.list<!om.frozenpath>>
  %4 = om.list_concat %3 : !om.list<!om.list<!om.path>>

  // CHECK: om.class.fields [[NOT_PATH_LIST]], [[BASE_PATH_LIST]], [[PATH_LIST]], [[NESTED_PATH_LIST]], [[CONCAT_PATH_LIST]] : !om.list<i1>, !om.list<!om.frozenbasepath>, !om.list<!om.frozenpath>, !om.list<!om.list<!om.frozenpath>>
  om.class.fields %0, %1, %2, %3, %4 : !om.list<i1>, !om.list<!om.basepath>, !om.list<!om.path>, !om.list<!om.list<!om.path>>, !om.list<!om.list<!om.path>>
}

// CHECK-LABEL om.class @PathListClass(%pathList: !om.list<!om.frozenpath>) -> (pathList: !om.list<!om.path>
om.class @PathListClass(%pathList : !om.list<!om.path>) -> (pathList: !om.list<!om.path>) {
  om.class.fields %pathList : !om.list<!om.path>
}

// CHECK-LABEL om.class @PathListTest(%arg: !om.list<!om.frozenpath>)
om.class @PathListTest(%arg : !om.list<!om.path>) {
  // CHECK: om.object @PathListClass(%arg) : (!om.list<!om.frozenpath>)
  om.object @PathListClass(%arg) : (!om.list<!om.path>) -> !om.class.type<@PathListClass>

  // CHECK: [[RES:%.+]] = om.list_create
  %0 = om.list_create : !om.path
  // CHECK: om.object @PathListClass([[RES]]) : (!om.list<!om.frozenpath>)
  om.object @PathListClass(%0) : (!om.list<!om.path>) -> !om.class.type<@PathListClass>

  om.class.fields
}

// CHECK-LABEL: om.class @ObjectFieldTest
om.class @ObjectFieldTest(%basepath : !om.basepath, %path : !om.path) -> (subfield: !om.list<!om.list<!om.path>>) {
  // CHECK: [[OBJ:%.+]] = om.object @PathTest
  %0 = om.object @PathTest(%basepath, %path) : (!om.basepath, !om.path) -> !om.class.type<@PathTest>

  // CHECK: [[SUBFIELD:%.+]] = om.object.field [[OBJ]], [@nestedpath] : (!om.class.type<@PathTest>) -> !om.list<!om.list<!om.frozenpath>>
  %1 = om.object.field %0, [@nestedpath] : (!om.class.type<@PathTest>) -> !om.list<!om.list<!om.path>>

  // CHECK: om.class.fields [[SUBFIELD]] : !om.list<!om.list<!om.frozenpath>>
  om.class.fields %1 : !om.list<!om.list<!om.path>>
}
