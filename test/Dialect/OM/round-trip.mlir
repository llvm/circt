// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: om.class @Thingy
// CHECK-SAME: (%blue_1: i8, %blue_2: i32)
om.class @Thingy(%blue_1: i8, %blue_2: i32) {
  // CHECK: %[[c5:.+]] = om.constant 5 : i8
  %0 = om.constant 5 : i8
  // CHECK: %[[c6:.+]] = om.constant 6 : i32
  %1 = om.constant 6 : i32
  // CHECK: %[[widget:.+]] = om.object @Widget(%[[c5]], %[[c6]]) : (i8, i32) -> !om.class.type<@Widget>
  %2 = om.object @Widget(%0, %1) : (i8, i32) -> !om.class.type<@Widget>
  // CHECK: om.class.field @widget, %[[widget]] : !om.class.type<@Widget>
  om.class.field @widget, %2 : !om.class.type<@Widget>

  // CHECK: %[[c7:.+]] = om.constant 7 : i8
  %3 = om.constant 7 : i8
  // CHECK: %[[c8:.+]] = om.constant 8 : i32
  %4 = om.constant 8 : i32
  // CHECK: %[[gadget:.+]] = om.object @Gadget(%[[c7]], %[[c8]]) : (i8, i32) -> !om.class.type<@Gadget>
  %5 = om.object @Gadget(%3, %4) : (i8, i32) -> !om.class.type<@Gadget>
  // CHECK: om.class.field @gadget, %[[gadget]] : !om.class.type<@Gadget>
  om.class.field @gadget, %5 : !om.class.type<@Gadget>

  // CHECK: om.class.field @blue_1, %blue_1 : i8
  om.class.field @blue_1, %blue_1 : i8

  // CHECK: %[[widget_field:.+]] = om.object.field %[[widget]], [@blue_1] : (!om.class.type<@Widget>) -> i8
  %6 = om.object.field %2, [@blue_1] : (!om.class.type<@Widget>) -> i8
  // CHECK: om.class.field @blue_2, %[[widget_field]] : i8
  om.class.field @blue_2, %6 : i8
}

// CHECK-LABEL: om.class @Widget
// CHECK-SAME: (%blue_1: i8, %green_1: i32)
om.class @Widget(%blue_1: i8, %green_1: i32) {
  // CHECK: om.class.field @blue_1, %blue_1 : i8
  om.class.field @blue_1, %blue_1 : i8
  // CHECK: om.class.field @green_1, %green_1 : i32
  om.class.field @green_1, %green_1 : i32
}

// CHECK-LABEL: om.class @Gadget
// CHECK-SAME: (%green_1: i8, %green_2: i32)
om.class @Gadget(%green_1: i8, %green_2: i32) {
  // CHECK: om.class.field @green_1, %green_1 : i8
  om.class.field @green_1, %green_1 : i8
  // CHECK: om.class.field @green_2, %green_2 : i32
  om.class.field @green_2, %green_2 : i32
}

// CHECK-LABEL: om.class @Empty
om.class @Empty() {}

// CHECK-LABEL: om.class @DiscardableAttrs
// CHECK-SAME: attributes {foo.bar = "baz"}
om.class @DiscardableAttrs() attributes {foo.bar="baz"} {}

// CHECK-LABEL: om.class.extern @Extern
// CHECK-SAME: (%param1: i1, %param2: i2)
om.class.extern @Extern(%param1: i1, %param2: i2) {
  // CHECK: om.class.extern.field @field1 : i3
  om.class.extern.field @field1 : i3

  // CHECK: om.class.extern.field @field2 : i4
  om.class.extern.field @field2 : i4
}

// CHECK-LABEL: om.class @ExternObject
// CHECK-SAME: (%[[P0:.+]]: i1, %[[P1:.+]]: i2)
om.class @ExternObject(%param1: i1, %param2: i2) {
  // CHECK: %[[O0:.+]] = om.object @Extern(%[[P0]], %[[P1]])
  %0 = om.object @Extern(%param1, %param2) : (i1, i2) -> !om.class.type<@Extern>

  // CHECK: om.object.field %[[O0]], [@field1]
  %1 = om.object.field %0, [@field1] : (!om.class.type<@Extern>) -> i3
}

om.class @NestedField1() {
  %0 = om.constant 1 : i1
  om.class.field @baz, %0 : i1
}

om.class @NestedField2() {
  %0 = om.object @NestedField1() : () -> !om.class.type<@NestedField1>
  om.class.field @bar, %0 : !om.class.type<@NestedField1>
}

om.class @NestedField3() {
  %0 = om.object @NestedField2() : () -> !om.class.type<@NestedField2>
  om.class.field @foo, %0 : !om.class.type<@NestedField2>
}

// CHECK-LABEL: @NestedField4
om.class @NestedField4() {
  // CHECK: %[[nested:.+]] = om.object @NestedField3
  %0 = om.object @NestedField3() : () -> !om.class.type<@NestedField3>
  // CHECK: %{{.+}} = om.object.field %[[nested]], [@foo, @bar, @baz] : (!om.class.type<@NestedField3>) -> i1
  %1 = om.object.field %0, [@foo, @bar, @baz] : (!om.class.type<@NestedField3>) -> i1
}

// CHECK-LABEL: @ReferenceParameter
// CHECK-SAME: !om.ref
// CHECK-SAME: !om.sym_ref
om.class @ReferenceParameter(%arg0: !om.ref, %arg1: !om.sym_ref) {
  // CHECK: om.class.field @myref
  om.class.field @myref, %arg0 : !om.ref
  // CHECK: om.class.field @sym
  om.class.field @sym, %arg1 : !om.sym_ref
}

// CHECK-LABEL: @ReferenceConstant
om.class @ReferenceConstant() {
  // CHECK: %[[const1:.+]] = om.constant #om.ref<<@A::@inst_1>> : !om.ref
  %0 = om.constant #om.ref<#hw.innerNameRef<@A::@inst_1>> : !om.ref
  // CHECK: om.class.field @myref, %[[const1]] : !om.ref
  om.class.field @myref, %0 : !om.ref

  // CHECK: %[[const2:.+]] = om.constant #om.sym_ref<@A> : !om.sym_ref
  %1 = om.constant #om.sym_ref<@A> : !om.sym_ref
  // CHECK: om.class.field @sym, %[[const2]] : !om.sym_ref
  om.class.field @sym, %1 : !om.sym_ref
}

// CHECK-LABEL: @ListConstant
om.class @ListConstant() {
  // CHECK: %[[const1:.+]] = om.constant #om.list<i64, [42]> : !om.list<i64>
  %0 = om.constant #om.list<i64, [42]> : !om.list<i64>
  // CHECK: om.class.field @list_i64, %[[const1]] : !om.list<i64>
  om.class.field @list_i64, %0 : !om.list<i64>

  // CHECK: %[[const2:.+]] = om.constant #om.list<i32, []> : !om.list<i32>
  %1 = om.constant #om.list<i32, []> : !om.list<i32>
  // CHECK: om.class.field @list_i32, %[[const2]] : !om.list<i32>
  om.class.field @list_i32, %1 : !om.list<i32>
}

// CHECK-LABEL: @ListCreate
om.class @ListCreate() {
  // CHECK: [[cst5_i8:%.+]] = om.constant 5 : i8
  %cst5_i8 = om.constant 5 : i8
  // CHECK: [[cst6_i8:%.+]] = om.constant 6 : i8
  %cst6_i8 = om.constant 6 : i8
  // CHECK: [[cst5_i32:%.+]] = om.constant 5 : i32
  %cst5_i32 = om.constant 5 : i32
  // CHECK: [[cst6_i32:%.+]] = om.constant 6 : i32
  %cst6_i32 = om.constant 6 : i32

  // CHECK: [[obj0:%.+]] = om.object @Widget([[cst5_i8]], [[cst6_i32]]) : (i8, i32) -> !om.class.type<@Widget>
  %obj0 = om.object @Widget(%cst5_i8, %cst6_i32) : (i8, i32) -> !om.class.type<@Widget>

  // CHECK: [[obj1:%.+]] = om.object @Widget([[cst6_i8]], [[cst5_i32]]) : (i8, i32) -> !om.class.type<@Widget>
  %obj1 = om.object @Widget(%cst6_i8, %cst5_i32) : (i8, i32) -> !om.class.type<@Widget>

  // CHECK: [[list:%.+]] = om.list_create [[obj0]], [[obj1]] : !om.class.type<@Widget>
  %list = om.list_create %obj0, %obj1 : !om.class.type<@Widget>

  // CHECK: om.class.field @list_field, [[list]] : !om.list<!om.class.type<@Widget>>
  om.class.field @list_field, %list : !om.list<!om.class.type<@Widget>>
}

// CHECK-LABEL: @Integer
om.class @IntegerConstant() {
  // CHECK: %[[const1:.+]] = om.constant #om.integer<36755551979133953793 : i67> : !om.integer
  %0 = om.constant #om.integer<36755551979133953793 : i67> : !om.integer
  // CHECK: om.class.field @int, %[[const1]] : !om.integer
  om.class.field @int, %0 : !om.integer
}

// CHECK-LABEL: @String
om.class @StringConstant() {
  // CHECK: %[[const1:.+]] = om.constant "foo" : !om.string
  %0 = om.constant "foo" : !om.string
  // CHECK: om.class.field @string, %[[const1]] : !om.string
  om.class.field @string, %0 : !om.string
}

// CHECK-LABEL: @LinkedList
om.class @LinkedList(%prev: !om.class.type<@LinkedList>) {
  om.class.field @prev, %prev : !om.class.type<@LinkedList>
}

// CHECK-LABEL: @ReferenceEachOther
om.class @ReferenceEachOther() {
  // CHECK-NEXT: %[[obj1:.+]] = om.object @LinkedList(%[[obj2:.+]]) : (!om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
  // CHECK-NEXT: %[[obj2]] = om.object @LinkedList(%[[obj1]]) : (!om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
  %0 = om.object @LinkedList(%1) : (!om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
  %1 = om.object @LinkedList(%0) : (!om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
}

// CHECK-LABEL: @RefecenceEachOthersField
om.class @RefecenceEachOthersField(%blue_1: i8, %green_1: i32) {
  // CHECK-NEXT: %[[obj1:.+]] = om.object @Widget(%blue_1, %[[field2:.+]]) : (i8, i32) -> !om.class.type<@Widget>
  %0 = om.object @Widget(%blue_1, %3) : (i8, i32) -> !om.class.type<@Widget>
  // CHECK-NEXT: %[[field1:.+]] = om.object.field %[[obj1]], [@blue_1] : (!om.class.type<@Widget>) -> i8
  %1 = om.object.field %0, [@blue_1] : (!om.class.type<@Widget>) -> i8

  // CHECK-NEXT: %[[obj2:.+]] = om.object @Widget(%[[field1]], %green_1) : (i8, i32) -> !om.class.type<@Widget>
  %2 = om.object @Widget(%1, %green_1) : (i8, i32) -> !om.class.type<@Widget>
  // CHECK-NEXT: %[[field2]] = om.object.field %[[obj2]], [@green_1] : (!om.class.type<@Widget>) -> i32
  %3 = om.object.field %2, [@green_1] : (!om.class.type<@Widget>) -> i32
}
// CHECK-LABEL: @Bool
om.class @BoolConstant(%b0 : i1) {
  // CHECK: %[[const1:.+]] = om.constant true
  %1 = om.constant true
  // CHECK: %[[const2:.+]] = om.constant false
  %2 = om.constant false
  // CHECK: om.class.field @bool, %b0
  om.class.field @bool, %b0 : i1
  // CHECK: om.class.field @bool2, %[[const1]]
  om.class.field @bool2, %1 : i1
  // CHECK: om.class.field @bool3, %[[const2]]
  om.class.field @bool3, %2 : i1
}

// CHECK-LABEL: @Map
// CHECK-SAME: !om.map<!om.string, !om.string>
om.class @Map(%map: !om.map<!om.string, !om.string>) {
  om.class.field @field, %map : !om.map<!om.string, !om.string>
}

// CHECK-LABEL: @Tuple
om.class @Tuple(%int: i1, %str: !om.string) {
  // CHECK: %[[tuple:.+]] = om.tuple_create %int, %str : i1, !om.string
  %tuple = om.tuple_create %int, %str  : i1, !om.string
  // CHECK-NEXT: om.class.field @tuple, %[[tuple]] : tuple<i1, !om.string>
  om.class.field @tuple, %tuple : tuple<i1, !om.string>
  // CHECK-NEXT: %[[tuple_get:.+]] = om.tuple_get %[[tuple]][1] : tuple<i1, !om.string>
  %val = om.tuple_get %tuple[1]  : tuple<i1, !om.string>
  // CHECK-NEXT: om.class.field @val, %[[tuple_get]] : !om.string
  om.class.field @val, %val : !om.string
}

// CHECK-LABEL: @MapConstant
om.class @MapConstant() {
  // CHECK: %[[const1:.+]] = om.constant #om.map<i64, {a = 42 : i64, b = 32 : i64}> : !om.map<!om.string, i64>
  %0 = om.constant #om.map<i64, {a = 42, b = 32}> : !om.map<!om.string, i64>
  // CHECK: om.class.field @map_i64, %[[const1]] : !om.map<!om.string, i64>
  om.class.field @map_i64, %0 : !om.map<!om.string, i64>
}

// CHECK-LABEL: @MapCreate
om.class @MapCreate(%e1: tuple<!om.string, !om.class.type<@Empty>>, %e2: tuple<!om.string, !om.class.type<@Empty>>) {
  // CHECK: %[[map:.+]] = om.map_create %e1, %e2 : !om.string, !om.class.type<@Empty>
  %map = om.map_create %e1, %e2 : !om.string, !om.class.type<@Empty>
  // CHECK-NEXT: om.class.field @map_field, %[[map]] : !om.map<!om.string, !om.class.type<@Empty>>
  om.class.field @map_field, %map : !om.map<!om.string, !om.class.type<@Empty>>
}

hw.hierpath @HierPath [@PathModule::@wire]
hw.module @PathModule() {
  %wire = hw.wire %wire sym @wire : i1
}
// CHECK-LABEL: @Path
om.class @Path(%basepath: !om.basepath) {
  // CHECK: %[[v0:.+]] = om.basepath_create %basepath @HierPath
  %0 = om.basepath_create %basepath @HierPath
  // CHECK: %[[v1:.+]] = om.path_create reference %basepath @HierPath
  %1 = om.path_create reference %basepath @HierPath
  // CHECK: #om<path[Foo:foo, Bar:bar]>
  %2 = om.constant 1 : i1 { foo = #om<path[Foo:foo, Bar:bar]>}
  // CHECK: %[[v3:.+]] = om.path_empty
  %3 = om.path_empty
  // CHECK: om.class.field @path_empty, %[[v3]] : !om.path
  om.class.field @path_empty, %3 : !om.path
}

om.class @FrozenPath(%basepath: !om.frozenbasepath) {
  // CHECK: %[[v0:.+]] = om.frozenbasepath_create %basepath "Foo/bar"
  %0 = om.frozenbasepath_create %basepath "Foo/bar"
  // CHECK: %[[v1:.+]] = om.frozenpath_create reference %basepath "Foo/bar:Bar>w.a"
  %1 = om.frozenpath_create reference %basepath "Foo/bar:Bar>w.a"
}

// CHECK-LABEL: @Any
// CHECK-SAME: %[[IN:.+]]: !om.class.type
om.class @Any(%in: !om.class.type<@Empty>) {
  // CHECK: %[[CAST:.+]] = om.any_cast %[[IN]]
  %0 = om.any_cast %in : (!om.class.type<@Empty>) -> !om.any
  // CHECK: om.class.field @field, %[[CAST]]
  om.class.field @field, %0 : !om.any
}
