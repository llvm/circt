// RUN: circt-opt %s -verify-diagnostics --mlir-print-local-scope --mlir-print-debuginfo | circt-opt --mlir-print-local-scope --mlir-print-debuginfo -verify-diagnostics | FileCheck %s

// CHECK-LABEL: om.class @Thingy
// CHECK-SAME: (%blue_1: i8, %blue_2: i32) -> (widget: !om.class.type<@Widget>, gadget: !om.class.type<@Gadget>, blue_1: i8, blue_2: i8)
om.class @Thingy(%blue_1: i8, %blue_2: i32) -> (widget: !om.class.type<@Widget>, gadget: !om.class.type<@Gadget>, blue_1: i8, blue_2: i8) {
  // CHECK: %[[c5:.+]] = om.constant 5 : i8
  %0 = om.constant 5 : i8
  // CHECK: %[[c6:.+]] = om.constant 6 : i32
  %1 = om.constant 6 : i32
  // CHECK: %[[widget:.+]] = om.object @Widget(%[[c5]], %[[c6]]) : (i8, i32) -> !om.class.type<@Widget>
  %2 = om.object @Widget(%0, %1) : (i8, i32) -> !om.class.type<@Widget>

  // CHECK: %[[c7:.+]] = om.constant 7 : i8
  %3 = om.constant 7 : i8
  // CHECK: %[[c8:.+]] = om.constant 8 : i32
  %4 = om.constant 8 : i32
  // CHECK: %[[gadget:.+]] = om.object @Gadget(%[[c7]], %[[c8]]) : (i8, i32) -> !om.class.type<@Gadget>
  %5 = om.object @Gadget(%3, %4) : (i8, i32) -> !om.class.type<@Gadget>

  // CHECK: %[[widget_field:.+]] = om.object.field %[[widget]], [@blue_1] : (!om.class.type<@Widget>) -> i8
  %6 = om.object.field %2, [@blue_1] : (!om.class.type<@Widget>) -> i8

  // CHECK: om.class.fields {test = "fieldsAttr"} %2, %5, %blue_1, %6 : !om.class.type<@Widget>, !om.class.type<@Gadget>, i8, i8 field_locs([loc("loc0"), loc("loc1"), loc("loc2"), loc("loc3")]) loc("test")
  om.class.fields {test = "fieldsAttr"} %2, %5, %blue_1, %6 : !om.class.type<@Widget>, !om.class.type<@Gadget>, i8, i8 field_locs([loc("loc0"), loc("loc1"), loc("loc2"), loc("loc3")]) loc("test")
}

// CHECK-LABEL: om.class @Widget
// CHECK-SAME: (%blue_1: i8, %green_1: i32) -> (blue_1: i8, green_1: i32)
om.class @Widget(%blue_1: i8, %green_1: i32) -> (blue_1: i8, green_1: i32) {
  // CHECK: om.class.fields %blue_1, %green_1 : i8, i32
  om.class.fields %blue_1, %green_1 : i8, i32
}

// CHECK-LABEL: om.class @Gadget
// CHECK-SAME: (%green_1: i8, %green_2: i32) -> (green_1: i8, green_2: i32)
om.class @Gadget(%green_1: i8, %green_2: i32) -> (green_1: i8, green_2: i32) {
  // CHECK: om.class.fields %green_1, %green_2 : i8, i32
  om.class.fields %green_1, %green_2 : i8, i32
}
 
// CHECK-LABEL: om.class @Empty
om.class @Empty() {
  om.class.fields
}

// CHECK-LABEL: om.class @DiscardableAttrs
// CHECK-SAME: attributes {foo.bar = "baz"}
om.class @DiscardableAttrs() attributes {foo.bar="baz"} {
  om.class.fields
}

// CHECK-LABEL: om.class.extern @Extern
// CHECK-SAME: (%param1: i1, %param2: i2) -> (field1: i3, field2: i4)
om.class.extern @Extern(%param1: i1, %param2: i2) -> (field1 : i3, field2 : i4) {}

// CHECK-LABEL: om.class @ExternObject
// CHECK-SAME: (%[[P0:.+]]: i1, %[[P1:.+]]: i2)
om.class @ExternObject(%param1: i1, %param2: i2) {
  // CHECK: %[[O0:.+]] = om.object @Extern(%[[P0]], %[[P1]])
  %0 = om.object @Extern(%param1, %param2) : (i1, i2) -> !om.class.type<@Extern>

  // CHECK: om.object.field %[[O0]], [@field1]
  %1 = om.object.field %0, [@field1] : (!om.class.type<@Extern>) -> i3
  om.class.fields
}

om.class @NestedField1() -> (baz: i1) {
  %0 = om.constant 1 : i1
  om.class.fields %0 : i1
}

om.class @NestedField2() -> (bar: !om.class.type<@NestedField1>) {
  %0 = om.object @NestedField1() : () -> !om.class.type<@NestedField1>
  om.class.fields %0 : !om.class.type<@NestedField1>
}

om.class @NestedField3() -> (foo: !om.class.type<@NestedField2>) {
  %0 = om.object @NestedField2() : () -> !om.class.type<@NestedField2>
  om.class.fields %0 : !om.class.type<@NestedField2>
}

// CHECK-LABEL: @NestedField4
om.class @NestedField4() {
  // CHECK: %[[nested:.+]] = om.object @NestedField3
  %0 = om.object @NestedField3() : () -> !om.class.type<@NestedField3>
  // CHECK: %{{.+}} = om.object.field %[[nested]], [@foo, @bar, @baz] : (!om.class.type<@NestedField3>) -> i1
  %1 = om.object.field %0, [@foo, @bar, @baz] : (!om.class.type<@NestedField3>) -> i1
  om.class.fields
}

// CHECK-LABEL: @ReferenceParameter
// CHECK-SAME: !om.ref
// CHECK-SAME: !om.sym_ref
// CHECK-SAME: -> (myref: !om.ref, sym: !om.sym_ref)
om.class @ReferenceParameter(%arg0: !om.ref, %arg1: !om.sym_ref) -> (myref: !om.ref, sym: !om.sym_ref) {
  // CHECK: om.class.fields %arg0, %arg1 : !om.ref, !om.sym_ref
  om.class.fields %arg0, %arg1 : !om.ref, !om.sym_ref
}

// CHECK-LABEL: @ReferenceConstant
// CHECK-SAME: -> (myref: !om.ref, sym: !om.sym_ref)
om.class @ReferenceConstant() -> (myref: !om.ref, sym: !om.sym_ref) {
  // CHECK: %[[const1:.+]] = om.constant #om.ref<<@A::@inst_1>> : !om.ref
  %0 = om.constant #om.ref<#hw.innerNameRef<@A::@inst_1>> : !om.ref

  // CHECK: %[[const2:.+]] = om.constant #om.sym_ref<@A> : !om.sym_ref
  %1 = om.constant #om.sym_ref<@A> : !om.sym_ref

  // CHECK: om.class.fields %[[const1]], %[[const2]] : !om.ref, !om.sym_ref
  om.class.fields %0, %1 : !om.ref, !om.sym_ref
}

// CHECK-LABEL: @ListConstant
// CHECK-SAME: -> (list_i64: !om.list<i64>, list_i32: !om.list<i32>)
om.class @ListConstant() -> (list_i64: !om.list<i64>, list_i32: !om.list<i32>) {
  // CHECK: %[[const1:.+]] = om.constant #om.list<i64, [42]> : !om.list<i64>
  %0 = om.constant #om.list<i64, [42]> : !om.list<i64>

  // CHECK: %[[const2:.+]] = om.constant #om.list<i32, []> : !om.list<i32>
  %1 = om.constant #om.list<i32, []> : !om.list<i32>

  // CHECK: om.class.fields %[[const1]], %[[const2]] : !om.list<i64>, !om.list<i32>
  om.class.fields %0, %1 : !om.list<i64>, !om.list<i32>
}

// CHECK-LABEL: @ListCreate
// CHECK-SAME: -> (list_field: !om.list<!om.class.type<@Widget>>)
om.class @ListCreate() -> (list_field: !om.list<!om.class.type<@Widget>>) {
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

  // CHECK: om.class.fields [[list]] : !om.list<!om.class.type<@Widget>>
  om.class.fields %list : !om.list<!om.class.type<@Widget>>
}

// CHECK-LABEL: @ListConcat
om.class @ListConcat() {
  %0 = om.constant #om.integer<0 : i8> : !om.integer
  %1 = om.constant #om.integer<1 : i8> : !om.integer
  %2 = om.constant #om.integer<2 : i8> : !om.integer

  // CHECK: [[L0:%.+]] = om.list_create %0, %1
  %l0 = om.list_create %0, %1 : !om.integer

  // CHECK: [[L1:%.+]] = om.list_create %2
  %l1 = om.list_create %2 : !om.integer

  // CHECK: om.list_concat [[L0]], [[L1]]
  %concat = om.list_concat %l0, %l1 : !om.list<!om.integer>

  om.class.fields
}

// CHECK-LABEL: @Integer
// CHECK-SAME: -> (int: !om.integer)
om.class @IntegerConstant() -> (int: !om.integer) {
  // CHECK: %[[const1:.+]] = om.constant #om.integer<36755551979133953793 : i67> : !om.integer
  %0 = om.constant #om.integer<36755551979133953793 : i67> : !om.integer
  // CHECK: om.class.fields %[[const1]] : !om.integer
  om.class.fields %0 : !om.integer
}

// CHECK-LABEL: @String
// CHECK-SAME: -> (string: !om.string)
om.class @StringConstant() -> (string: !om.string) {
  // CHECK: %[[const1:.+]] = om.constant "foo" : !om.string
  %0 = om.constant "foo" : !om.string
  // CHECK: om.class.fields %[[const1]] : !om.string
  om.class.fields %0 : !om.string
}

// CHECK-LABEL: @LinkedList
// CHECK-SAME: -> (prev: !om.class.type<@LinkedList>)
om.class @LinkedList(%prev: !om.class.type<@LinkedList>) -> (prev: !om.class.type<@LinkedList>) {
  om.class.fields %prev : !om.class.type<@LinkedList>
}

// CHECK-LABEL: @ReferenceEachOther
om.class @ReferenceEachOther() {
  // CHECK-NEXT: %[[obj1:.+]] = om.object @LinkedList(%[[obj2:.+]]) : (!om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
  // CHECK-NEXT: %[[obj2]] = om.object @LinkedList(%[[obj1]]) : (!om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
  %0 = om.object @LinkedList(%1) : (!om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
  %1 = om.object @LinkedList(%0) : (!om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
  om.class.fields
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
  om.class.fields
}

// CHECK-LABEL: @Bool
// CHECK-SAME: -> (bool: i1, bool2: i1, bool3: i1)
om.class @BoolConstant(%b0 : i1) -> (bool: i1, bool2: i1, bool3: i1) {
  // CHECK: %[[const1:.+]] = om.constant true
  %1 = om.constant true
  // CHECK: %[[const2:.+]] = om.constant false
  %2 = om.constant false
  // CHECK: om.class.fields %b0, %[[const1]], %[[const2]] : i1, i1, i1
  om.class.fields %b0, %1, %2 : i1, i1, i1
}

// CHECK-LABEL: @Tuple
// CHECK-SAME: -> (tuple: tuple<i1, !om.string>, val: !om.string)
om.class @Tuple(%int: i1, %str: !om.string) -> (tuple: tuple<i1, !om.string>, val: !om.string) {
  // CHECK: %[[tuple:.+]] = om.tuple_create %int, %str : i1, !om.string
  %tuple = om.tuple_create %int, %str  : i1, !om.string
  // CHECK-NEXT: %[[tuple_get:.+]] = om.tuple_get %[[tuple]][1] : tuple<i1, !om.string>
  %val = om.tuple_get %tuple[1]  : tuple<i1, !om.string>
  // CHECK-NEXT: om.class.fields %[[tuple]], %[[tuple_get]] : tuple<i1, !om.string>, !om.string
  om.class.fields %tuple, %val : tuple<i1, !om.string>, !om.string
}

hw.hierpath @HierPath [@PathModule::@wire]
hw.module @PathModule() {
  %wire = hw.wire %wire sym @wire : i1
}

// CHECK-LABEL: @Path
// CHECK: -> (path_empty: !om.path)
om.class @Path(%basepath: !om.basepath) -> (path_empty: !om.path) {
  // CHECK: %[[v0:.+]] = om.basepath_create %basepath @HierPath
  %0 = om.basepath_create %basepath @HierPath
  // CHECK: %[[v1:.+]] = om.path_create reference %basepath @HierPath
  %1 = om.path_create reference %basepath @HierPath
  // CHECK: #om<path[Foo:foo, Bar:bar]>
  %2 = om.constant 1 : i1 { foo = #om<path[Foo:foo, Bar:bar]>}
  // CHECK: %[[v3:.+]] = om.path_empty
  %3 = om.path_empty
  // CHECK: om.class.fields %[[v3]] : !om.path
  om.class.fields %3 : !om.path
}

om.class @FrozenPath(%basepath: !om.frozenbasepath) {
  // CHECK: %[[v0:.+]] = om.frozenbasepath_create %basepath "Foo/bar"
  %0 = om.frozenbasepath_create %basepath "Foo/bar"
  // CHECK: %[[v1:.+]] = om.frozenpath_create reference %basepath "Foo/bar:Bar>w.a"
  %1 = om.frozenpath_create reference %basepath "Foo/bar:Bar>w.a"
  om.class.fields
}

// CHECK-LABEL: @Any
// CHECK-SAME: %[[IN:.+]]: !om.class.type
// CHECK-SAME: -> (field: !om.any)
om.class @Any(%in: !om.class.type<@Empty>) -> (field: !om.any) {
  // CHECK: %[[CAST:.+]] = om.any_cast %[[IN]]
  %0 = om.any_cast %in : (!om.class.type<@Empty>) -> !om.any
  // CHECK: om.class.fields %[[CAST]] : !om.any
  om.class.fields %0 : !om.any
}

// CHECK-LABEL: @IntegerArithmetic
om.class @IntegerArithmetic() {
  %0 = om.constant #om.integer<1 : si3> : !om.integer
  %1 = om.constant #om.integer<2 : si3> : !om.integer

  // CHECK: om.integer.add %0, %1 : !om.integer
  %2 = om.integer.add %0, %1 : !om.integer

  // CHECK: om.integer.mul %0, %1 : !om.integer
  %3 = om.integer.mul %0, %1 : !om.integer

  // CHECK: om.integer.shr %0, %1 : !om.integer
  %4 = om.integer.shr %0, %1 : !om.integer

  // CHECK: om.integer.shl %0, %1 : !om.integer
  %5 = om.integer.shl %0, %1 : !om.integer

  om.class.fields
}
