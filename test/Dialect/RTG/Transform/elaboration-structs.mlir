// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

// Struct construction and projection during elaboration.
rtg.struct.decl @Point { value: index }

func.func @consume_index(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @struct_construct_project_target() template "struct_construct_project" target @target {
// CHECK-NEXT:     %0 = rtg.constant "5" : !rtg.string
// CHECK-NEXT:     rtg.comment %0
// CHECK-NEXT:   }
rtg.test @struct_construct_project() {
  %five = index.constant 5
  %point = rtg.struct_create %five : (index) -> !rtg.struct<@Point>
  %value = rtg.struct_extract %point[value] : !rtg.struct<@Point> -> index
  func.call @consume_index(%value) : (index) -> ()
}

// -----

// Projection from a constant struct.
rtg.struct.decl @Point { value: index }

func.func @consume_index(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @constant_struct_projection_target() template "constant_struct_projection" target @target {
// CHECK-NEXT:     %0 = rtg.constant "7" : !rtg.string
// CHECK-NEXT:     rtg.comment %0
// CHECK-NEXT:   }
rtg.test @constant_struct_projection() {
  %point = rtg.constant #rtg.struct<7 : index> : !rtg.struct<@Point>
  %value = rtg.struct_extract %point[value] : !rtg.struct<@Point> -> index
  func.call @consume_index(%value) : (index) -> ()
}

// -----

// Projection through a nested constant struct.
rtg.struct.decl @Point { value: index }
rtg.struct.decl @Envelope { point: !rtg.struct<@Point> }

func.func @consume_index(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @nested_constant_struct_projection_target() template "nested_constant_struct_projection" target @target {
// CHECK-NEXT:     %0 = rtg.constant "9" : !rtg.string
// CHECK-NEXT:     rtg.comment %0
// CHECK-NEXT:   }
rtg.test @nested_constant_struct_projection() {
  %envelope = rtg.constant #rtg.struct<#rtg.struct<9 : index> : !rtg.struct<@Point>> : !rtg.struct<@Envelope>
  %point = rtg.struct_extract %envelope[point] : !rtg.struct<@Envelope> -> !rtg.struct<@Point>
  %value = rtg.struct_extract %point[value] : !rtg.struct<@Point> -> index
  func.call @consume_index(%value) : (index) -> ()
}

// -----

// Struct values passed through tuples and function calls.
rtg.struct.decl @Pair { first: index, second: index }

func.func @identity_pair(%value: !rtg.struct<@Pair>) -> !rtg.struct<@Pair> {
  return %value : !rtg.struct<@Pair>
}

func.func @pair_from_tuple(
    %value: !rtg.tuple<!rtg.struct<@Pair>, index>) -> !rtg.struct<@Pair> {
  %pair = rtg.tuple_extract %value at 0 : !rtg.tuple<!rtg.struct<@Pair>, index>
  %result = func.call @identity_pair(%pair)
      : (!rtg.struct<@Pair>) -> !rtg.struct<@Pair>
  return %result : !rtg.struct<@Pair>
}

func.func @consume_index(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @struct_values_through_tuples_and_functions_target() template "struct_values_through_tuples_and_functions" target @target {
// CHECK-NEXT:     %0 = rtg.constant "5" : !rtg.string
// CHECK-NEXT:     rtg.comment %0
// CHECK-NEXT:     %1 = rtg.constant "9" : !rtg.string
// CHECK-NEXT:     rtg.comment %1
// CHECK-NEXT:   }
rtg.test @struct_values_through_tuples_and_functions() {
  %five = index.constant 5
  %nine = index.constant 9
  %pair = rtg.struct_create %five, %nine : (index, index) -> !rtg.struct<@Pair>
  %tuple = rtg.tuple_create %pair, %five : !rtg.struct<@Pair>, index
  %result = func.call @pair_from_tuple(%tuple)
      : (!rtg.tuple<!rtg.struct<@Pair>, index>) -> !rtg.struct<@Pair>
  %first = rtg.struct_extract %result[first] : !rtg.struct<@Pair> -> index
  %second = rtg.struct_extract %result[second] : !rtg.struct<@Pair> -> index
  func.call @consume_index(%first) : (index) -> ()
  func.call @consume_index(%second) : (index) -> ()
}

// -----

// Struct values crossing an opaque external region are rematerialized with
// their nominal type and field order intact.
rtg.struct.decl @Point { value: index }

func.func @consume_index(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @struct_materialization_target() template "struct_materialization" target @target {
// CHECK-NEXT:     %0 = rtg.constant #rtg.struct<5 : index> : !rtg.struct<@Point>
// CHECK-NEXT:     %1 = scf.execute_region -> !rtg.struct<@Point> {
// CHECK-NEXT:       scf.yield %0 : !rtg.struct<@Point>
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = rtg.struct_extract %1[value] : !rtg.struct<@Point> -> index
// CHECK-NEXT:     %3 = rtg.int_format %2
// CHECK-NEXT:     rtg.comment %3
// CHECK-NEXT:   }
rtg.test @struct_materialization() {
  %five = index.constant 5
  %point = rtg.struct_create %five : (index) -> !rtg.struct<@Point>
  %result = scf.execute_region -> !rtg.struct<@Point> {
    scf.yield %point : !rtg.struct<@Point>
  }
  %value = rtg.struct_extract %result[value] : !rtg.struct<@Point> -> index
  func.call @consume_index(%value) : (index) -> ()
}
