// RUN: circt-opt %s | circt-opt | FileCheck %s

// check inputs and outputs, usage
// CHECK: llhd.entity @foo (%[[ARG0:.*]] : !hw.inout<i64>, %[[ARG1:.*]] : !hw.inout<i64>) -> (%[[OUT0:.*]] : !hw.inout<i64>) {
"llhd.entity"() ({
^body(%arg0 : !hw.inout<i64>, %arg1 : !hw.inout<i64>, %out0 : !hw.inout<i64>):
  // CHECK-NEXT: %[[C0:.*]] = hw.constant 1
  %0 = hw.constant 1 : i64
  // CHECK-NEXT: %[[P0:.*]] = llhd.prb %[[ARG0]]
  %1 = llhd.prb %arg0 : !hw.inout<i64>
// CHECK-NEXT: }
}) {sym_name="foo", ins=2, function_type=(!hw.inout<i64>, !hw.inout<i64>, !hw.inout<i64>)->()} : () -> ()

// check 0 inputs, empty body
// CHECK-NEXT: llhd.entity @bar () -> (%{{.*}} : !hw.inout<i64>) {
"llhd.entity"() ({
^body(%0 : !hw.inout<i64>):
// CHECK-NEXT: }
}) {sym_name="bar", ins=0, function_type=(!hw.inout<i64>)->()} : () -> ()

// check 0 outputs, empty body
// CHECK-NEXT: llhd.entity @baz (%{{.*}} : !hw.inout<i64>) -> () {
"llhd.entity"() ({
^body(%arg0 : !hw.inout<i64>):
// CHECK-NEXT: }
}) {sym_name="baz", ins=1, function_type=(!hw.inout<i64>)->()} : () -> ()

//check 0 arguments, empty body
// CHECK-NEXT: llhd.entity @out_of_names () -> () {
"llhd.entity"() ({
^body:
// CHECK-NEXT: }
}) {sym_name="out_of_names", ins=0, function_type=()->()} : () -> ()
