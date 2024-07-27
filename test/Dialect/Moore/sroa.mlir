// RUN: circt-opt --sroa %s | FileCheck %s

// CHECK-LABEL: func.func @SplitStructs
func.func @SplitStructs(%arg0: !moore.i42, %arg1: !moore.i1337) {
  // Named variables

  // CHECK-DAG: %x.a = moore.variable : <i42>
  // CHECK-DAG: %x.b = moore.variable : <i1337>
  // CHECK-NOT: moore.struct_extract_ref
  %x = moore.variable : <struct<{a: i42, b: i1337}>>
  %0 = moore.struct_extract_ref %x, "a" : <struct<{a: i42, b: i1337}>> -> <i42>
  %1 = moore.struct_extract_ref %x, "b" : <struct<{a: i42, b: i1337}>> -> <i1337>
  // CHECK: moore.blocking_assign %x.a, %arg0
  // CHECK: moore.blocking_assign %x.b, %arg1
  moore.blocking_assign %0, %arg0 : !moore.i42
  moore.blocking_assign %1, %arg1 : !moore.i1337

  // Anonymous variables

  // CHECK-DAG: [[A:%.+]] = moore.variable : <i42>
  // CHECK-DAG: [[B:%.+]] = moore.variable : <i1337>
  // CHECK-NOT: moore.struct_extract_ref
  %2 = moore.variable : <struct<{a: i42, b: i1337}>>
  %3 = moore.struct_extract_ref %2, "a" : <struct<{a: i42, b: i1337}>> -> <i42>
  %4 = moore.struct_extract_ref %2, "b" : <struct<{a: i42, b: i1337}>> -> <i1337>
  // CHECK: moore.blocking_assign [[A]], %arg0
  // CHECK: moore.blocking_assign [[B]], %arg1
  moore.blocking_assign %3, %arg0 : !moore.i42
  moore.blocking_assign %4, %arg1 : !moore.i1337

  return
}

// CHECK-LABEL: func.func @SplitNestedStructs
func.func @SplitNestedStructs(%arg0: !moore.i42, %arg1: !moore.i1337, %arg2: !moore.i9001) {
  // CHECK-DAG: %k.a.x = moore.variable : <i42>
  // CHECK-DAG: %k.a.y = moore.variable : <i1337>
  // CHECK-DAG: %k.b.u = moore.variable : <i1337>
  // CHECK-DAG: %k.b.v = moore.variable : <i9001>
  %k = moore.variable : <struct<{a: struct<{x: i42, y: i1337}>, b: struct<{u: i1337, v: i9001}>}>>
  %0 = moore.struct_extract_ref %k, "a" : <struct<{a: struct<{x: i42, y: i1337}>, b: struct<{u: i1337, v: i9001}>}>> -> <struct<{x: i42, y: i1337}>>
  %1 = moore.struct_extract_ref %k, "b" : <struct<{a: struct<{x: i42, y: i1337}>, b: struct<{u: i1337, v: i9001}>}>> -> <struct<{u: i1337, v: i9001}>>
  %2 = moore.struct_extract_ref %0, "x" : <struct<{x: i42, y: i1337}>> -> <i42>
  %3 = moore.struct_extract_ref %0, "y" : <struct<{x: i42, y: i1337}>> -> <i1337>
  %4 = moore.struct_extract_ref %1, "u" : <struct<{u: i1337, v: i9001}>> -> <i1337>
  %5 = moore.struct_extract_ref %1, "v" : <struct<{u: i1337, v: i9001}>> -> <i9001>
  // CHECK: moore.blocking_assign %k.a.x, %arg0
  // CHECK: moore.blocking_assign %k.a.y, %arg1
  // CHECK: moore.blocking_assign %k.b.u, %arg1
  // CHECK: moore.blocking_assign %k.b.v, %arg2
  moore.blocking_assign %2, %arg0 : !moore.i42
  moore.blocking_assign %3, %arg1 : !moore.i1337
  moore.blocking_assign %4, %arg1 : !moore.i1337
  moore.blocking_assign %5, %arg2 : !moore.i9001
  return
}
