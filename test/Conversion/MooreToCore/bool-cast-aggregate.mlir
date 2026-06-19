// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// A packed struct used in a self-determined boolean context (IEEE 1800-2023
// 6.3.2) reduces to "any bit set" by OR-ing the per-field non-zero results.
// CHECK-LABEL: func.func @packedStruct(
// CHECK-SAME:    %arg0: !hw.struct<a: i1, b: i1>) -> i1
func.func @packedStruct(%arg0: !moore.struct<{a: i1, b: i1}>) -> !moore.i1 {
  // CHECK: %[[A:.+]] = hw.struct_extract %arg0["a"]
  // CHECK: %[[NZA:.+]] = comb.icmp ne %[[A]], %{{.+}} : i1
  // CHECK: %[[B:.+]] = hw.struct_extract %arg0["b"]
  // CHECK: %[[NZB:.+]] = comb.icmp ne %[[B]], %{{.+}} : i1
  // CHECK: %[[OR:.+]] = comb.or %[[NZA]], %[[NZB]] : i1
  // CHECK: return %[[OR]] : i1
  %0 = moore.bool_cast %arg0 : struct<{a: i1, b: i1}> -> i1
  return %0 : !moore.i1
}

// A packed array reduces the same way over its elements.
// CHECK-LABEL: func.func @packedArray(
// CHECK-SAME:    %arg0: !hw.array<2xi1>) -> i1
func.func @packedArray(%arg0: !moore.array<2 x i1>) -> !moore.i1 {
  // CHECK: %[[E0:.+]] = hw.array_get %arg0[%{{.+}}]
  // CHECK: %[[NZ0:.+]] = comb.icmp ne %[[E0]], %{{.+}} : i1
  // CHECK: %[[E1:.+]] = hw.array_get %arg0[%{{.+}}]
  // CHECK: %[[NZ1:.+]] = comb.icmp ne %[[E1]], %{{.+}} : i1
  // CHECK: %[[OR:.+]] = comb.or %[[NZ0]], %[[NZ1]] : i1
  // CHECK: return %[[OR]] : i1
  %0 = moore.bool_cast %arg0 : array<2 x i1> -> i1
  return %0 : !moore.i1
}
