// RUN: circt-opt %s | FileCheck %s

hw.type_scope @SVtypes{
  hw.typedecl @net_alias : !sv.net<i8>
  hw.typedecl @var_alias : !sv.var<i8>
}

// CHECK: !sv.net<i8>
// CHECK: !sv.var<i8>
// CHECK: !sv.net<!hw.array<4xi8>>
// CHECK: !sv.var<!hw.struct<a: i8, b: i1>>
hw.module @types() {
  %net = sv.wire : !sv.net<i8>
  %var = sv.var : !sv.var<i8>
  %array = sv.wire : !sv.net<!hw.array<4xi8>>
  %struct = sv.var : !sv.var<!hw.struct<a: i8, b: i1>>
}
