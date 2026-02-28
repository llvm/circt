// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test checks that only the reduction patterns of dialects that occur in
// the input file are registered

// RUN: circt-reduce %s --test /usr/bin/env --test-arg cat --list | FileCheck %s

// CHECK-DAG: arc-strip-sv
// CHECK-DAG: cse
// CHECK-DAG: arc-dedup
// CHECK-DAG: canonicalize
// CHECK-DAG: arc-state-elimination
// CHECK-DAG: operation-pruner
// CHECK-DAG: arc-canonicalizer
arc.define @DummyArc(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
}

