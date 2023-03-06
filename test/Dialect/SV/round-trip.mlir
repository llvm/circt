// RUN: circt-opt %s --verify-diagnostics --split-input-file | circt-opt --verify-diagnostics | FileCheck %s

// CHECK: sv.dpi.import @test(%arg0: i32, %arg1: i64) -> (res0: i5, res1: i6)
sv.dpi.import @test(%arg0: i32, %arg1: i64) -> (res0: i5, res1: i6)
