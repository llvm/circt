// RUN: circt-opt --rtg-insert-test-to-file-mapping="path=dirname split-output=true" %s | FileCheck %s --check-prefix=CHECK-SPLIT
// RUN: circt-opt --rtg-insert-test-to-file-mapping="path=filename split-output=false" %s | FileCheck %s

rtg.test @test0() {}

rtg.test @test1() {}

// CHECK-SPLIT-LABEL: emit.file "dirname{{.+}}test0.s" {
// CHECK-SPLIT-NEXT: emit.ref @test0
// CHECK-SPLIT-NEXT: }

// CHECK-SPLIT-LABEL: emit.file "dirname{{.+}}test1.s" {
// CHECK-SPLIT-NEXT: emit.ref @test1
// CHECK-SPLIT-NEXT: }

// CHECK-LABEL: emit.file "filename" {
// CHECK-NEXT: emit.ref @test0
// CHECK-NEXT: emit.ref @test1
// CHECK-NEXT: }
