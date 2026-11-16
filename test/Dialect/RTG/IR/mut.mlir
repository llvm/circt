// RUN: circt-opt %s --verify-roundtrip | FileCheck %s

// CHECK-LABEL: @mut_create_read_write
rtg.test @mut_create_read_write() {
  %zero = index.constant 0
  // CHECK: [[REF:%.+]] = rtg.mut_create [[INIT:%.+]] : index
  %ref = rtg.mut_create %zero : index
  // CHECK: [[V:%.+]] = rtg.mut_read [[REF]] : !rtg.mut<index>
  %v = rtg.mut_read %ref : !rtg.mut<index>
  %one = index.constant 1
  %v2 = index.add %v, %one
  // CHECK: rtg.mut_write [[REF]], {{.+}} : !rtg.mut<index>
  rtg.mut_write %ref, %v2 : !rtg.mut<index>
}
