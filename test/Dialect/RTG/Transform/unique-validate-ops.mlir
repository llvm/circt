// RUN: circt-opt --rtg-unique-validate %s | FileCheck %s

// CHECK-LABEL: @validate
rtg.test @validate() {
  %0 = rtg.fixed_reg #rtgtest.t0
  %1 = rtg.constant #rtg.isa.immediate<32, 0>

  // CHECK: rtg.validate %{{.*}}, %{{.*}}, "validation_id_0"
  // CHECK: rtg.validate %{{.*}}, %{{.*}}, "validation_id"
  // CHECK: rtg.validate %{{.*}}, %{{.*}}, "validation_id_1"
  %2 = rtg.validate %0, %1, "validation_id_0" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  %3 = rtg.validate %0, %1 : !rtgtest.ireg -> !rtg.isa.immediate<32>
  %4 = rtg.validate %0, %1 : !rtgtest.ireg -> !rtg.isa.immediate<32>
}
