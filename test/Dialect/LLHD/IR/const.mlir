//RUN: circt-opt %s | circt-opt | FileCheck %s

func @lap() {
  // CHECK: %{{.*}} = llhd.const 5 : i64
  %0 = "llhd.const"() {value = 5 : i64} : () -> i64
  // CHECK-NEXT: %{{.*}} = llhd.const #llhd.time<1ns, 2d, 3e> : !llhd.time
  %1 = "llhd.const"() {value = #llhd.time<1ns, 2d, 3e> : !llhd.time} : () -> !llhd.time
  return
}
