// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL:   ssp.instance "canis14_fig2" of "ModuloProblem" {
// CHECK:           ssp.operator_type @MemPort {latency = 1 : i32, limit = 1 : i32}
// CHECK:           ssp.operator_type @Add {latency = 1 : i32}
// CHECK:           ssp.operator_type @Implicit {latency = 0 : i32}
// CHECK:           %[[op_0:.*]] = ssp.operation @load_A(@store_A {distance = 1 : i32}) {opr = @MemPort, startTime = 2 : i32}
// CHECK:           %[[op_1:.*]] = ssp.operation @load_B() {opr = @MemPort, startTime = 0 : i32}
// CHECK:           %[[op_2:.*]] = ssp.operation @add(%[[op_0]], %[[op_1]]) {opr = @Add, startTime = 3 : i32}
// CHECK:           ssp.operation @store_A(%[[op_2]]) {opr = @MemPort, startTime = 4 : i32}
// CHECK:           ssp.operation @last(@store_A) {opr = @Implicit, startTime = 5 : i32}
// CHECK:         } {initiationInterval = 3 : i64}

ssp.instance "canis14_fig2" of "ModuloProblem" {
  ssp.operator_type @MemPort {latency = 1 : i32, limit = 1 :i32}
  ssp.operator_type @Add { latency = 1 : i32}
  ssp.operator_type @Implicit { latency = 0 : i32}
  %0 = ssp.operation @load_A(@store_A {distance = 1 : i32}) {opr = @MemPort, startTime = 2 : i32}
  %1 = ssp.operation @load_B() {opr = @MemPort, startTime = 0 : i32}
  %2 = ssp.operation @add(%0, %1) {opr = @Add, startTime = 3 : i32}
  ssp.operation @store_A(%2) {opr = @MemPort, startTime = 4 : i32}
  ssp.operation @last(@store_A) {opr = @Implicit, startTime = 5 : i32}
} {initiationInterval = 3}
