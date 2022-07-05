// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s -test-ssp-roundtrip | circt-opt | FileCheck %s

// 1) tests the plain parser/printer roundtrip.
// 2) roundtrips via the scheduling infra (i.e. populates a `Problem` instance and reconstructs the SSP IR from it.)

// CHECK: ssp.instance "no properties" of "Problem" {
// CHECK:   ssp.operator_type @NoProps
// CHECK:   %[[op_0:.*]] = ssp.operation @Op0()
// CHECK:   ssp.operation(%[[op_0]])
// CHECK:   ssp.operation(@Op0)
// CHECK:   ssp.operation(%[[op_0]], @Op0)
// CHECK: }
ssp.instance "no properties" of "Problem" {
  ssp.operator_type @NoProps
  %0 = ssp.operation @Op0()
  ssp.operation(%0)
  ssp.operation(@Op0)
  ssp.operation(%0, @Op0)
}

// CHECK: ssp.instance "arbitrary_latencies" of "Problem" {
// CHECK:   ssp.operator_type @unit [#ssp.Latency<1>]
// CHECK:   ssp.operator_type @extr [#ssp.Latency<0>]
// CHECK:   ssp.operator_type @add [#ssp.Latency<3>]
// CHECK:   ssp.operator_type @mult [#ssp.Latency<6>]
// CHECK:   ssp.operator_type @sqrt [#ssp.Latency<10>]
// CHECK:   %[[op_0:.*]] = ssp.operation() [#ssp.LinkedOperatorType<@extr>, #ssp.StartTime<0>]
// CHECK:   %[[op_1:.*]] = ssp.operation() [#ssp.LinkedOperatorType<@extr>, #ssp.StartTime<10>]
// CHECK:   %[[op_2:.*]] = ssp.operation(%[[op_0]], %[[op_0]]) [#ssp.LinkedOperatorType<@mult>, #ssp.StartTime<20>]
// CHECK:   %[[op_3:.*]] = ssp.operation(%[[op_1]], %[[op_1]]) [#ssp.LinkedOperatorType<@mult>, #ssp.StartTime<30>]
// CHECK:   %[[op_4:.*]] = ssp.operation(%[[op_2]], %[[op_3]]) [#ssp.LinkedOperatorType<@add>, #ssp.StartTime<40>]
// CHECK:   %[[op_5:.*]] = ssp.operation(%[[op_4]]) [#ssp.LinkedOperatorType<@sqrt>, #ssp.StartTime<50>]
// CHECK:   ssp.operation(%[[op_5]]) [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<60>]
// CHECK: }
ssp.instance "arbitrary_latencies" of "Problem" {
  ssp.operator_type @unit [#ssp.Latency<1>]
  ssp.operator_type @extr [#ssp.Latency<0>]
  ssp.operator_type @add [#ssp.Latency<3>]
  ssp.operator_type @mult [#ssp.Latency<6>]
  ssp.operator_type @sqrt [#ssp.Latency<10>]
  %0 = ssp.operation() [#ssp.LinkedOperatorType<@extr>, #ssp.StartTime<0>]
  %1 = ssp.operation() [#ssp.LinkedOperatorType<@extr>, #ssp.StartTime<10>]
  %2 = ssp.operation(%0, %0) [#ssp.LinkedOperatorType<@mult>, #ssp.StartTime<20>]
  %3 = ssp.operation(%1, %1) [#ssp.LinkedOperatorType<@mult>, #ssp.StartTime<30>]
  %4 = ssp.operation(%2, %3) [#ssp.LinkedOperatorType<@add>, #ssp.StartTime<40>]
  %5 = ssp.operation(%4) [#ssp.LinkedOperatorType<@sqrt>, #ssp.StartTime<50>]
  ssp.operation(%5) [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<60>]
}

// CHECK: ssp.instance "self_arc" of "CyclicProblem" [#ssp.InitiationInterval<3>] {
// CHECK:   ssp.operator_type @unit [#ssp.Latency<1>]
// CHECK:   ssp.operator_type @_3 [#ssp.Latency<3>]
// CHECK:   %[[op_0:.*]] = ssp.operation() [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<0>]
// CHECK:   %[[op_1:.*]] = ssp.operation @self(%[[op_0]], %[[op_0]], @self [#ssp.Distance<1>]) [#ssp.LinkedOperatorType<@_3>, #ssp.StartTime<1>]
// CHECK:   ssp.operation(%[[op_1]]) [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<4>]
// CHECK: }
ssp.instance "self_arc" of "CyclicProblem" [#ssp.InitiationInterval<3>] {
  ssp.operator_type @unit [#ssp.Latency<1>]
  ssp.operator_type @_3 [#ssp.Latency<3>]
  %0 = ssp.operation() [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<0>]
  %1 = ssp.operation @self(%0, %0, @self [#ssp.Distance<1>]) [#ssp.LinkedOperatorType<@_3>, #ssp.StartTime<1>]
  ssp.operation(%1) [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<4>]
}

// CHECK: ssp.instance "multiple_oprs" of "SharedOperatorsProblem" {
// CHECK:   ssp.operator_type @slowAdd [#ssp.Latency<3>, #ssp.Limit<2>]
// CHECK:   ssp.operator_type @fastAdd [#ssp.Latency<1>, #ssp.Limit<1>]
// CHECK:   ssp.operator_type @_0 [#ssp.Latency<0>]
// CHECK:   ssp.operator_type @_1 [#ssp.Latency<1>]
// CHECK:   %[[op_0:.*]] = ssp.operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<0>]
// CHECK:   %[[op_1:.*]] = ssp.operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<1>]
// CHECK:   %[[op_2:.*]] = ssp.operation() [#ssp.LinkedOperatorType<@fastAdd>, #ssp.StartTime<0>]
// CHECK:   %[[op_3:.*]] = ssp.operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<1>]
// CHECK:   %[[op_4:.*]] = ssp.operation() [#ssp.LinkedOperatorType<@fastAdd>, #ssp.StartTime<1>]
// CHECK:   %[[op_5:.*]] = ssp.operation(%[[op_0]], %[[op_1]], %[[op_2]], %[[op_3]], %[[op_4]]) [#ssp.LinkedOperatorType<@_0>, #ssp.StartTime<10>]
// CHECK:   ssp.operation() [#ssp.LinkedOperatorType<@_1>, #ssp.StartTime<10>]
// CHECK: }
ssp.instance "multiple_oprs" of "SharedOperatorsProblem" {
  ssp.operator_type @slowAdd [#ssp.Latency<3>, #ssp.Limit<2>]
  ssp.operator_type @fastAdd [#ssp.Latency<1>, #ssp.Limit<1>]
  ssp.operator_type @_0 [#ssp.Latency<0>]
  ssp.operator_type @_1 [#ssp.Latency<1>]
  %0 = ssp.operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<0>]
  %1 = ssp.operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<1>]
  %2 = ssp.operation() [#ssp.LinkedOperatorType<@fastAdd>, #ssp.StartTime<0>]
  %3 = ssp.operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<1>]
  %4 = ssp.operation() [#ssp.LinkedOperatorType<@fastAdd>, #ssp.StartTime<1>]
  %5 = ssp.operation(%0, %1, %2, %3, %4) [#ssp.LinkedOperatorType<@_0>, #ssp.StartTime<10>]
  ssp.operation() [#ssp.LinkedOperatorType<@_1>, #ssp.StartTime<10>]
}

// CHECK: ssp.instance "canis14_fig2" of "ModuloProblem" [#ssp.InitiationInterval<3>] {
// CHECK:   ssp.operator_type @MemPort [#ssp.Latency<1>, #ssp.Limit<1>]
// CHECK:   ssp.operator_type @Add [#ssp.Latency<1>]
// CHECK:   ssp.operator_type @Implicit [#ssp.Latency<0>]
// CHECK:   %[[op_0:.*]] = ssp.operation @load_A(@store_A [#ssp.Distance<1>]) [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<2>]
// CHECK:   %[[op_1:.*]] = ssp.operation @load_B() [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<0>]
// CHECK:   %[[op_2:.*]] = ssp.operation @add(%[[op_0]], %[[op_1]]) [#ssp.LinkedOperatorType<@Add>, #ssp.StartTime<3>]
// CHECK:   ssp.operation @store_A(%[[op_2]]) [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<4>]
// CHECK:   ssp.operation @last(@store_A) [#ssp.LinkedOperatorType<@Implicit>, #ssp.StartTime<5>]
// CHECK: }
ssp.instance "canis14_fig2" of "ModuloProblem" [#ssp.InitiationInterval<3>] {
  ssp.operator_type @MemPort [#ssp.Latency<1>, #ssp.Limit<1>]
  ssp.operator_type @Add [#ssp.Latency<1>]
  ssp.operator_type @Implicit [#ssp.Latency<0>]
  %0 = ssp.operation @load_A(@store_A [#ssp.Distance<1>]) [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<2>]
  %1 = ssp.operation @load_B() [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<0>]
  %2 = ssp.operation @add(%0, %1) [#ssp.LinkedOperatorType<@Add>, #ssp.StartTime<3>]
  ssp.operation @store_A(%2) [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<4>]
  ssp.operation @last(@store_A) [#ssp.LinkedOperatorType<@Implicit>, #ssp.StartTime<5>]
}
