// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s -test-ssp-roundtrip | circt-opt | FileCheck %s

// 1) tests the plain parser/printer roundtrip.
// 2) roundtrips via the scheduling infra (i.e. populates a `Problem` instance and reconstructs the SSP IR from it.)

// CHECK: ssp.instance "no properties" of "Problem" {
// CHECK:   operator_type @NoProps
// CHECK:   %[[op_0:.*]] = operation @Op0()
// CHECK:   operation(%[[op_0]])
// CHECK:   operation(@Op0)
// CHECK:   operation(%[[op_0]], @Op0)
// CHECK: }
ssp.instance "no properties" of "Problem" {
  operator_type @NoProps
  %0 = operation @Op0()
  operation(%0)
  operation(@Op0)
  operation(%0, @Op0)
}

// CHECK: ssp.instance "arbitrary_latencies" of "Problem" {
// CHECK:   operator_type @unit [#ssp.Latency<1>]
// CHECK:   operator_type @extr [#ssp.Latency<0>]
// CHECK:   operator_type @add [#ssp.Latency<3>]
// CHECK:   operator_type @mult [#ssp.Latency<6>]
// CHECK:   operator_type @sqrt [#ssp.Latency<10>]
// CHECK:   %[[op_0:.*]] = operation() [#ssp.LinkedOperatorType<@extr>, #ssp.StartTime<0>]
// CHECK:   %[[op_1:.*]] = operation() [#ssp.LinkedOperatorType<@extr>, #ssp.StartTime<10>]
// CHECK:   %[[op_2:.*]] = operation(%[[op_0]], %[[op_0]]) [#ssp.LinkedOperatorType<@mult>, #ssp.StartTime<20>]
// CHECK:   %[[op_3:.*]] = operation(%[[op_1]], %[[op_1]]) [#ssp.LinkedOperatorType<@mult>, #ssp.StartTime<30>]
// CHECK:   %[[op_4:.*]] = operation(%[[op_2]], %[[op_3]]) [#ssp.LinkedOperatorType<@add>, #ssp.StartTime<40>]
// CHECK:   %[[op_5:.*]] = operation(%[[op_4]]) [#ssp.LinkedOperatorType<@sqrt>, #ssp.StartTime<50>]
// CHECK:   operation(%[[op_5]]) [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<60>]
// CHECK: }
ssp.instance "arbitrary_latencies" of "Problem" {
  operator_type @unit [#ssp.Latency<1>]
  operator_type @extr [#ssp.Latency<0>]
  operator_type @add [#ssp.Latency<3>]
  operator_type @mult [#ssp.Latency<6>]
  operator_type @sqrt [#ssp.Latency<10>]
  %0 = operation() [#ssp.LinkedOperatorType<@extr>, #ssp.StartTime<0>]
  %1 = operation() [#ssp.LinkedOperatorType<@extr>, #ssp.StartTime<10>]
  %2 = operation(%0, %0) [#ssp.LinkedOperatorType<@mult>, #ssp.StartTime<20>]
  %3 = operation(%1, %1) [#ssp.LinkedOperatorType<@mult>, #ssp.StartTime<30>]
  %4 = operation(%2, %3) [#ssp.LinkedOperatorType<@add>, #ssp.StartTime<40>]
  %5 = operation(%4) [#ssp.LinkedOperatorType<@sqrt>, #ssp.StartTime<50>]
  operation(%5) [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<60>]
}

// CHECK: ssp.instance "self_arc" of "CyclicProblem" [#ssp.InitiationInterval<3>] {
// CHECK:   operator_type @unit [#ssp.Latency<1>]
// CHECK:   operator_type @_3 [#ssp.Latency<3>]
// CHECK:   %[[op_0:.*]] = operation() [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<0>]
// CHECK:   %[[op_1:.*]] = operation @self(%[[op_0]], %[[op_0]], @self [#ssp.Distance<1>]) [#ssp.LinkedOperatorType<@_3>, #ssp.StartTime<1>]
// CHECK:   operation(%[[op_1]]) [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<4>]
// CHECK: }
ssp.instance "self_arc" of "CyclicProblem" [#ssp.InitiationInterval<3>] {
  operator_type @unit [#ssp.Latency<1>]
  operator_type @_3 [#ssp.Latency<3>]
  %0 = operation() [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<0>]
  %1 = operation @self(%0, %0, @self [#ssp.Distance<1>]) [#ssp.LinkedOperatorType<@_3>, #ssp.StartTime<1>]
  operation(%1) [#ssp.LinkedOperatorType<@unit>, #ssp.StartTime<4>]
}

// CHECK: ssp.instance "multiple_oprs" of "SharedOperatorsProblem" {
// CHECK:   operator_type @slowAdd [#ssp.Latency<3>, #ssp.Limit<2>]
// CHECK:   operator_type @fastAdd [#ssp.Latency<1>, #ssp.Limit<1>]
// CHECK:   operator_type @_0 [#ssp.Latency<0>]
// CHECK:   operator_type @_1 [#ssp.Latency<1>]
// CHECK:   %[[op_0:.*]] = operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<0>]
// CHECK:   %[[op_1:.*]] = operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<1>]
// CHECK:   %[[op_2:.*]] = operation() [#ssp.LinkedOperatorType<@fastAdd>, #ssp.StartTime<0>]
// CHECK:   %[[op_3:.*]] = operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<1>]
// CHECK:   %[[op_4:.*]] = operation() [#ssp.LinkedOperatorType<@fastAdd>, #ssp.StartTime<1>]
// CHECK:   %[[op_5:.*]] = operation(%[[op_0]], %[[op_1]], %[[op_2]], %[[op_3]], %[[op_4]]) [#ssp.LinkedOperatorType<@_0>, #ssp.StartTime<10>]
// CHECK:   operation() [#ssp.LinkedOperatorType<@_1>, #ssp.StartTime<10>]
// CHECK: }
ssp.instance "multiple_oprs" of "SharedOperatorsProblem" {
  operator_type @slowAdd [#ssp.Latency<3>, #ssp.Limit<2>]
  operator_type @fastAdd [#ssp.Latency<1>, #ssp.Limit<1>]
  operator_type @_0 [#ssp.Latency<0>]
  operator_type @_1 [#ssp.Latency<1>]
  %0 = operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<0>]
  %1 = operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<1>]
  %2 = operation() [#ssp.LinkedOperatorType<@fastAdd>, #ssp.StartTime<0>]
  %3 = operation() [#ssp.LinkedOperatorType<@slowAdd>, #ssp.StartTime<1>]
  %4 = operation() [#ssp.LinkedOperatorType<@fastAdd>, #ssp.StartTime<1>]
  %5 = operation(%0, %1, %2, %3, %4) [#ssp.LinkedOperatorType<@_0>, #ssp.StartTime<10>]
  operation() [#ssp.LinkedOperatorType<@_1>, #ssp.StartTime<10>]
}

// CHECK: ssp.instance "canis14_fig2" of "ModuloProblem" [#ssp.InitiationInterval<3>] {
// CHECK:   operator_type @MemPort [#ssp.Latency<1>, #ssp.Limit<1>]
// CHECK:   operator_type @Add [#ssp.Latency<1>]
// CHECK:   operator_type @Implicit [#ssp.Latency<0>]
// CHECK:   %[[op_0:.*]] = operation @load_A(@store_A [#ssp.Distance<1>]) [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<2>]
// CHECK:   %[[op_1:.*]] = operation @load_B() [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<0>]
// CHECK:   %[[op_2:.*]] = operation @add(%[[op_0]], %[[op_1]]) [#ssp.LinkedOperatorType<@Add>, #ssp.StartTime<3>]
// CHECK:   operation @store_A(%[[op_2]]) [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<4>]
// CHECK:   operation @last(@store_A) [#ssp.LinkedOperatorType<@Implicit>, #ssp.StartTime<5>]
// CHECK: }
ssp.instance "canis14_fig2" of "ModuloProblem" [#ssp.InitiationInterval<3>] {
  operator_type @MemPort [#ssp.Latency<1>, #ssp.Limit<1>]
  operator_type @Add [#ssp.Latency<1>]
  operator_type @Implicit [#ssp.Latency<0>]
  %0 = operation @load_A(@store_A [#ssp.Distance<1>]) [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<2>]
  %1 = operation @load_B() [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<0>]
  %2 = operation @add(%0, %1) [#ssp.LinkedOperatorType<@Add>, #ssp.StartTime<3>]
  operation @store_A(%2) [#ssp.LinkedOperatorType<@MemPort>, #ssp.StartTime<4>]
  operation @last(@store_A) [#ssp.LinkedOperatorType<@Implicit>, #ssp.StartTime<5>]
}
