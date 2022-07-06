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
// CHECK:   operator_type @unit [latency<1>]
// CHECK:   operator_type @extr [latency<0>]
// CHECK:   operator_type @add [latency<3>]
// CHECK:   operator_type @mult [latency<6>]
// CHECK:   operator_type @sqrt [latency<10>]
// CHECK:   %[[op_0:.*]] = operation() [opr<@extr>, t<0>]
// CHECK:   %[[op_1:.*]] = operation() [opr<@extr>, t<10>]
// CHECK:   %[[op_2:.*]] = operation(%[[op_0]], %[[op_0]]) [opr<@mult>, t<20>]
// CHECK:   %[[op_3:.*]] = operation(%[[op_1]], %[[op_1]]) [opr<@mult>, t<30>]
// CHECK:   %[[op_4:.*]] = operation(%[[op_2]], %[[op_3]]) [opr<@add>, t<40>]
// CHECK:   %[[op_5:.*]] = operation(%[[op_4]]) [opr<@sqrt>, t<50>]
// CHECK:   operation(%[[op_5]]) [opr<@unit>, t<60>]
// CHECK: }
ssp.instance "arbitrary_latencies" of "Problem" {
  operator_type @unit [latency<1>]
  operator_type @extr [latency<0>]
  operator_type @add [latency<3>]
  operator_type @mult [latency<6>]
  operator_type @sqrt [latency<10>]
  %0 = operation() [opr<@extr>, t<0>]
  %1 = operation() [opr<@extr>, t<10>]
  %2 = operation(%0, %0) [opr<@mult>, t<20>]
  %3 = operation(%1, %1) [opr<@mult>, t<30>]
  %4 = operation(%2, %3) [opr<@add>, t<40>]
  %5 = operation(%4) [opr<@sqrt>, t<50>]
  operation(%5) [opr<@unit>, t<60>]
}

// CHECK: ssp.instance "self_arc" of "CyclicProblem" [II<3>] {
// CHECK:   operator_type @unit [latency<1>]
// CHECK:   operator_type @_3 [latency<3>]
// CHECK:   %[[op_0:.*]] = operation() [opr<@unit>, t<0>]
// CHECK:   %[[op_1:.*]] = operation @self(%[[op_0]], %[[op_0]], @self [dist<1>]) [opr<@_3>, t<1>]
// CHECK:   operation(%[[op_1]]) [opr<@unit>, t<4>]
// CHECK: }
ssp.instance "self_arc" of "CyclicProblem" [II<3>] {
  operator_type @unit [latency<1>]
  operator_type @_3 [latency<3>]
  %0 = operation() [opr<@unit>, t<0>]
  %1 = operation @self(%0, %0, @self [dist<1>]) [opr<@_3>, t<1>]
  operation(%1) [opr<@unit>, t<4>]
}

// CHECK: ssp.instance "multiple_oprs" of "SharedOperatorsProblem" {
// CHECK:   operator_type @slowAdd [latency<3>, limit<2>]
// CHECK:   operator_type @fastAdd [latency<1>, limit<1>]
// CHECK:   operator_type @_0 [latency<0>]
// CHECK:   operator_type @_1 [latency<1>]
// CHECK:   %[[op_0:.*]] = operation() [opr<@slowAdd>, t<0>]
// CHECK:   %[[op_1:.*]] = operation() [opr<@slowAdd>, t<1>]
// CHECK:   %[[op_2:.*]] = operation() [opr<@fastAdd>, t<0>]
// CHECK:   %[[op_3:.*]] = operation() [opr<@slowAdd>, t<1>]
// CHECK:   %[[op_4:.*]] = operation() [opr<@fastAdd>, t<1>]
// CHECK:   %[[op_5:.*]] = operation(%[[op_0]], %[[op_1]], %[[op_2]], %[[op_3]], %[[op_4]]) [opr<@_0>, t<10>]
// CHECK:   operation() [opr<@_1>, t<10>]
// CHECK: }
ssp.instance "multiple_oprs" of "SharedOperatorsProblem" {
  operator_type @slowAdd [latency<3>, limit<2>]
  operator_type @fastAdd [latency<1>, limit<1>]
  operator_type @_0 [latency<0>]
  operator_type @_1 [latency<1>]
  %0 = operation() [opr<@slowAdd>, t<0>]
  %1 = operation() [opr<@slowAdd>, t<1>]
  %2 = operation() [opr<@fastAdd>, t<0>]
  %3 = operation() [opr<@slowAdd>, t<1>]
  %4 = operation() [opr<@fastAdd>, t<1>]
  %5 = operation(%0, %1, %2, %3, %4) [opr<@_0>, t<10>]
  operation() [opr<@_1>, t<10>]
}

// CHECK: ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
// CHECK:   operator_type @MemPort [latency<1>, limit<1>]
// CHECK:   operator_type @Add [latency<1>]
// CHECK:   operator_type @Implicit [latency<0>]
// CHECK:   %[[op_0:.*]] = operation @load_A(@store_A [dist<1>]) [opr<@MemPort>, t<2>]
// CHECK:   %[[op_1:.*]] = operation @load_B() [opr<@MemPort>, t<0>]
// CHECK:   %[[op_2:.*]] = operation @add(%[[op_0]], %[[op_1]]) [opr<@Add>, t<3>]
// CHECK:   operation @store_A(%[[op_2]]) [opr<@MemPort>, t<4>]
// CHECK:   operation @last(@store_A) [opr<@Implicit>, t<5>]
// CHECK: }
ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
  operator_type @MemPort [latency<1>, limit<1>]
  operator_type @Add [latency<1>]
  operator_type @Implicit [latency<0>]
  %0 = operation @load_A(@store_A [dist<1>]) [opr<@MemPort>, t<2>]
  %1 = operation @load_B() [opr<@MemPort>, t<0>]
  %2 = operation @add(%0, %1) [opr<@Add>, t<3>]
  operation @store_A(%2) [opr<@MemPort>, t<4>]
  operation @last(@store_A) [opr<@Implicit>, t<5>]
}
