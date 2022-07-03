// RUN: circt-opt %s | circt-opt | FileCheck %s

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

// CHECK: ssp.instance "no properties" of "IncompleteProblem" {
// CHECK:   ssp.operator_type @NoProps
// CHECK:   %[[op_0:.*]] = ssp.operation @Op0()
// CHECK:   ssp.operation @Op1(%[[op_0]])
// CHECK:   ssp.operation @Op2(@Op0)
// CHECK:   ssp.operation @Op3(%[[op_0]], @Op0)
// CHECK: }
ssp.instance "no properties" of "IncompleteProblem" {
  ssp.operator_type @NoProps
  %0 = ssp.operation @Op0()
  ssp.operation @Op1(%0)
  ssp.operation @Op2(@Op0)
  ssp.operation @Op3(%0, @Op0)
}
