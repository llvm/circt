// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s -test-ssp-roundtrip | circt-opt | FileCheck %s --check-prefix=INFRA

// 1) tests the plain parser/printer roundtrip.
// CHECK: ssp.library @Lib {
// CHECK:   operator_type @MemPort [latency<1>, limit<1>]
// CHECK: }
// CHECK: module @SomeModule {
// CHECK:   ssp.library @Lib {
// CHECK:     operator_type @Add [latency<1>]
// CHECK:   }
// CHECK: }
// CHECK: ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
// CHECK:   library {
// CHECK:     operator_type @Implicit [latency<0>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<@Lib::@MemPort> @load_A(@store_A [dist<1>]) [t<2>]
// CHECK:     %[[op_1:.*]] = operation<@Lib::@MemPort> @load_B() [t<0>]
// CHECK:     %[[op_2:.*]] = operation<@SomeModule::@Lib::@Add> @add(%[[op_0]], %[[op_1]]) [t<3>]
// CHECK:     operation<@Lib::@MemPort> @store_A(%[[op_2]]) [t<4>]
// CHECK:     operation<@Implicit> @last(@store_A) [t<5>]
// CHECK:   }
// CHECK: }

// 2) Import/export via the scheduling infra (i.e. populates a `Problem` instance and reconstructs the SSP IR from it.)
//    Operator types from standalone libraries are appended to the instance's internal library.
// INFRA: ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
// INFRA:   library {
// INFRA:     operator_type @Implicit [latency<0>]
// INFRA:     operator_type @MemPort [latency<1>, limit<1>]
// INFRA:     operator_type @Add [latency<1>]
// INFRA:   }
// INFRA:   graph {
// INFRA:     %[[op_0:.*]] = operation<@MemPort> @load_A(@store_A [dist<1>]) [t<2>]
// INFRA:     %[[op_1:.*]] = operation<@MemPort> @load_B() [t<0>]
// INFRA:     %[[op_2:.*]] = operation<@Add> @add(%[[op_0]], %[[op_1]]) [t<3>]
// INFRA:     operation<@MemPort> @store_A(%[[op_2]]) [t<4>]
// INFRA:     operation<@Implicit> @last(@store_A) [t<5>]
// INFRA:   }
// INFRA: }

ssp.library @Lib {
  operator_type @MemPort [latency<1>, limit<1>]
}
module @SomeModule {
  ssp.library @Lib {
    operator_type @Add [latency<1>]
  }
}
ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
  library {
    operator_type @Implicit [latency<0>]
  }
  graph {
    %0 = operation<@Lib::@MemPort> @load_A(@store_A [dist<1>]) [t<2>]
    %1 = operation<> @load_B() [t<0>, opr<@Lib::@MemPort>]
    %2 = operation<@SomeModule::@Lib::@Add> @add(%0, %1) [t<3>]
    operation<@Lib::@MemPort> @store_A(%2) [t<4>]
    operation<@Implicit> @last(@store_A) [t<5>]
  }
}
