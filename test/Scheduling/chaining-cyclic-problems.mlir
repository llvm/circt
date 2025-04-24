// RUN: circt-opt %s -ssp-roundtrip=verify
// RUN: circt-opt %s -ssp-schedule="scheduler=simplex options=cycle-time=5.0" | FileCheck %s -check-prefixes=CHECK,SIMPLEX

// test from cyclic-problems.mlir

// CHECK-LABEL: cyclic
// SIMPLEX-SAME: [II<2>]
ssp.instance @cyclic of "ChainingCyclicProblem" [II<2>] {
  library {
    operator_type @_0 [latency<0>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_1 [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_2 [latency<2>, incDelay<0.0>, outDelay<0.0>]
  }
  resource {
  }
  graph {
    %0 = operation<@_1>() [t<0>, z<0.000000e+00 : f32>]
    %1 = operation<@_0>(@op4 [dist<1>]) [t<1>, z<0.000000e+00 : f32>]
    %2 = operation<@_2>(@op4 [dist<2>]) [t<0>, z<0.000000e+00 : f32>]
    %3 = operation<@_1>(%1, %2) [t<2>, z<0.000000e+00 : f32>]
    %4 = operation<@_1> @op4(%2, %0) [t<2>, z<0.000000e+00 : f32>]
    operation<@_1> @last(%4) [t<3>, z<0.000000e+00 : f32>]
    // SIMPLEX: operation<@_1> @last(%{{.*}}) [t<3>, z<0.000000e+00 : f32>]
  }
}

// CHECK-LABEL: mobility
// SIMPLEX-SAME: [II<3>]
ssp.instance @mobility of "ChainingCyclicProblem" [II<3>] {
  library {
    operator_type @_1 [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_4 [latency<4>, incDelay<0.0>, outDelay<0.0>]
  }
  resource {
  }
  graph {
    %0 = operation<@_1>() [t<0>, z<0.000000e+00 : f32>]
    %1 = operation<@_4>(%0) [t<1>, z<0.000000e+00 : f32>]
    %2 = operation<@_1>(%0, @op5 [dist<1>]) [t<4>, z<0.000000e+00 : f32>]
    %3 = operation<@_1>(%1, %2) [t<5>, z<0.000000e+00 : f32>]
    %4 = operation<@_4>(%3) [t<6>, z<0.000000e+00 : f32>]
    %5 = operation<@_1> @op5(%3) [t<6>, z<0.000000e+00 : f32>]
    operation<@_1> @last(%4, %5) [t<10>, z<0.000000e+00 : f32>]
    // SIMPLEX: @last(%{{.*}}, %{{.*}}) [t<10>, z<0.000000e+00 : f32>]
  }
}

// CHECK-LABEL: interleaved_cycles
// SIMPLEX-SAME: [II<4>]
ssp.instance @interleaved_cycles of "ChainingCyclicProblem" [II<4>] {
  library {
    operator_type @_1 [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_10 [latency<10>, incDelay<0.0>, outDelay<0.0>]
  }
  resource {
  }
  graph {
    %0 = operation<@_1>() [t<0>, z<0.000000e+00 : f32>]
    %1 = operation<@_10>(%0) [t<1>, z<0.000000e+00 : f32>]
    %2 = operation<@_1>(%0, @op6 [dist<2>]) [t<10>, z<0.000000e+00 : f32>]
    %3 = operation<@_1>(%1, %2) [t<11>, z<0.000000e+00 : f32>]
    %4 = operation<@_10>(%3) [t<12>, z<0.000000e+00 : f32>]
    %5 = operation<@_1>(%3, @op9 [dist<2>]) [t<16>, z<0.000000e+00 : f32>]
    %6 = operation<@_1> @op6(%5) [t<17>, z<0.000000e+00 : f32>]
    %7 = operation<@_1>(%4, %6) [t<22>, z<0.000000e+00 : f32>]
    %8 = operation<@_10>(%7) [t<23>, z<0.000000e+00 : f32>]
    %9 = operation<@_1> @op9(%7) [t<23>, z<0.000000e+00 : f32>]
    operation<@_1> @last(%8, %9) [t<33>, z<0.000000e+00 : f32>]
    // SIMPLEX: @last(%{{.*}}, %{{.*}}) [t<33>, z<0.000000e+00 : f32>]
  }
}

// CHECK-LABEL: self_arc
// SIMPLEX-SAME: [II<3>]
ssp.instance @self_arc of "ChainingCyclicProblem" [II<3>] {
  library {
    operator_type @_1 [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_3 [latency<3>, incDelay<0.0>, outDelay<0.0>]
  }
  resource {
  }
  graph {
    %0 = operation<@_1>() [t<0>, z<0.000000e+00 : f32>]
    %1 = operation<@_3> @op1(%0, @op1 [dist<1>]) [t<1>, z<0.000000e+00 : f32>]
    %2 = operation<@_1> @last(%1) [t<4>, z<0.000000e+00 : f32>]
    // SIMPLEX: operation<@_1> @last(%{{.*}}) [t<4>, z<0.000000e+00 : f32>]
  }
}

// test from chaining-problems.mlir

// CHECK-LABEL: adder_chain
ssp.instance @adder_chain of "ChainingCyclicProblem" [II<1>] {
  library {
    operator_type @_0 [latency<0>, incDelay<2.34>, outDelay<2.34>]
    operator_type @_1 [latency<1>, incDelay<0.0>, outDelay<0.0>]
  }
  resource {
  }
  graph {
    %0 = operation<@_0>() [t<0>, z<0.0>]
    %1 = operation<@_0>(%0) [t<0>, z<2.34>]
    %2 = operation<@_0>(%1) [t<0>, z<4.68>]
    %3 = operation<@_0>(%2) [t<1>, z<0.0>]
    %4 = operation<@_0>(%3) [t<1>, z<2.34>]
    operation<@_1> @last(%4) [t<2>, z<0.0>]
    // SIMPLEX: @last(%{{.*}}) [t<2>,
  }
}

// CHECK-LABEL: multi_cycle
ssp.instance @multi_cycle of "ChainingCyclicProblem" [II<1>] {
  library {
    operator_type @_0 [latency<0>, incDelay<2.34>, outDelay<2.34>]
    operator_type @_1 [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_3 [latency<3>, incDelay<2.5>, outDelay<3.75>]
  }
  resource {
  }
  graph {
    %0 = operation<@_0>() [t<0>, z<0.0>]
    %1 = operation<@_0>(%0) [t<0>, z<2.34>]
    %2 = operation<@_3>(%1, %0) [t<1>, z<0.00>]
    %3 = operation<@_0>(%2, %1) [t<5>, z<0.00>]
    %4 = operation<@_0>(%3, %2) [t<5>, z<2.34>]
    operation<@_1> @last(%4) [t<5>, z<4.68>]
    // SIMPLEX: @last(%{{.*}}) [t<5>,
  }
}

// CHECK-LABEL: mco_outgoing_delays
ssp.instance @mco_outgoing_delays of "ChainingCyclicProblem" [II<1>] {
  library {
    operator_type @_2 [latency<2>, incDelay<0.1>, outDelay<0.1>]
    operator_type @_3 [latency<3>, incDelay<5.0>, outDelay<0.1>]
  }
  resource {
  }
  // SIMPLEX: graph
  graph {
    // SIMPLEX-NEXT: [t<0>, z<0.000000e+00 : f32>]
    %0 = operation<@_2>() [t<0>, z<0.0>]

    // Next op cannot start in cycle 2 due to %0's outgoing delay: 0.1+5.0 > 5.0.
    // SIMPLEX-NEXT: [t<3>, z<0.000000e+00 : f32>]
    %1 = operation<@_3>(%0) [t<3>, z<0.0>]

    // SIMPLEX-NEXT: [t<6>, z<1.000000e-01 : f32>]
    %2 = operation<@_2>(%1) [t<6>, z<0.1>]

    // Next op should have SITC=0.1 (not: 0.2), because we only consider %2's outgoing delay.
    // SIMPLEX-NEXT: [t<8>, z<1.000000e-01 : f32>]
    operation<@_2> @last(%2) [t<8>, z<0.1>]
  }
}
// custom tests

// CHECK-LABEL: chaining_and_cyclic
// SIMPLEX-SAME: [II<2>]
ssp.instance @chaining_and_cyclic of "ChainingCyclicProblem" [II<2>] {
  library {
    operator_type @_0 [latency<0>, incDelay<1.0>, outDelay<1.0>]
    operator_type @_1 [latency<0>, incDelay<3.0>, outDelay<3.0>]
    operator_type @_2 [latency<1>, incDelay<0.0>, outDelay<0.75>]
    operator_type @_3 [latency<0>, incDelay<3.5>, outDelay<3.5>]
    operator_type @_4 [latency<1>, incDelay<1.2>, outDelay<1.2>]
    operator_type @_5 [latency<0>, incDelay<3.8>, outDelay<3.8>]
  }
  resource {
  }
  graph {
    %0 = operation<@_0>(@op2 [dist<1>]) [t<0>, z<0.0>]
    %1 = operation<@_1>(%0) [t<0>, z<1.0>]
    %2 = operation<@_2> @op2(%0, @op4 [dist<1>]) [t<0>, z<1.0>]
    %3 = operation<@_3>(%0) [t<0>, z<1.0>]
    %4 = operation<@_4> @op4(%1, %3) [t<1>, z<0.0>]
    // SIMPLEX: @last(%{{.*}}) [t<2>, z<1.200000e+00 : f32>]
    %5 = operation<@_5> @last(%4, %2) [t<2>, z<1.2>]
  }
}

// CHECK-LABEL: backedge_delay_propagation
// SIMPLEX-SAME: [II<1>]
ssp.instance @backedge_delay_propagation of "ChainingCyclicProblem" [II<1>] {
  library {
    operator_type @_0 [latency<0>, incDelay<1.0>, outDelay<1.0>]
  }
  resource {
  }
  graph {
    // SIMPLEX: @last(@last [dist<1>]) [t<0>, z<0.000000e+00 : f32>]
    %0 = operation<@_0> @last(@last [dist<1>]) [t<0>, z<0.000000e+00 : f32>]
  }
}
