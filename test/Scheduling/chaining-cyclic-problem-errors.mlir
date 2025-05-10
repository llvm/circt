// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Def-use dependence cannot have non-zero distance.}}
ssp.instance @defUse_distance of "ChainingCyclicProblem" {
  library {
    operator_type @_0 [latency<0>, incDelay<1.0>, outDelay<1.0>]
  }
  resource {
  }
  graph {
    %0 = ssp.operation<@_0> ()
    %1 = ssp.operation<@_0> (%0 [dist<3>])
  }
}
