// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Resource type 'limited_rsrc' is oversubscribed}}
ssp.instance @oversubscribed of "ModuloProblem" [II<2>] {
  library {
    operator_type @limited [latency<1>]
  }
  resource {
    resource_type @limited_rsrc [limit<2>]
  }
  graph {
    operation<@limited> uses[@limited_rsrc]() [t<1>]
    operation<@limited> uses[@limited_rsrc]() [t<3>]
    operation<@limited> uses[@limited_rsrc]() [t<5>]
  }
}
