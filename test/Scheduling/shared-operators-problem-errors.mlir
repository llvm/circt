// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Operator type 'limited' using limited resource 'limited_rsrc' has zero latency.}}
ssp.instance @limited_but_zero_latency of "SharedOperatorsProblem" {
  library {
    operator_type @limited [latency<0>]
  }
  resource {
    resource_type @limited_rsrc [limit<1>]
  }
  graph {
    operation<@limited> uses<@limited_rsrc>() [t<0>]
  }
}

// -----

// expected-error@+1 {{Resource type 'limited_rsrc' is oversubscribed}}
ssp.instance @oversubscribed of "SharedOperatorsProblem" {
  library {
    operator_type @limited [latency<1>]
  }
  resource {
    resource_type @limited_rsrc [limit<2>]
  }
  graph {
    operation<@limited> uses<@limited_rsrc>() [t<0>]
    operation<@limited> uses<@limited_rsrc>() [t<0>]
    operation<@limited> uses<@limited_rsrc>() [t<0>]
  }
}
