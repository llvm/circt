// RUN: circt-opt %s -ssp-roundtrip=verify
// RUN: circt-opt %s -ssp-schedule=scheduler=simplex | FileCheck %s -check-prefixes=CHECK,SIMPLEX
// RUN: %if or-tools %{ circt-opt %s -ssp-schedule=scheduler=cpsat | FileCheck %s -check-prefixes=CHECK,CPSAT %} 

// CHECK-LABEL: full_load
ssp.instance @full_load of "SharedResourcesProblem" {
  library {
    operator_type @L1_3 [latency<3>]
    operator_type @_1 [latency<1>]
  }
  resource {
    resource_type @L1_rsrc [limit<1>]
  }
  graph {
    %0 = operation<@L1_3>() uses[@L1_rsrc] [t<0>]
    %1 = operation<@L1_3>() uses[@L1_rsrc] [t<1>]
    %2 = operation<@L1_3>() uses[@L1_rsrc] [t<2>]
    %3 = operation<@L1_3>() uses[@L1_rsrc] [t<3>]
    %4 = operation<@L1_3>() uses[@L1_rsrc] [t<4>]
    %5 = operation<@_1>(%0, %1, %2, %3, %4) [t<7>]
    // SIMPLEX: @last(%{{.*}}) [t<8>]
    // CPSAT: @last(%{{.*}}) [t<8>]
    operation<@_1> @last(%5) [t<8>]
  }
}

// CHECK-LABEL: partial_load
ssp.instance @partial_load of "SharedResourcesProblem" {
  library {
    operator_type @L3_3 [latency<3>]
    operator_type @_1 [latency<1>]
  }
  resource {
    resource_type @L3_rsrc [limit<3>]
  }
  graph {
    %0 = operation<@L3_3>() uses[@L3_rsrc] [t<0>]
    %1 = operation<@L3_3>() uses[@L3_rsrc] [t<1>]
    %2 = operation<@L3_3>() uses[@L3_rsrc] [t<0>]
    %3 = operation<@L3_3>() uses[@L3_rsrc] [t<2>]
    %4 = operation<@L3_3>() uses[@L3_rsrc] [t<1>]
    %5 = operation<@_1>(%0, %1, %2, %3, %4) [t<10>]
    // SIMPLEX: @last(%{{.*}}) [t<5>]
    // CPSAT: @last(%{{.*}}) [t<5>]
    operation<@_1> @last(%5) [t<11>]
  }
}

// CHECK-LABEL: multiple
ssp.instance @multiple of "SharedResourcesProblem" {
  library {
    operator_type @L3_2 [latency<3>]
    operator_type @L1_1 [latency<1>]
    operator_type @_1 [latency<1>]
  }
  resource {
    resource_type @L3_rsrc [limit<2>]
    resource_type @L1_rsrc [limit<1>]
  }
  graph {
    %0 = operation<@L3_2>() uses[@L3_rsrc] [t<0>]
    %1 = operation<@L3_2>() uses[@L3_rsrc] [t<1>]
    %2 = operation<@L1_1>() uses[@L1_rsrc] [t<0>]
    %3 = operation<@L3_2>() uses[@L3_rsrc] [t<1>]
    %4 = operation<@L1_1>() uses[@L1_rsrc] [t<1>]
    %5 = operation<@_1>(%0, %1, %2, %3, %4) [t<10>]
    // SIMPLEX: @last(%{{.*}}) [t<5>]
    // CPSAT: @last(%{{.*}}) [t<5>]
    operation<@_1> @last(%5) [t<11>]
  }
}

// CHECK-LABEL: if_else
ssp.instance @if_else_exclusive of "SharedResourcesProblem" {
  library {
    operator_type @ThenOperator [latency<2>]
    operator_type @ElseOperator [latency<3>]
    operator_type @_1 [latency<2>]
  }
  resource {
    resource_type @adder_then_br [limit<1>]
    resource_type @mem_port_then_br [limit<1>]
    resource_type @adder_else_br [limit<1>]
    resource_type @mem_port_else_br [limit<1>]
  }
  graph {
    %0 = operation<@ThenOperator> @compute_then_br() uses[@adder_then_br, @mem_port_then_br] [t<0>]
    %1 = operation<@ElseOperator> @compute_else_br() uses[@adder_else_br, @mem_port_else_br] [t<0>]
    %2 = operation<@_1>(%0, %1) [t<4>]
    // SIMPLEX: @last(%{{.*}}) [t<5>]
    // CPSAT: @last(%{{.*}}) [t<5>]
    operation<@_1> @last(%2) [t<7>]
  }
}
