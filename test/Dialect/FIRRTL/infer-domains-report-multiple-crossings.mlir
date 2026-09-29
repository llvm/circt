// RUN: not circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all report-json=%t.json}))' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: FileCheck %s --check-prefix=JSON < %t.json

// ERROR: illegal domain crossing in operation
// JSON: "illegal_crossings": [
// JSON: "domain_type_id": 0
// JSON: "lhs_source_value_id": 1
// JSON: "rhs_source_value_id": 0
// JSON: "domain_type_id": 1
// JSON: "lhs_source_value_id": 3
// JSON: "rhs_source_value_id": 2

firrtl.circuit "MultipleDomainCrossings" {
  firrtl.domain @ClockDomain
  firrtl.domain @ResetDomain

  firrtl.module @MultipleDomainCrossings(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>,
    in %C: !firrtl.domain<@ResetDomain()>,
    in %D: !firrtl.domain<@ResetDomain()>,
    in %a: !firrtl.uint<1> domains [%A, %C],
    out %b: !firrtl.uint<1> domains [%B, %D]
  ) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
