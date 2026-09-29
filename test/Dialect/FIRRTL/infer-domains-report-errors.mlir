// RUN: not circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all report-json=%t.json}))' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: FileCheck %s --check-prefix=JSON < %t.json
// RUN: not firtool %s --format=mlir -domain-mode=infer-all --domain-report-json=%t.firtool.json -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: FileCheck %s --check-prefix=JSON < %t.firtool.json

// ERROR: illegal domain crossing in operation
// JSON-DAG: "format": "circt-domain-inference"
// JSON-DAG: "version": 3
// JSON-DAG: "complete": false
// JSON-DAG: "association"
// JSON-DAG: "domain_type_id": 0
// JSON-DAG: "illegal_crossings": [
// JSON-DAG: "lhs_domain_value_id": 1
// JSON-DAG: "lhs_source_value_id": 1
// JSON-DAG: "lhs_value_id": 3
// JSON-DAG: "rhs_domain_value_id": 0
// JSON-DAG: "rhs_source_value_id": 0
// JSON-DAG: "rhs_value_id": 2

firrtl.circuit "DomainReportError" {
  firrtl.domain @ClockDomain

  firrtl.module @DomainReportError(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
