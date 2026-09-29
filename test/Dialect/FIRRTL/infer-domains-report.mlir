// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all report-json=%t.json}))' %s -o /dev/null && FileCheck %s --check-prefix=JSON < %t.json
// RUN: firtool %s --format=mlir -domain-mode=infer-all --domain-report-json=%t.firtool.json -o /dev/null && FileCheck %s --check-prefix=FIRTOOL-JSON < %t.firtool.json

// JSON-DAG: "format": "circt-domain-inference"
// JSON-DAG: "version": 3
// JSON-DAG: "complete": true
// JSON-DAG: "illegal_crossings": []
// JSON-DAG: "name": "ClockDomain"
// JSON-DAG: "operation_kinds": [
// JSON-DAG: "firrtl.matchingconnect"
// JSON-DAG: "provenance_edge_kinds": [
// JSON-DAG: "constraint"
// JSON-DAG: "provenance_edge_fields": [
// JSON-DAG: "operation_kind_id"
// JSON-DAG: "instance_binding_direction": "parent_instance_to_module_template"
// JSON-DAG: "file": "{{.*}}infer-domains-report.mlir"
// JSON-DAG: "domain_value_id": {{[0-9]+}}
// JSON-DAG: "name": "middle"
// JSON-DAG: "name": "first"
// JSON-DAG: "name": "second"
// JSON-DAG: "effective_domain_value_id": {{[0-9]+}}
// JSON-DAG: "effective_domain_value_name": "first.childClock"
// JSON-DAG: "effective_domain_value_name": "second.childClock"
// FIRTOOL-JSON-DAG: "format": "circt-domain-inference"
// FIRTOOL-JSON-DAG: "version": 3
// FIRTOOL-JSON-DAG: "illegal_crossings": []
// FIRTOOL-JSON-DAG: "effective_domain_value_name": "first.childClock"
// FIRTOOL-JSON-DAG: "effective_domain_value_name": "second.childClock"

firrtl.circuit "DomainReport" {
  firrtl.domain @ClockDomain

  firrtl.module @Child(
    in %childClock: !firrtl.domain<@ClockDomain()>,
    in %childIn: !firrtl.uint<1> domains [%childClock],
    out %childOut: !firrtl.uint<1>
  ) {
    %middle = firrtl.wire : !firrtl.uint<1>
    firrtl.matchingconnect %middle, %childIn : !firrtl.uint<1>
    firrtl.matchingconnect %childOut, %middle : !firrtl.uint<1>
  }

  firrtl.module @DomainReport(
    in %clockA: !firrtl.domain<@ClockDomain()>,
    in %clockB: !firrtl.domain<@ClockDomain()>,
    in %inputA: !firrtl.uint<1> domains [%clockA],
    in %inputB: !firrtl.uint<1> domains [%clockB]
  ) {
    %firstClock, %firstIn, %firstOut = firrtl.instance first @Child(
      in childClock: !firrtl.domain<@ClockDomain()>,
      in childIn: !firrtl.uint<1> domains [childClock],
      out childOut: !firrtl.uint<1>
    )
    %secondClock, %secondIn, %secondOut = firrtl.instance second @Child(
      in childClock: !firrtl.domain<@ClockDomain()>,
      in childIn: !firrtl.uint<1> domains [childClock],
      out childOut: !firrtl.uint<1>
    )
    firrtl.domain.define %firstClock, %clockA : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %secondClock, %clockB : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %firstIn, %inputA : !firrtl.uint<1>
    firrtl.matchingconnect %secondIn, %inputB : !firrtl.uint<1>
  }
}
