// RUN: domaintool --module Foo --domain ClockDomain,A,A,synchronous --assign 0 %s | FileCheck %s

om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %source_in: !om.string,
  %relationship_in: !om.string
)  -> (
  name_out: !om.string,
  source_out: !om.string,
  relationship_out: !om.string
) {
  om.class.fields %name_in, %source_in, %relationship_in : !om.string, !om.string, !om.string
}

om.class @ClockDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>,
  %clockGates_registry_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>,
  clockGates_registry_out: !om.list<!om.frozenpath>
) {
  om.class.fields %domainInfo_in, %associations_in, %clockGates_registry_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>, !om.list<!om.frozenpath>
}

om.class @Foo_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>
)  -> (
  A_out: !om.class.type<@ClockDomain_out>
) {
  %assoc0 = om.frozenpath_create reference %basepath "Foo>a"
  %assocs = om.list_create %assoc0 : !om.frozenpath
  %cg0 = om.frozenpath_create reference %basepath "Foo/bar1:Bar>clockGate"
  %cg1 = om.frozenpath_create reference %basepath "Foo/bar2:Bar>clockGate"
  %gates = om.list_create %cg0, %cg1 : !om.frozenpath
  %0 = om.object @ClockDomain_out(%basepath, %A, %assocs, %gates) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  om.class.fields %0 : !om.class.type<@ClockDomain_out>
}

// CHECK:      "name_pattern": "A"
// CHECK:      "clock_gates": [
// CHECK-NEXT:   "OMReferenceTarget:~Foo|Foo/bar1:Bar>clockGate",
// CHECK-NEXT:   "OMReferenceTarget:~Foo|Foo/bar2:Bar>clockGate"
// CHECK-NEXT: ]
