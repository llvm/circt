// RUN: not domaintool --module Scalar --domain ClockDomain,A,42,A,synchronous --assign 0 %s 2>&1 | FileCheck %s --check-prefix=SCALAR
// RUN: not domaintool --module Elements --domain ClockDomain,A,42,A,synchronous --assign 0 %s 2>&1 | FileCheck %s --check-prefix=ELEMENTS
// RUN: not domaintool --module Scalar --domain ClockDomain,A,42,17,synchronous --assign 0 %s 2>&1 | FileCheck %s --check-prefix=SOURCE

// Structurally invalid domain fields should produce diagnostics, not assertions.
// Any-typed fields let us construct these values at the evaluator API boundary.
om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %clockGates_in: !om.any,
  %source_in: !om.any,
  %relationship_in: !om.string
) -> (
  name_out: !om.string,
  clockGates_out: !om.any,
  source_out: !om.any,
  relationship_out: !om.string
) {
  om.class.fields %name_in, %clockGates_in, %source_in, %relationship_in : !om.string, !om.any, !om.any, !om.string
}

om.class @ClockDomain_out(
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>
) -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
}

om.class @Scalar_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>
) -> (A_out: !om.class.type<@ClockDomain_out>) {
  %empty = om.list_create : !om.frozenpath
  %out = om.elaborated_object @ClockDomain_out(%A, %empty) : (!om.class.type<@ClockDomain>, !om.list<!om.frozenpath>) -> !om.class.type<@ClockDomain_out>
  om.class.fields %out : !om.class.type<@ClockDomain_out>
}

om.class @Elements_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>
) -> (A_out: !om.class.type<@ClockDomain_out>) {
  %name = om.object.field %A["name_out"] : (!om.class.type<@ClockDomain>) -> !om.string
  %source = om.object.field %A["source_out"] : (!om.class.type<@ClockDomain>) -> !om.any
  %relationship = om.object.field %A["relationship_out"] : (!om.class.type<@ClockDomain>) -> !om.string
  %integer = om.constant #om.integer<42 : i64> : !om.integer
  %gates = om.list_create %integer : !om.integer
  %anyGates = om.any_cast %gates : (!om.list<!om.integer>) -> !om.any
  %domain = om.object @ClockDomain(%basepath, %name, %anyGates, %source, %relationship) : (!om.frozenbasepath, !om.string, !om.any, !om.any, !om.string) -> !om.class.type<@ClockDomain>
  %empty = om.list_create : !om.frozenpath
  %out = om.elaborated_object @ClockDomain_out(%domain, %empty) : (!om.class.type<@ClockDomain>, !om.list<!om.frozenpath>) -> !om.class.type<@ClockDomain_out>
  om.class.fields %out : !om.class.type<@ClockDomain_out>
}

// SCALAR: error: expected clockGates_out to be a known list of paths
// ELEMENTS: error: expected clockGates_out element to be a known path, but got '!om.integer'
// SOURCE: error: expected clock domain field source_out to be a known string
