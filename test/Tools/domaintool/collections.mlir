// RUN: domaintool --module Foo --domain ClockDomain,A,42,A,synchronous --domain ClockDomain,B,17,A,rational --assign 0 --assign 1 %s | FileCheck %s
// RUN: not domaintool --module Foo --domain ClockDomain,A,42,A --assign 0 %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not domaintool --module Foo --domain ClockDomain,A,42,A,synchronous,extra --assign 0 %s 2>&1 | FileCheck %s --check-prefix=EXCESS

// Collection fields do not consume command-line arguments, even when they are
// interspersed with scalar fields or have element types other than paths.
om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %tags_in: !om.collection<!om.string>,
  %name_in: !om.string,
  %clockGates_in: !om.collection<!om.frozenpath>,
  %period_in: !om.integer,
  %source_in: !om.string,
  %relationship_in: !om.string,
  %flags_in: !om.collection<!om.integer>
) -> (
  tags_out: !om.collection<!om.string>,
  name_out: !om.string,
  clockGates_out: !om.collection<!om.frozenpath>,
  period_out: !om.integer,
  source_out: !om.string,
  relationship_out: !om.string,
  flags_out: !om.collection<!om.integer>
) {
  om.class.fields %tags_in, %name_in, %clockGates_in, %period_in, %source_in, %relationship_in, %flags_in : !om.collection<!om.string>, !om.string, !om.collection<!om.frozenpath>, !om.integer, !om.string, !om.string, !om.collection<!om.integer>
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

om.class @Foo_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>,
  %B: !om.class.type<@ClockDomain>
) -> (
  A_out: !om.class.type<@ClockDomain_out>,
  B_out: !om.class.type<@ClockDomain_out>
) {
  %gates = om.object.field %A["clockGates_out"] : (!om.class.type<@ClockDomain>) -> !om.collection<!om.frozenpath>
  %g1base = om.frozenbasepath_create %basepath "Foo/g1"
  %g1 = om.frozenpath_create reference %g1base "Gate>clock"
  %g0base = om.frozenbasepath_create %basepath "Foo/g0"
  %g0 = om.frozenpath_create reference %g0base "Gate>clock"
  // Sort unordered contents for stable output, but do not deduplicate them.
  om.collection_push %gates, %g1 : !om.collection<!om.frozenpath>, !om.frozenpath
  om.collection_push %gates, %g0 : !om.collection<!om.frozenpath>, !om.frozenpath
  om.collection_push %gates, %g0 : !om.collection<!om.frozenpath>, !om.frozenpath
  %port = om.frozenpath_create reference %basepath "Foo>clock"
  %associations = om.list_create %port : !om.frozenpath
  %a = om.elaborated_object @ClockDomain_out(%A, %associations) : (!om.class.type<@ClockDomain>, !om.list<!om.frozenpath>) -> !om.class.type<@ClockDomain_out>
  %b = om.elaborated_object @ClockDomain_out(%B, %associations) : (!om.class.type<@ClockDomain>, !om.list<!om.frozenpath>) -> !om.class.type<@ClockDomain_out>
  om.class.fields %a, %b : !om.class.type<@ClockDomain_out>, !om.class.type<@ClockDomain_out>
}

// CHECK:      "name_pattern": "A",
// CHECK-NEXT: "define_period": "A_PERIOD",
// CHECK-NEXT: "clock_gates": [
// CHECK-NEXT:   "OMReferenceTarget:~Foo|Foo/g0:Gate>clock",
// CHECK-NEXT:   "OMReferenceTarget:~Foo|Foo/g0:Gate>clock",
// CHECK-NEXT:   "OMReferenceTarget:~Foo|Foo/g1:Gate>clock"
// CHECK-NEXT: ],
// CHECK-NEXT: "clock_relationships": []
// CHECK:      "name_pattern": "B",
// CHECK-NEXT: "define_period": "B_PERIOD",
// CHECK-NEXT: "clock_gates": [],
// CHECK-NEXT: "clock_relationships": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "name_pattern": "A",
// CHECK-NEXT:     "relationship": "sync"
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK:      "port_patterns": [
// CHECK-NEXT:   "clock"
// CHECK-NEXT: ]

// MISSING: actual parameter list length (7) does not match formal parameter list length (8)
// EXCESS: actual parameter list length (9) does not match formal parameter list length (8)
