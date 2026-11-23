// RUN: domaintool --split-input-file --verify-diagnostics --module Foo --domain ClockDomain,A,10 --assign 0 %s

// expected-error @below {{domain class 'ClockDomain' does not match the schema expected by the Clock Spec JSON handler: missing field 'source_out'}}
// expected-error @below {{domain class 'ClockDomain' does not match the schema expected by the Clock Spec JSON handler: missing field 'relationship_out'}}
om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %period_in: !om.integer
)  -> (
  name_out: !om.string,
  period_out: !om.integer
) {
  om.class.fields %name_in, %period_in : !om.string, !om.integer
}

om.class @ClockDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
}

om.class @Foo_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>
)  -> (
  A_out: !om.class.type<@ClockDomain_out>
) {
  %0 = om.object @ClockDomain_out(%basepath, %A, %2) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %1 = om.frozenpath_create reference %basepath "Foo>a"
  %2 = om.list_create %1 : !om.frozenpath
  om.class.fields %0 : !om.class.type<@ClockDomain_out>
}

// -----

// expected-error @below {{domain class 'ClockDomain' does not match the schema expected by the Clock Spec JSON handler: field 'source_out' has type '!om.integer', expected '!om.string'}}
om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %period_in: !om.integer
)  -> (
  name_out: !om.string,
  source_out: !om.integer,
  relationship_out: !om.string
) {
  om.class.fields %name_in, %period_in, %name_in : !om.string, !om.integer, !om.string
}

om.class @ClockDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
}

om.class @Foo_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>
)  -> (
  A_out: !om.class.type<@ClockDomain_out>
) {
  %0 = om.object @ClockDomain_out(%basepath, %A, %2) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %1 = om.frozenpath_create reference %basepath "Foo>a"
  %2 = om.list_create %1 : !om.frozenpath
  om.class.fields %0 : !om.class.type<@ClockDomain_out>
}

// -----

om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %period_in: !om.integer
)  -> (
  name_out: !om.string,
  source_out: !om.string,
  relationship_out: !om.string
) {
  %0 = om.constant "" : !om.string
  om.class.fields %name_in, %0, %0 : !om.string, !om.string, !om.string
}

om.class @ClockDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  // expected-error @below {{domain of class 'ClockDomain' is unknown}}
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
}

// The output domain is unknown, such as with an anonymous domain.
om.class @Foo_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>
)  -> (
  B_out: !om.class.type<@ClockDomain_out>
) {
  %0 = om.unknown : !om.class.type<@ClockDomain>
  %1 = om.object @ClockDomain_out(%basepath, %0, %3) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %2 = om.frozenpath_create reference %basepath "Foo>b"
  %3 = om.list_create %2 : !om.frozenpath
  om.class.fields %1 : !om.class.type<@ClockDomain_out>
}

// -----

om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %period_in: !om.integer
)  -> (
  name_out: !om.string,
  source_out: !om.string,
  relationship_out: !om.string
) {
  %0 = om.constant "" : !om.string
  om.class.fields %name_in, %0, %0 : !om.string, !om.string, !om.string
}

om.class @ClockDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  // expected-error @below {{domain of class 'ClockDomain' has no known string value for field 'name_out'}}
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
}

// The output domain is created from an unknown input.
om.class @Foo_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>,
  %name: !om.string
)  -> (
  B_out: !om.class.type<@ClockDomain_out>
) {
  %0 = om.constant #om.integer<10 : si64> : !om.integer
  %1 = om.object @ClockDomain(%basepath, %name, %0) : (
    !om.frozenbasepath,
    !om.string,
    !om.integer
  ) -> !om.class.type<@ClockDomain>
  %2 = om.object @ClockDomain_out(%basepath, %1, %4) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %3 = om.frozenpath_create reference %basepath "Foo>b"
  %4 = om.list_create %3 : !om.frozenpath
  om.class.fields %2 : !om.class.type<@ClockDomain_out>
}

// -----

om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %period_in: !om.integer
)  -> (
  name_out: !om.string,
  source_out: !om.string,
  relationship_out: !om.string
) {
  %0 = om.constant "" : !om.string
  om.class.fields %name_in, %0, %0 : !om.string, !om.string, !om.string
}

om.class @ClockDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
}

om.class @Foo_Class(
  %basepath: !om.frozenbasepath,
  %A: !om.class.type<@ClockDomain>
)  -> (
  A_out: !om.class.type<@ClockDomain_out>
) {
  %0 = om.object @ClockDomain_out(%basepath, %A, %2) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  // expected-error @below {{association is a path to a deleted or unknown target}}
  %1 = om.frozenpath_empty
  %2 = om.list_create %1 : !om.frozenpath
  om.class.fields %0 : !om.class.type<@ClockDomain_out>
}
