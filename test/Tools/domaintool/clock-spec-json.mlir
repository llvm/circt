// RUN: domaintool --module Foo --domain ClockDomain,A,10 --assign 0 %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: domaintool --module Foo --domain ClockDomain,A,10 --assign 0 --sifive-clock-domain-async=A %s | FileCheck %s --check-prefixes=CHECK,ASYNC
// RUN: domaintool --module Foo --domain ClockDomain,A,10 --assign 0 --sifive-clock-domain-static=A %s | FileCheck %s --check-prefixes=CHECK,STATIC

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
  %0 = om.object @ClockDomain_out(%basepath, %A, %3) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %1 = om.frozenpath_create reference %basepath "Foo>a"
  %2 = om.frozenpath_create reference %basepath "Foo>b"
  %3 = om.list_create %1, %2 : !om.frozenpath
  om.class.fields %0 : !om.class.type<@ClockDomain_out>
}

// CHECK:        {
//
// DEFAULT-NEXT:   "clocks": [
// DEFAULT-NEXT:     {
// DEFAULT-NEXT:       "name_pattern": "A",
// DEFAULT-NEXT:       "define_period": 10,
// DEFAULT-NEXT:       "clock_relationships": []
// DEFAULT-NEXT:     }
// DEFAULT-NEXT:   ],
// ASYNC-NEXT:     "clocks": [],
// STATIC-NEXT:    "clocks": [],
//
// DEFAULT-NEXT:   "static_ports": [],
// ASYNC-NEXT:     "static_ports": [],
// STATIC-NEXT:    "static_ports": [
// STATIC-NEXT:      "a",
// STATIC-NEXT:      "b"
// STATIC-NEXT:    ],
//
// DEFAULT-NEXT:   "asynchronous_ports": [],
// ASYNC-NEXT:     "asynchronous_ports": [
// ASYNC-NEXT:       "a",
// ASYNC-NEXT:       "b"
// ASYNC-NEXT:     ],
// STATIC-NEXT:    "asynchronous_ports": [],
//
// DEFAULT-NEXT:   "synchronous_ports": [
// DEFAULT-NEXT:     {
// DEFAULT-NEXT:       "name_pattern": "A",
// DEFAULT-NEXT:       "port_patterns": [
// DEFAULT-NEXT:         "a"
// DEFAULT-NEXT:         "b"
// DEFAULT-NEXT:       ],
// DEFAULT-NEXT:       "comment": null
// DEFAULT-NEXT:     }
// DEFAULT-NEXT:   ]
// ASYNC-NEXT:     "synchronous_ports": []
// STATIC-NEXT:    "synchronous_ports": []
//
// CHECK-NEXT:   }
