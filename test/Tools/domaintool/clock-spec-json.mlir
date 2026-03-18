// RUN: domaintool --module Foo --domain ClockDomain,A,"" --domain ClockDomain,B,"" --assign 0 --assign 1 %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: domaintool --module Foo --domain ClockDomain,A,"" --domain ClockDomain,B,"" --assign 0 --assign 1 --sifive-clock-domain-async=A %s | FileCheck %s --check-prefixes=CHECK,ASYNC
// RUN: domaintool --module Foo --domain ClockDomain,A,"" --domain ClockDomain,B,"" --assign 0 --assign 1 --sifive-clock-domain-static=A %s | FileCheck %s --check-prefixes=CHECK,STATIC
// RUN: domaintool --module Foo --domain ClockDomain,A,"" --domain ClockDomain,B,A --assign 0 --assign 1 %s | FileCheck %s --check-prefixes=CHECK,SYNC

om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %synchronousTo_in: !om.string
)  -> (
  name_out: !om.string,
  synchronousTo_out: !om.string
) {
  om.class.fields %name_in, %synchronousTo_in : !om.string, !om.string
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
  %A: !om.class.type<@ClockDomain>,
  %base_address: !om.integer,
  %B: !om.class.type<@ClockDomain>
)  -> (
  A_out: !om.class.type<@ClockDomain_out>,
  effective_address: !om.integer,
  B_out: !om.class.type<@ClockDomain_out>
) {
  %0 = om.object @ClockDomain_out(%basepath, %A, %3) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %1 = om.frozenpath_create reference %basepath "Foo>a"
  %2 = om.frozenpath_create reference %basepath "Foo>b"
  %3 = om.list_create %1, %2 : !om.frozenpath
  %4 = om.constant #om.integer<1 : si4> : !om.integer
  %5 = om.integer.add %base_address, %4 : !om.integer
  %6 = om.object @ClockDomain_out(%basepath, %B, %7) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %7 = om.list_create %1, %2 : !om.frozenpath
  om.class.fields %0, %5, %6 : !om.class.type<@ClockDomain_out>, !om.integer, !om.class.type<@ClockDomain_out>
}

// CHECK:        {
//
// DEFAULT-NEXT:   "clocks": [
// DEFAULT-NEXT:     {
// DEFAULT-NEXT:       "name_pattern": "A",
// DEFAULT-NEXT:       "define_period": "A_PERIOD",
// DEFAULT-NEXT:       "clock_relationships": []
// DEFAULT-NEXT:     },
// DEFAULT-NEXT:     {
// DEFAULT-NEXT:       "name_pattern": "B",
// DEFAULT-NEXT:       "define_period": "B_PERIOD",
// DEFAULT-NEXT:       "clock_relationships": []
// DEFAULT-NEXT:     }
// DEFAULT-NEXT:   ],
// ASYNC-NEXT:     "clocks": [
// ASYNC-NEXT:       {
// ASYNC-NEXT:         "name_pattern": "B",
// ASYNC-NEXT:         "define_period": "B_PERIOD",
// ASYNC-NEXT:         "clock_relationships": []
// ASYNC-NEXT:       }
// ASYNC-NEXT:     ],
// STATIC-NEXT:    "clocks": [
// STATIC-NEXT:      {
// STATIC-NEXT:        "name_pattern": "B",
// STATIC-NEXT:        "define_period": "B_PERIOD",
// STATIC-NEXT:        "clock_relationships": []
// STATIC-NEXT:      }
// STATIC-NEXT:    ],
// SYNC-NEXT:      "clocks": [
// SYNC-NEXT:        {
// SYNC-NEXT:          "name_pattern": "A",
// SYNC-NEXT:          "define_period": "A_PERIOD",
// SYNC-NEXT:          "clock_relationships": []
// SYNC-NEXT:        },
// SYNC-NEXT:        {
// SYNC-NEXT:          "name_pattern": "B",
// SYNC-NEXT:          "define_period": "B_PERIOD",
// SYNC-NEXT:          "clock_relationships": [
// SYNC-NEXT:            {
// SYNC-NEXT:              "name_pattern": "A",
// SYNC-NEXT:              "relationship": "sync"
// SYNC-NEXT:            }
// SYNC-NEXT:          ]
// SYNC-NEXT:        }
// SYNC-NEXT:      ],
//
// DEFAULT-NEXT:   "static_ports": [],
// ASYNC-NEXT:     "static_ports": [],
// STATIC-NEXT:    "static_ports": [
// STATIC-NEXT:      "a",
// STATIC-NEXT:      "b"
// STATIC-NEXT:    ],
// SYNC-NEXT:      "static_ports": [],
//
// DEFAULT-NEXT:   "asynchronous_ports": [],
// ASYNC-NEXT:     "asynchronous_ports": [
// ASYNC-NEXT:       "a",
// ASYNC-NEXT:       "b"
// ASYNC-NEXT:     ],
// STATIC-NEXT:    "asynchronous_ports": [],
// SYNC-NEXT:      "asynchronous_ports": [],
//
// DEFAULT-NEXT:   "synchronous_ports": [
// DEFAULT-NEXT:     {
// DEFAULT-NEXT:       "name_pattern": "A",
// DEFAULT-NEXT:       "port_patterns": [
// DEFAULT-NEXT:         "a",
// DEFAULT-NEXT:         "b"
// DEFAULT-NEXT:       ],
// DEFAULT-NEXT:       "comment": null
// DEFAULT-NEXT:     },
// DEFAULT-NEXT:     {
// DEFAULT-NEXT:       "name_pattern": "B",
// DEFAULT-NEXT:       "port_patterns": [
// DEFAULT-NEXT:         "a",
// DEFAULT-NEXT:         "b"
// DEFAULT-NEXT:       ],
// DEFAULT-NEXT:       "comment": null
// DEFAULT-NEXT:     }
// DEFAULT-NEXT:   ]
// ASYNC-NEXT:     "synchronous_ports": [
// ASYNC-NEXT:       {
// ASYNC-NEXT:         "name_pattern": "B",
// ASYNC-NEXT:         "port_patterns": [
// ASYNC-NEXT:           "a",
// ASYNC-NEXT:           "b"
// ASYNC-NEXT:         ],
// ASYNC-NEXT:         "comment": null
// ASYNC-NEXT:       }
// ASYNC-NEXT:     ]
// STATIC-NEXT:    "synchronous_ports": [
// STATIC-NEXT:      {
// STATIC-NEXT:        "name_pattern": "B",
// STATIC-NEXT:        "port_patterns": [
// STATIC-NEXT:          "a",
// STATIC-NEXT:          "b"
// STATIC-NEXT:        ],
// STATIC-NEXT:        "comment": null
// STATIC-NEXT:      }
// STATIC-NEXT:    ]
// SYNC-NEXT:      "synchronous_ports": [
// SYNC-NEXT:        {
// SYNC-NEXT:          "name_pattern": "A",
// SYNC-NEXT:          "port_patterns": [
// SYNC-NEXT:            "a",
// SYNC-NEXT:            "b"
// SYNC-NEXT:          ],
// SYNC-NEXT:          "comment": null
// SYNC-NEXT:        },
// SYNC-NEXT:        {
// SYNC-NEXT:          "name_pattern": "B",
// SYNC-NEXT:          "port_patterns": [
// SYNC-NEXT:            "a",
// SYNC-NEXT:            "b"
// SYNC-NEXT:          ],
// SYNC-NEXT:          "comment": null
// SYNC-NEXT:        }
// SYNC-NEXT:      ]
//
// CHECK-NEXT:   }
