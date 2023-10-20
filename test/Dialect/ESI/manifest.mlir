// RUN: circt-opt %s --esi-build-manifest=to-file=%t1.json
// RUN: FileCheck --input-file=%t1.json %s

hw.type_scope @__hw_typedecls {
  hw.typedecl @foo, "Foo" : i1
}
!alias = !hw.typealias<@__hw_typedecls::@foo, i1>

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  %0 = esi.null : !esi.channel<si14>
  esi.cosim %clk, %rst, %0, "t2" : !esi.channel<si14> -> !esi.channel<i32>

  %1 = esi.null : !esi.channel<!esi.list<ui14>>
  esi.cosim %clk, %rst, %1, "t1" : !esi.channel<!esi.list<ui14>> -> !esi.channel<!hw.array<3xi5>>

  %2 = esi.null : !esi.channel<!esi.any>
  esi.cosim %clk, %rst, %2, "t2" : !esi.channel<!esi.any> -> !esi.channel<!hw.struct<"foo": i5>>

  %3 = esi.null : !esi.channel<!alias>
  esi.cosim %clk, %rst, %3, "t3" : !esi.channel<!alias> -> !esi.channel<i32>
}

// CHECK:      {
// CHECK-NEXT:   "api_version": 1,
// CHECK-LABEL:  "types": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<si14>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "si14",
// CHECK-NEXT:         "dialect": "builtin",
// CHECK-NEXT:         "hw_bitwidth": 14,
// CHECK-NEXT:         "mnemonic": "int",
// CHECK-NEXT:         "signedness": "signed"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<i32>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "i32",
// CHECK-NEXT:         "dialect": "builtin",
// CHECK-NEXT:         "hw_bitwidth": 32,
// CHECK-NEXT:         "mnemonic": "int",
// CHECK-NEXT:         "signedness": "signless"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!esi.list<ui14>>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!esi.list<ui14>",
// CHECK-NEXT:         "dialect": "esi",
// CHECK-NEXT:         "element": {
// CHECK-NEXT:           "circt_name": "ui14",
// CHECK-NEXT:           "dialect": "builtin",
// CHECK-NEXT:           "hw_bitwidth": 14,
// CHECK-NEXT:           "mnemonic": "int",
// CHECK-NEXT:           "signedness": "unsigned"
// CHECK-NEXT:         },
// CHECK-NEXT:         "mnemonic": "list"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!hw.array<3xi5>>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!hw.array<3xi5>",
// CHECK-NEXT:         "dialect": "hw",
// CHECK-NEXT:         "element": {
// CHECK-NEXT:           "circt_name": "i5",
// CHECK-NEXT:           "dialect": "builtin",
// CHECK-NEXT:           "hw_bitwidth": 5,
// CHECK-NEXT:           "mnemonic": "int",
// CHECK-NEXT:           "signedness": "signless"
// CHECK-NEXT:         },
// CHECK-NEXT:         "hw_bitwidth": 15,
// CHECK-NEXT:         "mnemonic": "array",
// CHECK-NEXT:         "size": 3
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!esi.any>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!esi.any",
// CHECK-NEXT:         "dialect": "esi",
// CHECK-NEXT:         "mnemonic": "any"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!hw.struct<foo: i5>>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!hw.struct<foo: i5>",
// CHECK-NEXT:         "dialect": "hw",
// CHECK-NEXT:         "fields": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "name": "foo",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:               "circt_name": "i5",
// CHECK-NEXT:               "dialect": "builtin",
// CHECK-NEXT:               "hw_bitwidth": 5,
// CHECK-NEXT:               "mnemonic": "int",
// CHECK-NEXT:               "signedness": "signless"
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         ],
// CHECK-NEXT:         "hw_bitwidth": 5,
// CHECK-NEXT:         "mnemonic": "struct"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!hw.typealias<@__hw_typedecls::@foo, i1>>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!hw.typealias<@__hw_typedecls::@foo, i1>",
// CHECK-NEXT:         "dialect": "hw",
// CHECK-NEXT:         "hw_bitwidth": 1,
// CHECK-NEXT:         "inner": {
// CHECK-NEXT:           "circt_name": "i1",
// CHECK-NEXT:           "dialect": "builtin",
// CHECK-NEXT:           "hw_bitwidth": 1,
// CHECK-NEXT:           "mnemonic": "int",
// CHECK-NEXT:           "signedness": "signless"
// CHECK-NEXT:         },
// CHECK-NEXT:         "mnemonic": "alias",
// CHECK-NEXT:         "name": "Foo"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
