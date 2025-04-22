; RUN: firtool %s --grand-central-companion-mode=instantiate | FileCheck %s --check-prefixes=INST
; RUN: firtool %s --grand-central-companion-mode=drop | FileCheck %s --check-prefixes=DROP
; RUN: firtool %s --grand-central-companion-mode=bind | FileCheck %s --check-prefixes=BIND

FIRRTL version 4.0.0
circuit Foo : %[[
  {
    "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
    "name": "bar_view",
    "companion": "~Foo|Companion",
    "parent": "~Foo|Foo",
    "view": {
      "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
      "defName": "Bar_View",
      "elements": [
        {
          "name": "bar_in",
          "description": "bar in",
          "tpe": {
            "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
            "ref": {
              "module": "Foo",
              "path": [],
              "ref": "a",
              "component": []
            },
            "tpe": {
              "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
            }
          }
        }
      ]
    }
  },
  {
    "class": "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    "directory": "Wire/firrtl/gct",
    "filename": "Wire/firrtl/bindings.sv"
  },
  {
    "class": "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
    "filename": "gct.yaml"
  }
]]
  module Companion :
    input clock: Clock
    input a: UInt<1>

    assert(clock, a, UInt<1>(1), "hello")

  public module Foo :
    input clock: Clock
    input a_in: UInt<1>

    wire a: UInt<1>
    connect a, a_in

    inst companion of Companion
    connect companion.clock, clock
    connect companion.a, a

    ; INST:     module Foo
    ; INST-NOT: endmodule
    ; INST-NOT:   emitted as a bind statement
    ; INST:       Companion companion
    ; INST:     endmodule
    ; INST:     FILE "gct.yaml"
    ; INST:     []

    ; DROP-NOT: module Companion
    ; DROP:     module Foo
    ; DROP-NOT: Companion
    ; DROP: endmodule
    ; DROP:     FILE "gct.yaml"
    ; DROP:     []

    ; BIND: module Foo
    ; BIND:     FILE "gct.yaml"
    ; BIND:     []
    ; BIND: module Companion
    ; BIND: bind Foo Companion companion
