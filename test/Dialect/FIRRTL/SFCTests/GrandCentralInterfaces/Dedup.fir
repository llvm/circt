; RUN: firtool --lowering-options=emitBindComments -dedup  %s | FileCheck %s

circuit Foo : %[[
  {
    "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
    "name": "bar_view",
    "companion": "~Foo|Companion_0",
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
              "circuit": "Foo",
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
    "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
    "name": "bar_view",
    "companion": "~Foo|Companion_1",
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
              "circuit": "Foo",
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
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Foo>a"
  }
]]
  module Companion_0 :

  module Companion_1 :

  module Foo :
    wire a: UInt<1>
    a is invalid

    inst bar of Companion_0
    inst baz of Companion_1


; CHECK:      module Companion_0(
; CHECK-NEXT:   input bar_view_bar_in__bore
; CHECK-NEXT: );

; CHECK:        Bar_View bar_view();
; CHECK-NEXT:   wire bar_in_wire;
; CHECK-NEXT:   assign bar_view.bar_in = bar_in_wire;
; CHECK-NEXT:   assign bar_in_wire = bar_view_bar_in__bore;
; CHECK-NEXT: endmodule

; CHECK:      module Foo();
; CHECK-NEXT:   wire a = 1'h0;
; CHECK-NEXT:   Companion_0 bar (
; CHECK-NEXT:     .bar_view_bar_in__bore (a)
; CHECK-NEXT:   );
; CHECK-NEXT:   Companion_0 baz (
; CHECK-NEXT:     .bar_view_bar_in__bore (a)
; CHECK-NEXT:   );
; CHECK-NEXT: endmodule

; CHECK:      interface Bar_View;
; CHECK-NEXT:   // bar in
; CHECK-NEXT:   logic bar_in;
; CHECK-NEXT: endinterface
