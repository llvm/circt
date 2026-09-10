; RUN: firtool %s | FileCheck %s

FIRRTL version 4.0.0
circuit PortDelete : %[[
  {
    "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
    "name": "MyView",
    "companion": "~PortDelete|MyView_companion",
    "parent": "~PortDelete|DUT",
    "view": {
      "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
      "defName": "MyInterface",
      "elements": [
        {
          "name": "ground",
          "description": "a ground type port",
          "tpe": {
            "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
            "ref": {
              "module": "PortDelete",
              "path": [
                {
                  "_1": {
                    "class": "firrtl.annotations.TargetToken$Instance",
                    "value": "dut"
                  },
                  "_2": {
                    "class": "firrtl.annotations.TargetToken$OfModule",
                    "value": "DUT"
                  }
                }
              ],
              "ref": "w",
              "component": []
            },
            "tpe": {
              "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
            }
          }
        }
      ]
    }
  }
]]
  module MyView_companion :

    wire _WIRE : UInt<1>
    connect _WIRE, UInt<1>(0h0)

  module DUT :
    output out : UInt<1>

    wire w : UInt<1>
    connect w, UInt<1>(0h1)
    connect out, w
    inst MyView_companion of MyView_companion

  public module PortDelete :
    output out : UInt<1>

    inst dut of DUT
    connect out, dut.out

; Assert that the DUT has no ports in Verilog.  This ensures that even though
; wire "w" is part of a Grand Central View, it does not block optimizations.
; The constant one is allowed to flow through w and out of the design which
; makes the output port "out" on the DUT capable of being deleted.
;
; CHECK-LABEL: module DUT();
