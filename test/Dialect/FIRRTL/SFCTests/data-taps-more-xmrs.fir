; RUN: firtool --verilog %s | FileCheck %s

circuit Top : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "sink":"~Top|Submodule>tap_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "sink":"~Top|Submodule>tap_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "sink":"~Top|Submodule>tap_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "sink":"~Top|Submodule>tap_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "sink":"~Top|Submodule>tap_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "sink":"~Top|Submodule>tap_5"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "sink":"~Top|DUT>tap_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "sink":"~Top|DUT>tap_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "sink":"~Top|DUT>tap_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "sink":"~Top|DUT>tap_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "sink":"~Top|DUT>tap_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "sink":"~Top|DUT>tap_5"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "sink":"~Top|Top>tap[0]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "sink":"~Top|Top>tap[1]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "sink":"~Top|Top>tap[2]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "sink":"~Top|Top>tap[3]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "sink":"~Top|Top>tap[4]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "sink":"~Top|Top>tap[5]"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_0"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_1"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_2"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_3"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_4"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_5"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_0"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_1"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_2"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_3"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_4"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_5"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap"
  }
]]
  module Submodule :
    output port_Submodule: UInt<1>
    port_Submodule is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Submodule: UInt<1>
    wire_Submodule <= inv

    wire tap_0 : UInt<1>
    wire tap_1 : UInt<1>
    wire tap_2 : UInt<1>
    wire tap_3 : UInt<1>
    wire tap_4 : UInt<1>
    wire tap_5 : UInt<1>
    tap_0 is invalid
    tap_1 is invalid
    tap_2 is invalid
    tap_3 is invalid
    tap_4 is invalid
    tap_5 is invalid

  module DUT :
    output port_DUT: UInt<1>
    port_DUT is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_DUT: UInt<1>
    wire_DUT <= inv

    inst submodule of Submodule

    wire tap_0 : UInt<1>
    wire tap_1 : UInt<1>
    wire tap_2 : UInt<1>
    wire tap_3 : UInt<1>
    wire tap_4 : UInt<1>
    wire tap_5 : UInt<1>
    tap_0 is invalid
    tap_1 is invalid
    tap_2 is invalid
    tap_3 is invalid
    tap_4 is invalid
    tap_5 is invalid

  module Top :
    output port_Top : UInt<1>
    port_Top is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Top: UInt<1>
    wire_Top <= inv

    inst dut of DUT
    wire tap : UInt<1>[6]
    tap is invalid

    ; CHECK-LABEL: module Submodule
    ; CHECK:         wire tap_3 = 1'h0
    ; CHECK-NEXT:    wire tap_4 = 1'h0
    ; CHECK-DAG:     assign tap_0 = wire_Submodule
    ; CHECK-DAG:     assign tap_1 = DUT.wire_DUT
    ; CHECK-DAG:     assign tap_2 = Top.wire_Top
    ; CHECK-DAG:     assign tap_5 = Top.port_Top

    ; CHECK-LABEL: module DUT
    ; CHECK:         wire tap_3 = 1'h0
    ; CHECK-NEXT:    wire tap_4 = 1'h0
    ; CHECK-DAG:     assign tap_0 = DUT.submodule.wire_Submodule
    ; CHECK-DAG:     assign tap_1 = wire_DUT
    ; CHECK-DAG:     assign tap_2 = Top.wire_Top
    ; CHECK-DAG:     assign tap_5 = Top.port_Top

    ; CHECK-LABEL: module Top
    ; CHECK:         wire tap_3 = 1'h0
    ; CHECK-NEXT:    wire tap_4 = 1'h0
    ; CHECK-NEXT:    wire tap_5 = 1'h0
    ; CHECK-DAG:     assign tap_0 = Top.dut.submodule.wire_Submodule
    ; CHECK-DAG:     assign tap_1 = Top.dut.wire_DUT
    ; CHECK-DAG:     assign tap_2 = wire_Top
