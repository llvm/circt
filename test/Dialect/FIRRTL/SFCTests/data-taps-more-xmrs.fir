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
        "sink":"~Top|DUT>tap_6"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "sink":"~Top|DUT>tap_7"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "sink":"~Top|DUT>tap_8"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "sink":"~Top|DUT>tap_9"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "sink":"~Top|DUT>tap_10"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "sink":"~Top|DUT>tap_11"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "sink":"~Top|Top>tap_12[0]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "sink":"~Top|Top>tap_12[1]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "sink":"~Top|Top>tap_12[2]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "sink":"~Top|Top>tap_12[3]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "sink":"~Top|Top>tap_12[4]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "sink":"~Top|Top>tap_12[5]"
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
    "target": "~Top|DUT>tap_6"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_7"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_8"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_9"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_10"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_11"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_12"
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

    wire tap_6 : UInt<1>
    wire tap_7 : UInt<1>
    wire tap_8 : UInt<1>
    wire tap_9 : UInt<1>
    wire tap_10 : UInt<1>
    wire tap_11 : UInt<1>
    tap_6 is invalid
    tap_7 is invalid
    tap_8 is invalid
    tap_9 is invalid
    tap_10 is invalid
    tap_11 is invalid

  module Top :
    output port_Top : UInt<1>
    port_Top is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Top: UInt<1>
    wire_Top <= inv

    inst dut of DUT
    wire tap_12 : UInt<1>[6]
    tap_12 is invalid

    ; CHECK-LABEL: module Submodule
    ; CHECK-NEXT:    input [[Submodule_tap_1_port:[a-zA-Z0-9_]+]]
    ; CHECK-NEXT:          [[Submodule_tap_2_port:[a-zA-Z0-9_]+]]
    ; CHECK-NEXT:          [[Submodule_tap_5_port:[a-zA-Z0-9_]+]]
    ;
    ; CHECK-DAG:     tap_0 = wire_Submodule;
    ; CHECK-DAG:     tap_1 = [[Submodule_tap_1_port]];
    ; CHECK-DAG:     tap_2 = [[Submodule_tap_2_port]];
    ; CHECK-DAG:     tap_3 = 1'h0;
    ; CHECK-DAG:     tap_4 = 1'h0;
    ; CHECK-DAG:     tap_5 = [[Submodule_tap_5_port]];

    ; CHECK-LABEL: module DUT
    ; CHECK-NEXT:    input [[DUT_tap_2_port:[a-zA-Z0-9_]+]]
    ; CHECK-NEXT:          [[DUT_tap_5_port:[a-zA-Z0-9_]+]]
    ; CHECK-NEXT:          [[DUT_tap_8_port:[a-zA-Z0-9_]+]]
    ; CHECK-NEXT:          [[DUT_tap_11_port:[a-zA-Z0-9_]+]]
    ;
    ; CHECK-DAG:     tap_6 = DUT.submodule.wire_Submodule;
    ; CHECK-DAG:     tap_7 = wire_DUT;
    ; CHECK-DAG:     tap_8 = [[DUT_tap_8_port]];
    ; CHECK-DAG:     tap_9 = 1'h0;
    ; CHECK-DAG:     tap_10 = 1'h0;
    ; CHECK-DAG:     tap_11 = [[DUT_tap_11_port]];
    ;
    ; CHECK:         Submodule submodule (
    ; CHECK-DAG:       .[[Submodule_tap_1_port]] (wire_DUT)
    ; CHECK-DAG:       .[[Submodule_tap_2_port]] ([[DUT_tap_2_port]])
    ; CHECK-DAG:       .[[Submodule_tap_5_port]] ([[DUT_tap_5_port]])

    ; CHECK-LABEL: module Top
    ;
    ; CHECK-DAG:     tap_12_0 = Top.dut.submodule.wire_Submodule
    ; CHECK-DAG:     tap_12_1 = Top.dut.wire_DUT
    ; CHECK-DAG:     tap_12_2 = inv;
    ; CHECK-DAG:     tap_12_3 = 1'h0
    ; CHECK-DAG:     tap_12_4 = 1'h0
    ; CHECK-DAG:     tap_12_5 = 1'h0
    ;
    ; CHECK:         DUT dut (
    ; CHECK-DAG:       .[[DUT_tap_2_port]] (inv)
    ; CHECK-DAG:       .[[DUT_tap_5_port]] (1'h0)
    ; CHECK-DAG:       .[[DUT_tap_8_port]] (inv)
    ; CHECK-DAG:       .[[DUT_tap_11_port]] (1'h0)
