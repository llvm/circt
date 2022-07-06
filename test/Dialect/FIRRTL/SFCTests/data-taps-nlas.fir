; RUN: firtool --firrtl-grand-central --verilog --split-input-file %s | FileCheck %s
; See https://github.com/llvm/circt/issues/2691

; A number of tests use an invalidated wire with a DontTouchAnnotation to block
; constant sinking into the XMR.  This is to ensure that the test can check a
; valid XMR as opposed to "assign foo = 1'h0".

circuit Top : %[[{
  "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
  "blackBox": "~Top|DataTap",
  "keys": [
    {
      "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      "source": "~Top|Top/foo:Foo>f",
      "portName": "~Top|DataTap>b"
    },
    {
      "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      "source": "~Top|Top/foo:Foo>g",
      "portName": "~Top|DataTap>c"
    },
    {
      "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      "source": "~Top|Top>k",
      "portName": "~Top|DataTap>d"
    }
]}, {
  "class": "firrtl.transforms.DontTouchAnnotation",
  "target": "~Top|Foo>inv"
}, {
  "class": "firrtl.transforms.DontTouchAnnotation",
  "target": "~Top|Top>inv"
}]]
  extmodule DataTap :
    output b : UInt<1>
    output c : UInt<1>
    output d : UInt<1>
    defname = DataTap

  module Foo :
    output g : UInt<1>
    wire inv: UInt<1>
    inv is invalid
    wire f : UInt<1>
    g <= inv
    f <= inv

  module Top:
    inst foo of Foo
    inst bar of Foo
    inst dataTap of DataTap
    wire inv: UInt<1>
    inv is invalid
    wire k : UInt<1>
    k <= inv

; CHECK: module DataTap_impl_0(
; CHECK:   output b,
; CHECK:          c,
; CHECK:          d
; CHECK: );
; CHECK:   assign b = Top.foo.f;
; CHECK:   assign c = Top.foo.g;
; CHECK:   assign d = Top.k;
; CHECK: endmodule

; // -----

circuit TestHarness : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~TestHarness|DataTap",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TestHarness|TestHarness/system:Top/test:Test>signal",
        "portName":"~TestHarness|DataTap>_0"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~TestHarness|Test>inv"
  }
]]
  module TestHarness :
    inst system of Top
  module Top :
    inst test of Test
    inst Companion of Companion
  extmodule DataTap :
    output _0 : UInt<8>
    defname = DataTap
  module Companion :
    inst DataTap of DataTap
  module Test :
    wire inv: UInt<1>
    inv is invalid
    wire signal : UInt<8>
    signal <= inv

; CHECK:     module DataTap
; CHECK-NOT: endmodule
; CHECK:       assign _0 = Top.test.signal;

; // -----

circuit Top : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~Top|DataTap_Submodule",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "portName":"~Top|DataTap_Submodule>_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "portName":"~Top|DataTap_Submodule>_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "portName":"~Top|DataTap_Submodule>_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "portName":"~Top|DataTap_Submodule>_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "portName":"~Top|DataTap_Submodule>_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "portName":"~Top|DataTap_Submodule>_5"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~Top|DataTap_DUT",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "portName":"~Top|DataTap_DUT>_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "portName":"~Top|DataTap_DUT>_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "portName":"~Top|DataTap_DUT>_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "portName":"~Top|DataTap_DUT>_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "portName":"~Top|DataTap_DUT>_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "portName":"~Top|DataTap_DUT>_5"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~Top|DataTap_Top",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "portName":"~Top|DataTap_Top>_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "portName":"~Top|DataTap_Top>_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "portName":"~Top|DataTap_Top>_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "portName":"~Top|DataTap_Top>_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "portName":"~Top|DataTap_Top>_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "portName":"~Top|DataTap_Top>_5"
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
  }
]]
  extmodule DataTap_Submodule :
    output _0 : UInt<1>
    output _1 : UInt<1>
    output _2 : UInt<1>
    output _3 : UInt<1>
    output _4 : UInt<1>
    output _5 : UInt<1>


  extmodule DataTap_DUT :
    output _0 : UInt<1>
    output _1 : UInt<1>
    output _2 : UInt<1>
    output _3 : UInt<1>
    output _4 : UInt<1>
    output _5 : UInt<1>

  extmodule DataTap_Top :
    output _0 : UInt<1>
    output _1 : UInt<1>
    output _2 : UInt<1>
    output _3 : UInt<1>
    output _4 : UInt<1>
    output _5 : UInt<1>

  module Submodule :
    output port_Submodule: UInt<1>
    port_Submodule is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Submodule: UInt<1>
    wire_Submodule <= inv

    inst tap of DataTap_Submodule

  module DUT :
    output port_DUT: UInt<1>
    port_DUT is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_DUT: UInt<1>
    wire_DUT <= inv

    inst submodule of Submodule

    inst tap of DataTap_DUT

  module Top :
    output port_Top : UInt<1>
    port_Top is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Top: UInt<1>
    wire_Top <= inv

    inst dut of DUT
    inst tap of DataTap_Top

    ; CHECK:      module DataTap_Submodule
    ; CHECK:      assign _0 = Submodule.wire_Submodule
    ; CHECK-NEXT: assign _1 = DUT.wire_DUT
    ; CHECK-NEXT: assign _2 = Top.wire_Top
    ; CHECK:      assign _3 = Submodule.port_Submodule
    ; CHECK-NEXT: assign _4 = DUT.port_DUT
    ; CHECK-NEXT: assign _5 = Top.port_Top

    ; CHECK: module DataTap_DUT
    ; CHECK:      assign _0 = DUT.submodule.wire_Submodule
    ; CHECK-NEXT: assign _1 = DUT.wire_DUT
    ; CHECK-NEXT: assign _2 = Top.wire_Top
    ; CHECK:      assign _3 = DUT.submodule.port_Submodule
    ; CHECK-NEXT: assign _4 = DUT.port_DUT
    ; CHECK-NEXT: assign _5 = Top.port_Top

    ; CHECK: module DataTap_Top
    ; CHECK:      assign _0 = Top.dut.submodule.wire_Submodule
    ; CHECK-NEXT: assign _1 = Top.dut.wire_DUT
    ; CHECK-NEXT: assign _2 = Top.wire_Top
    ; CHECK:      assign _3 = Top.dut.submodule.port_Submodule
    ; CHECK-NEXT: assign _4 = Top.dut.port_DUT
    ; CHECK-NEXT: assign _5 = Top.port_Top
