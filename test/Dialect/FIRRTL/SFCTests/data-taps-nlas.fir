; RUN: firtool --firrtl-grand-central --verilog --split-input-file %s | FileCheck %s
; See https://github.com/llvm/circt/issues/2691

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
]}]]
  module Foo :
    output g : UInt<1>
    wire f : UInt<1>
    g <= UInt<1>(0)
    f <= UInt<1>(1)

  module Top:
    inst foo of Foo
    inst bar of Foo
    inst dataTap of DataTap
    wire k : UInt<1>
    k is invalid

  extmodule DataTap :
    output b : UInt<1>
    output c : UInt<1>
    output d : UInt<1>
    defname = DataTap

; CHECK-LABEL: module Foo();
; CHECK-NEXT:    wire [[g_tap:.+]] = 1'h0;
; CHECK-NEXT:    wire f = 1'h1;
; CHECK-NEXT:    wire [[f_tap:.+]] = 1'h1;
; CHECK:       endmodule

; CHECK-LABEL: module Top();
; CHECK:         wire k = 1'h0;
; CHECK:         wire [[k_tap:.+]] = 1'h0;

; CHECK-LABEL: module DataTap_impl_0(
; CHECK:   output b,
; CHECK:          c,
; CHECK:          d
; CHECK: );
; CHECK:   assign b = Top.foo.[[f_tap]];
; CHECK:   assign c = Top.foo.[[g_tap]];
; CHECK:   assign d = Top.[[k_tap]];
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
  }
]]
  module TestHarness :
    inst system of Top
  module Top :
    inst test of Test
    inst Companion of Companion
  extmodule DataTap :
    output _0 : UInt<1>
    defname = DataTap
  module Companion :
    inst DataTap of DataTap
  module Test :
    wire signal : UInt<8>
    signal is invalid

; CHECK:     module DataTap
; CHECK-NOT: endmodule
; CHECK:       assign _0 = TestHarness.system.test.[[signal_tap:.+]];

; CHECK:     module Test()
; CHECK-NOT: endmodule
; CHECK:       wire [7:0] [[signal_tap]] = 8'h0;
