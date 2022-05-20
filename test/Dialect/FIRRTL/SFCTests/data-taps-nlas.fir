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
  extmodule DataTap :
    output b : UInt<1>
    output c : UInt<1>
    output d : UInt<1>
    defname = DataTap

  module Foo :
    output g : UInt<1>
    wire f : UInt<1>
    g is invalid
    f is invalid

  module Top:
    inst foo of Foo
    inst bar of Foo
    inst dataTap of DataTap
    wire k : UInt<1>
    k is invalid

; CHECK: module DataTap_impl_0(
; CHECK:   output b, 
; CHECK:          c,
; CHECK:          d
; CHECK: );
; CHECK:   assign b = Top.foo.[[F:.+]];
; CHECK:   assign c = Top.foo.[[G:.+]];
; CHECK:   assign d = Top.k;
; CHECK: endmodule

; CHECK: module Foo();
; CHECK:   wire [[G]] =
; CHECK:   wire [[F]] =
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
; CHECK:       assign _0 = TestHarness.system.test.signal;
