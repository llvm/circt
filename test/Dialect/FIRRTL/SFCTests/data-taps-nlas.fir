; RUN: firtool --firrtl-grand-central --verilog --split-input-file %s | FileCheck %s
; See https://github.com/llvm/circt/issues/2691

; Note: This includes DontTouchAnnotations to prevent these from getting
; optimized away to constants as the purpose of this test is not to check
; constant propagation behavior, but that NLAs work.
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
  ]
},
{
  "class": "firrtl.transforms.DontTouchAnnotation",
  "target": "~Top|Foo>g"
},
{
  "class": "firrtl.transforms.DontTouchAnnotation",
  "target": "~Top|Foo>f"
},
{
  "class": "firrtl.transforms.DontTouchAnnotation",
  "target": "~Top|Top>k"
}]]
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

; CHECK-LABEL: module Foo(
; CHECK:         assign [[g_tap:_gctTap.*]] = {{.*g.*}};
; CHECK:         assign [[f_tap:_gctTap.*]] = f;
; CHECK:       endmodule

; CHECK-LABEL: module Top();
; CHECK:         assign [[k_tap:.+]] = k;

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

; Note: This includes DontTouchAnnotations to prevent these from getting
; optimized away to constants as the purpose of this test is not to check
; constant propagation behavior, but that NLAs work.
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
    "target": "~TestHarness|TestHarness/system:Top/test:Test>signal"
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
    wire signal : UInt<8>
    signal is invalid

; CHECK:     module DataTap
; CHECK-NOT: endmodule
; CHECK:       assign _0 = TestHarness.system.test.[[signal_tap:.+]];

; CHECK:     module Test()
; CHECK-NOT: endmodule
; CHECK:       assign [[signal_tap]] = signal;
