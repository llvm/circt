; RUN: firtool --no-dedup --split-input-file %s | FileCheck %s

; A number of tests use an invalidated wire with a DontTouchAnnotation to block
; constant sinking into the XMR.  This is to ensure that the test can check a
; valid XMR as opposed to "assign foo = 1'h0".

; This test was extracted from:
; - github.com/sifive/$internal:
;   - src/test/scala/grandcentral/DataTapsTest.scala
FIRRTL version 4.0.0
circuit Top : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"foo.bar.in",
        "sink":"~Top|Top>tap_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"foo.bar.out",
        "sink":"~Top|Top>tap_1"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>in"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>out"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_0"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_1"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_2"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Child>always"
  },
  {
    "class":"sifive.enterprise.grandcentral.MemTapAnnotation",
    "source": "~Top|Child>always",
    "sink":["~Top|Top>tap_2[0]", "~Top|Top>tap_2[1]"]
  }
]]
  extmodule BlackBox :
    input in : UInt<1>
    output out : UInt<1>
    defname = BlackBox

  module Child :
    input clock : Clock
    input reset : Reset
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst localparam of BlackBox
    invalidate localparam.out
    invalidate localparam.in
    connect localparam.in, io.in
    connect io.out, localparam.out

    cmem always : UInt<1>[2]

    infer mport w = always[io.in], clock
    connect w, io.in

    ; mem always :
    ;   data-type => UInt<1>
    ;   depth => 2
    ;   read-latency => 0
    ;   write-latency => 1
    ;   writer => w
    ;   read-under-write => undefined
    ; connect always.w.clk, clock
    ; connect always.w.en, UInt<1>(1)
    ; connect always.w.addr, io.in
    ; connect always.w.data, io.in
    ; connect always.w.mask, UInt<1>(1)

  module ChildWrapper :
    input clock : Clock
    input reset : Reset
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst signed of Child
    connect signed.clock, clock
    connect signed.reset, reset
    connect signed.io.in, io.in
    connect io.out, signed.io.out

  public module Top :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst unsigned of ChildWrapper
    connect unsigned.clock, clock
    connect unsigned.reset, reset
    wire in : UInt<1>
    wire out : UInt<1>
    node _child_io_in_T = and(io.in, in)
    connect unsigned.io.in, _child_io_in_T
    node _io_out_T = and(unsigned.io.out, out)
    connect io.out, _io_out_T

    wire tap_0: UInt<1>
    invalidate tap_0

    wire tap_1: UInt<1>
    invalidate tap_1

    wire tap_2: UInt<1>[2]
    invalidate tap_2

    connect in, or(tap_0, tap_2[0])
    connect out, or(tap_1, tap_2[1])

; CHECK-LABEL:      module Top
; CHECK:        tap_0 = Top.unsigned_0.signed_0.localparam_0.foo.bar.in;
; CHECK-NEXT:   tap_1 = Top.unsigned_0.signed_0.localparam_0.foo.bar.out;
; CHECK-NEXT:   tap_2_0 = Top.unsigned_0.signed_0.always_ext.Memory[0];
; CHECK-NEXT:   tap_2_1 = Top.unsigned_0.signed_0.always_ext.Memory[1];

; // -----

FIRRTL version 4.0.0
circuit Top : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo>f",
        "sink": "~Top|Top>b"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo>g",
        "sink": "~Top|Top>c"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top>k",
        "sink": "~Top|Top>d"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Foo>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Foo>g"
  }
]]
  module Foo :
    output g : UInt<1>
    wire inv: UInt<1>
    invalidate inv
    wire f : UInt<1>
    connect g, inv
    connect f, inv

  public module Top:
    output io : { b : UInt<1>, c : UInt<1>, d: UInt<1> }

    inst foo of Foo
    wire inv: UInt<1>
    invalidate inv
    wire k : UInt<1>
    connect k, inv

    wire b: UInt<1>
    invalidate b

    wire c: UInt<1>
    invalidate c

    wire d: UInt<1>
    invalidate d

    connect io.b, b
    connect io.c, c
    connect io.d, d

    ; CHECK-LABEL:      module Top
    ; TODO: fix having constants carry names
    ; CHECK:        wire k = 1'h0
    ; CHECK:        io_b = Top.foo.f_probe;
    ; CHECK-NEXT:   io_c = Top.foo.g_probe;
    ; CHECK-NEXT:   io_d = k;

; // -----

FIRRTL version 4.0.0
circuit TestHarness : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TestHarness|TestHarness/system:Top/test:Test>signal",
        "sink":"~TestHarness|Companion>tap_0"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~TestHarness|Test>inv"
  }
]]
  public module TestHarness :
    output io : UInt<1>
    inst system of Top
    connect io, system.io
  module Top :
    output io : UInt<1>
    inst test of Test
    inst Companion of Companion
    connect io, Companion.io
  module Companion :
    output io : UInt<1>
    wire tap_0: UInt<8>
    invalidate tap_0

    connect io, tap_0
  module Test :
    wire inv: UInt<1>
    invalidate inv
    wire signal : UInt<8>
    connect signal, inv

; CHECK-LABEL: module Top(
; CHECK-NOT:   endmodule
; CHECK:         Companion Companion (
; CHECK-NEXT:    .io (io),
; CHECK-NEXT:    .[[bore:[a-zA-Z0-9_]+]] (Top.test.signal_probe)
; CHECK:       endmodule

; CHECK-LABEL: module Companion(
; CHECK-NOT:   endmodule
; CHECK-NEXT:    output io,
; CHECK-NEXT:    input [7:0] [[bore]]
; CHECK:         io = [[bore]][0];
; CHECK:       endmodule

; // -----

FIRRTL version 4.0.0
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
    invalidate port_Submodule

    wire inv: UInt<1>
    invalidate inv

    wire wire_Submodule: UInt<1>
    connect wire_Submodule, inv

    wire tap_0 : UInt<1>
    wire tap_1 : UInt<1>
    wire tap_2 : UInt<1>
    wire tap_3 : UInt<1>
    wire tap_4 : UInt<1>
    wire tap_5 : UInt<1>
    invalidate tap_0
    invalidate tap_1
    invalidate tap_2
    invalidate tap_3
    invalidate tap_4
    invalidate tap_5

  module DUT :
    output port_DUT: UInt<1>
    invalidate port_DUT

    wire inv: UInt<1>
    invalidate inv

    wire wire_DUT: UInt<1>
    connect wire_DUT, inv

    inst submodule of Submodule

    wire tap_6 : UInt<1>
    wire tap_7 : UInt<1>
    wire tap_8 : UInt<1>
    wire tap_9 : UInt<1>
    wire tap_10 : UInt<1>
    wire tap_11 : UInt<1>
    invalidate tap_6
    invalidate tap_7
    invalidate tap_8
    invalidate tap_9
    invalidate tap_10
    invalidate tap_11

  public module Top :
    output port_Top : UInt<1>
    invalidate port_Top

    wire inv: UInt<1>
    invalidate inv

    wire wire_Top: UInt<1>
    connect wire_Top, inv

    inst dut of DUT
    wire tap_12 : UInt<1>[6]
    invalidate tap_12

; CHECK-LABEL: module Submodule
; CHECK-NEXT:    input [[Submodule_tap_1_port:[a-zA-Z0-9_]+]]
; CHECK-NEXT:          [[Submodule_tap_2_port:[a-zA-Z0-9_]+]]
; CHECK-NEXT:          [[Submodule_tap_5_port:[a-zA-Z0-9_]+]]
;
; CHECK-DAG:     tap_0 = inv;
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
; CHECK-DAG:     tap_6 = DUT.submodule.inv_probe;
; CHECK-DAG:     tap_7 = inv;
; CHECK-DAG:     tap_8 = [[DUT_tap_8_port]];
; CHECK-DAG:     tap_9 = 1'h0;
; CHECK-DAG:     tap_10 = 1'h0;
; CHECK-DAG:     tap_11 = [[DUT_tap_11_port]];
;
; CHECK:         Submodule submodule (
; CHECK-DAG:       .[[Submodule_tap_1_port]] (inv)
; CHECK-DAG:       .[[Submodule_tap_2_port]] ([[DUT_tap_2_port]])
; CHECK-DAG:       .[[Submodule_tap_5_port]] ([[DUT_tap_5_port]])

; CHECK-LABEL: module Top
;
; CHECK-DAG:     tap_12_0 = Top.dut.submodule.inv_probe
; CHECK-DAG:     tap_12_1 = Top.dut.inv_probe
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

; // -----

FIRRTL version 4.0.0
circuit Top : %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Top|Child>sum"
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|Child",
        "internalPath":"sum",
        "sink":"~Top|Top>dt._0"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|Child",
        "internalPath":"not.a.real.path",
        "sink":"~Top|Top>dt._1"
      }
    ]
  }
]]
  module Child :
    output io : { flip in : UInt<8>, out : UInt<8>}

    node sum = tail(add(io.in, UInt<1>(1)), 1)

    connect io.out, sum

  public module Top :
    output io : { flip in : UInt<8>, out : UInt<8>}
    output taps : UInt<8>[2]

    inst child of Child
    connect io, child.io

    wire dt: {_0: UInt<8>, _1: UInt<8>}
    invalidate dt

    connect taps[0], dt._0
    connect taps[1], dt._1

; CHECK:     module Top(
; CHECK-NOT: endmodule
; CHECK:       taps_0 = Top.child.sum;
; CHECK-NEXT:  taps_1 = Top.child.not.a.real.path;
; CHECK:     endmodule
