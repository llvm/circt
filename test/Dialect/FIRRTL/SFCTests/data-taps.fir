; RUN: firtool --no-dedup --split-input-file %s | FileCheck %s

; This test now only checks MemTaps, all DataTap support has been removed.

; A number of tests use an invalidated wire with a DontTouchAnnotation to block
; constant sinking into the XMR.  This is to ensure that the test can check a
; valid XMR as opposed to "assign foo = 1'h0".

; This test was extracted from:
; - github.com/sifive/$internal:
;   - src/test/scala/grandcentral/DataTapsTest.scala
FIRRTL version 4.0.0
circuit Top : %[[
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_2"
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

; CHECK:      module Top
; CHECK:        tap_2_0 = Top.unsigned_0.signed_0.always_ext.Memory[0];
; CHECK-NEXT:   tap_2_1 = Top.unsigned_0.signed_0.always_ext.Memory[1];

