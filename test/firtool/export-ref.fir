; RUN: rm -rf %t
; RUN: firtool %s | FileCheck %s

; CHECK-LABEL: module Top()
; CHECK:   wire [2:0] [[xmr_probe:.+]] = 3'h0;
; CHECK:   wire       [[xmr_probe_0:.+]] = 1'h0;
; CHECK:   wire [1:0] [[xmr_probe_1:.+]] = 2'h0;
; 
; CHECK-LABEL: module Inner()
; CHECK:   wire [1:0] x = 2'h0;
; 
; CHECK: `define ref_Top_direct_probe [[xmr_probe_0]]
; CHECK: `define ref_Top_inner_x_probe inner.x
; CHECK: `define ref_Top_inner_y_probe [[xmr_probe_1]]
; CHECK: `define ref_Top_keyword_probe [[xmr_probe]]

FIRRTL version 3.0.0
circuit Top: %[[
{"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|Inner>x"}
]]
  module Top:
    output direct_probe: Probe<UInt<1>>
    output inner_x_probe: Probe<UInt<2>>
    output inner_y_probe: Probe<UInt<2>>
    output keyword_probe: Probe<UInt<3>>

    wire direct: UInt<1>
    invalidate direct
    define direct_probe = probe(direct)

    inst inner of Inner
    define inner_x_probe = inner.x_probe
    define inner_y_probe = inner.y_probe

    wire always: UInt<3>
    invalidate always
    define keyword_probe = probe(always)

  module Inner:
    output x_probe: Probe<UInt<2>>
    output y_probe: Probe<UInt<2>>

    wire x: UInt<2>
    invalidate x
    wire y : UInt<2>
    invalidate y

    define x_probe = probe(x)
    define y_probe = probe(y)
