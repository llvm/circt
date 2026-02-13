; RUN: rm -rf %t
; RUN: firtool %s -split-verilog -o %t
; RUN: cat %t/ref_Top.sv | FileCheck %s

; CHECK:      `define ref_Top_direct_probe _GEN{{(_[[0-9]+])?}}
; CHECK-NEXT: `define ref_Top_inner_x_probe inner.x
; CHECK-NEXT: `define ref_Top_inner_y_probe _GEN{{(_[0-9]+])?}}
; CHECK-NEXT: `define ref_Top_keyword_probe _GEN{{(_[[0-9]+])?}}

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
