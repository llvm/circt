; RUN: firtool %s | FileCheck %s

FIRRTL version 4.0.0

; Check that layerblocks inside an inline child module are merged with
; layerblocks in the parent.

; CHECK-NOT: module Child_Verification
; CHECK-NOT: module Child

; CHECK: module Top_Verification_Assert
; CHECK:   assert(p) else $error("before child");
; CHECK:   assert(p) else $error("in child");
; CHECK:   assert(p) else $error("after child");
; CHECK: endmodule

circuit Top: %[[
  { "class": "firrtl.passes.InlineAnnotation", "target": "~Top|Child" }
]]

  layer Verification, bind:
    layer Assert, bind:

  module Child:
    input p : UInt<1>
    input e : UInt<1>
    input c : Clock

    layerblock Verification:
      layerblock Assert:
        assert(c, p, e, "in child")
        
  public module Top:
    input p : UInt<1>
    input e : UInt<1>
    input c : Clock

    layerblock Verification:
      layerblock Assert:
        assert(c, p, e, "before child")
    
    inst child of Child
    connect child.p, p
    connect child.e, e
    connect child.c, c

    layerblock Verification:
      layerblock Assert:
        assert(c, p, e, "after child")
