; RUN: firtool %s | FileCheck %s
FIRRTL version 4.0.0

; This test relies on IM const prop to propagate a constant predicate down
; to an assertion, which then will allow the assertion canonicalizer to erase
; the assertion.
;
; However, lower layers will pull the assertion out into a new module
; (corresponding to the surrounding layerblock), which undoes the IMCP work, and
; will block the assertion from being canonicalized away.
;
; This test ensures that IMCP and canonicalizers are run together, and that
; dead assertions under layerblocks will be removed.

; CHECK-NOT: MYASSERT
; CHECK-NOT: MYLABEL
; CHECK-NOT: $error

circuit Top:
  layer Verification, bind:
    layer Assert, bind:

  module DoAssert:
    input c : Clock
    input r : UInt<1>
    input p : UInt<1>
    layerblock Verification:
      layerblock Assert:
        node x = asUInt(r)
        node y = eq(x, UInt<1>(0h0))
        intrinsic(circt_chisel_ifelsefatal<format = "MYASSERT", label = "MYLABEL">, c, p, y)

  public module Top:
    input c : Clock
    input r : UInt<1>
    inst do_assert of DoAssert
    connect do_assert.c, c
    connect do_assert.r, r
    connect do_assert.p, UInt<1>(1)
