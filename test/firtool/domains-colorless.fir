; RUN: firtool %s -domain-mode=infer -o /dev/null
; RUN: firtool %s -domain-mode=infer -output-final-mlir - -o /dev/null | FileCheck %s

; Reproducer for llvm/circt#10830: a constant, or an expression fully composed
; of constants, is colorless and may be consumed by values in two different
; domains without being treated as an illegal domain crossing.
;
;   - `x` is a node bound directly to a constant.
;   - `y` is a node bound to a constant *expression* (not just a bare
;     constant).
;
; Both are used across two distinct domains A and B.

FIRRTL version 7.0.0
circuit Foo:

  domain ClockDomain:

  public module Foo:
    input A: Domain of ClockDomain
    input B: Domain of ClockDomain
    output a: UInt<1> domains [A]
    output b: UInt<1> domains [B]
    output c: UInt<1> domains [A]
    output d: UInt<1> domains [B]

    ; Bare constant used in two domains.
    node x = UInt<1>(1)
    connect a, x
    connect b, x

    ; Constant expression used in two domains.
    node y = eq(UInt<1>(1), UInt<1>(1))
    connect c, y
    connect d, y

; CHECK-LABEL: om.class @Foo_Class
