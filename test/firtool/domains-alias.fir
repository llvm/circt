; RUN: firtool %s -domain-mode=infer -o /dev/null
; RUN: firtool %s -domain-mode=infer-all -o /dev/null
; RUN: firtool %s -domain-mode=check -o /dev/null

; Regression test for https://github.com/llvm/circt/issues/10755. Bar's
; output domain is an alias of its input domain. That alias must be visible
; at the instance in Foo so that X and A are recognized as the same domain.
FIRRTL version 7.0.0
circuit Foo:
  domain ClockDomain:
  module Bar:
    input A: Domain of ClockDomain
    output B: Domain of ClockDomain
    domain_define B = A
  public module Foo:
    input A: Domain of ClockDomain
    input in: UInt<1> domains [A]
    inst bar of Bar
    domain_define bar.A = A
    wire X: Domain of ClockDomain
    domain_define X = bar.B
    wire x: UInt<1> domains [X]
    connect x, in
