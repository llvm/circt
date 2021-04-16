; RUN: circt-translate -import-firrtl %s -verify-diagnostics
; XFAIL: *

circuit TypeCanonicalizationStillConsideredHarmful:
  module TypeCanonicalizationStillConsideredHarmful:

    ; This test checks for erroneous situations involving connections
    ; and sink flows.  Type canonicalization makes some of these tests
    ; impossible to check.  E.g., "wire d" looks identical to "wire a"
    ; after canonicalization of its two flips, but disallows "d.a" on
    ; the RHS of a connection.

    output a: {a:      {     a: UInt<1>}}
    output b: {a:      {flip a: UInt<1>}}
    output c: {flip a: {     a: UInt<1>}}
    output d: {flip a: {flip a: UInt<1>}}

    wire ax: {a:      {     a: UInt<1>}}
    wire bx: {a:      {flip a: UInt<1>}}
    wire cx: {flip a: {     a: UInt<1>}}
    wire dx: {flip a: {flip a: UInt<1>}}

    ; @expected-error @+1 {{invalid}}
    b.a.a <- bx.a.a
    ; @expected-error @+1 {{invalid}}
    b.a.a <= bx.a.a

    ; @expected-error @+1 {{invalid}}
    c.a <- cx.a
    ; @expected-error @+1 {{invalid}}
    c.a <= cx.a
    ; @expected-error @+1 {{invalid}}
    c.a.a <- cx.a.a
    ; @expected-error @+1 {{invalid}}
    c.a.a <= cx.a.a

    ; @expected-error @+1 {{invalid}}
    d.a <- dx.a
    ; @expected-error @+1 {{invalid}}
    d.a <= dx.a
