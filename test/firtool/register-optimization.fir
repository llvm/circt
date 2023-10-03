; RUN: firtool %s | FileCheck %s
; Check that `r` is optimized and the constant is propagated to the top-level port. 
; CHECK-NOT:   Passthrough
; CHECK-NOT:   Child
; CHECK-LABEL: module Example
; CHECK:       assign out = 1'h0;

circuit Example :
  module Passthrough:
    input en: UInt<1>
    output out: UInt<1>
    out <= en

  module Child:
    input clock: Clock
    input zero: UInt<1>
    input unknown: UInt<1>
    output out: UInt<1>
    
    reg r : UInt<1>, clock
    r <= zero
    node b = and(r, unknown)
    inst p of Passthrough
    p.en <= b
    out <= p.out

  module Example:
    input clock: Clock
    input unknown: UInt<1>
    output out: UInt<1>
    inst c of Child
    c.clock <= clock
    c.zero <= UInt<1>(0)
    c.unknown <= unknown
    out <= c.out
