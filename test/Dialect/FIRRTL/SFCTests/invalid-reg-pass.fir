; RUN: firtool -split-input-file -verilog %s | FileCheck %s

; This test checks register removal behavior for situations where the register
; is invalidated _through a primitive operation_.  This is intended to tease out
; gnarly bugs where, due to a combination of canonicalization, folding, and
; constant propagation, CIRCT does not remove registers which the Scala FIRRTL
; Compiler (SFC) does.  The CHECK/CHECK-NOT statements in this test indicate the
; SFC behavior.
;
; This test contains PASSING cases which are known to work.  For failing cases
; (which should be fixed and migrated into this file) see invalid-reg-fail.fir.
;
; The FIRRTL circuits in this file were generated using:
;   https://github.com/seldridge/firrtl-torture/blob/main/Invalid.scala

circuit and :
  module and :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<1>
    input in_1 : UInt<1>
    output out_0 : UInt<1>
    output out_1 : UInt<1>
    output out_2 : UInt<1>
    output out_3 : UInt<1>

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_3)
    node _T = and(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = and(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = and(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = and(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module and
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

circuit asAsyncReset :
  module asAsyncReset :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<1>
    output out_0 : AsyncReset
    output out_1 : AsyncReset

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : AsyncReset, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : AsyncReset, clock with :
      reset => (UInt<1>("h0"), r_1)
    node _T = asAsyncReset(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = asAsyncReset(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module asAsyncReset
    ; CHECK:         r_0
    ; CHECK:         r_1

    ; // -----

circuit asClock :
  module asClock :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<1>
    output out_0 : Clock
    output out_1 : Clock

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : Clock, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : Clock, clock with :
      reset => (UInt<1>("h0"), r_1)
    node _T = asClock(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = asClock(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module asClock
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

circuit cvt :
  module cvt :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : SInt<5>
    output out_1 : SInt<5>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : SInt<5>, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : SInt<5>, clock with :
      reset => (UInt<1>("h0"), r_1)
    node _T = cvt(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = cvt(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module cvt
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

circuit eq :
  module eq :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<4>
    input in_1 : UInt<4>
    output out_0 : UInt<1>
    output out_1 : UInt<1>
    output out_2 : UInt<1>
    output out_3 : UInt<1>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_3)
    node _T = eq(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = eq(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = eq(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = eq(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module eq
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

circuit neg :
  module neg :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<5>
    output out_1 : UInt<5>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<5>, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : UInt<5>, clock with :
      reset => (UInt<1>("h0"), r_1)
    node _T = sub(UInt<1>("h0"), in)
    node _T_1 = tail(_T, 1)
    r_0 <= _T_1
    out_0 <= r_0
    node _T_2 = sub(UInt<1>("h0"), invalid)
    node _T_3 = tail(_T_2, 1)
    r_1 <= _T_3
    out_1 <= r_1

    ; CHECK-LABEL: module neg
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3


    ; // -----

circuit neq :
  module neq :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<4>
    input in_1 : UInt<4>
    output out_0 : UInt<1>
    output out_1 : UInt<1>
    output out_2 : UInt<1>
    output out_3 : UInt<1>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_3)
    node _T = neq(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = neq(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = neq(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = neq(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module neq
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

circuit or :
  module or :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<1>
    input in_1 : UInt<1>
    output out_0 : UInt<1>
    output out_1 : UInt<1>
    output out_2 : UInt<1>
    output out_3 : UInt<1>

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_3)
    node _T = or(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = or(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = or(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = or(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module or
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

circuit pad :
  module pad :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<1>
    output out_0 : UInt<2>
    output out_1 : UInt<2>

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : UInt<2>, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : UInt<2>, clock with :
      reset => (UInt<1>("h0"), r_1)
    node _T = pad(in, 2)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = pad(invalid, 2)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module pad
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

circuit xor :
  module xor :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<1>
    input in_1 : UInt<1>
    output out_0 : UInt<1>
    output out_1 : UInt<1>
    output out_2 : UInt<1>
    output out_3 : UInt<1>

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), r_3)
    node _T = xor(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = xor(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = xor(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = xor(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module xor
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3
