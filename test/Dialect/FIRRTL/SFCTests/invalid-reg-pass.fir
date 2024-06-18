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

FIRRTL version 4.0.0
circuit add :
  public module add :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<4>
    input in_1 : UInt<4>
    output out_0 : UInt<5>
    output out_1 : UInt<5>
    output out_2 : UInt<5>
    output out_3 : UInt<5>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = add(in_1, in_0)
    node _T_1 = tail(_T, 1)
    r_0 <= _T_1
    out_0 <= r_0
    node _T_2 = add(in_1, invalid)
    node _T_3 = tail(_T_2, 1)
    r_1 <= _T_3
    out_1 <= r_1
    node _T_4 = add(invalid, in_0)
    node _T_5 = tail(_T_4, 1)
    r_2 <= _T_5
    out_2 <= r_2
    node _T_6 = add(invalid, invalid)
    node _T_7 = tail(_T_6, 1)
    r_3 <= _T_7
    out_3 <= r_3

    ; CHECK-LABEL: module add
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit and :
  public module and :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
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

FIRRTL version 4.0.0
circuit asAsyncReset :
  public module asAsyncReset :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<1>
    output out_0 : AsyncReset
    output out_1 : AsyncReset

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : AsyncReset, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : AsyncReset, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = asAsyncReset(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = asAsyncReset(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module asAsyncReset
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1  <-- fixed; upstream to Scala FIRRTL impl?

    ; // -----

FIRRTL version 4.0.0
circuit asClock :
  public module asClock :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<1>
    output out_0 : Clock
    output out_1 : Clock

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : Clock, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : Clock, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = asClock(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = asClock(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module asClock
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1  <-- fixed; upstream to Scala FIRRTL impl?
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit cvt :
  public module cvt :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : SInt<5>
    output out_1 : SInt<5>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : SInt<5>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : SInt<5>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = cvt(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = cvt(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module cvt
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1  <-- fixed; upstream to Scala FIRRTL impl?
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit eq :
  public module eq :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
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

FIRRTL version 4.0.0
circuit neg :
  public module neg :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<5>
    output out_1 : UInt<5>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = sub(UInt<1>(0h0), in)
    node _T_1 = tail(_T, 1)
    r_0 <= _T_1
    out_0 <= r_0
    node _T_2 = sub(UInt<1>(0h0), invalid)
    node _T_3 = tail(_T_2, 1)
    r_1 <= _T_3
    out_1 <= r_1

    ; CHECK-LABEL: module neg
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3


    ; // -----

FIRRTL version 4.0.0
circuit neq :
  public module neq :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
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

FIRRTL version 4.0.0
circuit or :
  public module or :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
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

FIRRTL version 4.0.0
circuit pad :
  public module pad :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<1>
    output out_0 : UInt<2>
    output out_1 : UInt<2>

    wire invalid : UInt<1>
    invalid is invalid
    reg r_0 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_1)
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

FIRRTL version 4.0.0
circuit xor :
  public module xor :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
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

    ; // -----

FIRRTL version 4.0.0
circuit andr :
  public module andr :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<1>
    output out_1 : UInt<1>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = andr(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = andr(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module andr
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit asSInt :
  public module asSInt :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<2>
    output out_0 : SInt<2>
    output out_1 : SInt<2>

    wire invalid : UInt<2>
    invalid is invalid
    reg r_0 : SInt<2>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : SInt<2>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = asSInt(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = asSInt(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module asSInt
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit asUInt :
  public module asUInt :
    input clock : Clock
    input reset : UInt<1>
    input in : SInt<2>
    output out_0 : UInt<2>
    output out_1 : UInt<2>

    wire invalid : SInt<2>
    invalid is invalid
    reg r_0 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = asUInt(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = asUInt(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module asUInt
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit bits :
  public module bits :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<2>
    output out_1 : UInt<2>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = bits(in, 3, 2)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = bits(invalid, 3, 2)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module bits
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit cat :
  public module cat :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<2>
    input in_1 : UInt<2>
    output out_0 : UInt<4>
    output out_1 : UInt<4>
    output out_2 : UInt<4>
    output out_3 : UInt<4>

    wire invalid : UInt<2>
    invalid is invalid
    reg r_0 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = cat(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = cat(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = cat(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = cat(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module cat
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit dshl :
  public module dshl :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<2>
    input in_1 : UInt<2>
    output out_0 : UInt<5>
    output out_1 : UInt<5>
    output out_2 : UInt<5>
    output out_3 : UInt<5>

    wire invalid : UInt<2>
    invalid is invalid
    reg r_0 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = dshl(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = dshl(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = dshl(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = dshl(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module dshl
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit dshr :
  public module dshr :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<2>
    input in_1 : UInt<2>
    output out_0 : UInt<2>
    output out_1 : UInt<2>
    output out_2 : UInt<2>
    output out_3 : UInt<2>

    wire invalid : UInt<2>
    invalid is invalid
    reg r_0 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = dshr(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = dshr(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = dshr(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = dshr(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module dshr
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit head :
  public module head :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<2>
    output out_1 : UInt<2>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = head(in, 2)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = head(invalid, 2)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module head
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit lt :
  public module lt :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = lt(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = lt(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = lt(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = lt(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module lt
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit gt :
  public module gt :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = gt(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = gt(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = gt(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = gt(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module gt
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit leq :
  public module leq :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = leq(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = leq(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = leq(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = leq(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module leq
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit geq :
  public module geq :
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
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = geq(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = geq(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = geq(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = geq(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module geq
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit mul :
  public module mul :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<4>
    input in_1 : UInt<4>
    output out_0 : UInt<8>
    output out_1 : UInt<8>
    output out_2 : UInt<8>
    output out_3 : UInt<8>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<8>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<8>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<8>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<8>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = mul(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = mul(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = mul(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = mul(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module mul
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1  <-- fixed; upstream to Scala FIRRTL impl?
    ; CHECK-NOT:     r_2  <-- fixed; upstream to Scala FIRRTL impl?
    ; CHECK-NOT:     r_3  <-- fixed; upstream to Scala FIRRTL impl?

    ; // -----

FIRRTL version 4.0.0
circuit not :
  public module not :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<4>
    output out_1 : UInt<4>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = not(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = not(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module not
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit orr :
  public module orr :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<1>
    output out_1 : UInt<1>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = orr(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = orr(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module orr
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit shl :
  public module shl :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<2>
    output out_0 : UInt<4>
    output out_1 : UInt<4>

    wire invalid : UInt<2>
    invalid is invalid
    reg r_0 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = shl(in, 2)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = shl(invalid, 2)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module shl
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit shr :
  public module shr :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<2>
    output out_1 : UInt<2>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = shr(in, 2)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = shr(invalid, 2)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module shr
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit sub :
  public module sub :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<4>
    input in_1 : UInt<4>
    output out_0 : UInt<5>
    output out_1 : UInt<5>
    output out_2 : UInt<5>
    output out_3 : UInt<5>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = sub(in_1, in_0)
    node _T_1 = tail(_T, 1)
    r_0 <= _T_1
    out_0 <= r_0
    node _T_2 = sub(in_1, invalid)
    node _T_3 = tail(_T_2, 1)
    r_1 <= _T_3
    out_1 <= r_1
    node _T_4 = sub(invalid, in_0)
    node _T_5 = tail(_T_4, 1)
    r_2 <= _T_5
    out_2 <= r_2
    node _T_6 = sub(invalid, invalid)
    node _T_7 = tail(_T_6, 1)
    r_3 <= _T_7
    out_3 <= r_3

    ; CHECK-LABEL: module sub
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK:         r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit tail :
  public module tail :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<2>
    output out_1 : UInt<2>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<2>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = tail(in, 2)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = tail(invalid, 2)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module tail
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit div :
  public module div :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<4>
    input in_1 : UInt<4>
    output out_0 : UInt<4>
    output out_1 : UInt<4>
    output out_2 : UInt<4>
    output out_3 : UInt<4>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = div(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = div(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = div(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = div(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module div
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK-NOT:     r_2  <-- fixed; upstream to Scala FIRRTL impl?
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit rem :
  public module rem :
    input clock : Clock
    input reset : UInt<1>
    input in_0 : UInt<4>
    input in_1 : UInt<4>
    output out_0 : UInt<4>
    output out_1 : UInt<4>
    output out_2 : UInt<4>
    output out_3 : UInt<4>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_1)
    reg r_2 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_2)
    reg r_3 : UInt<4>, clock with :
      reset => (UInt<1>(0h0), r_3)
    node _T = rem(in_1, in_0)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = rem(in_1, invalid)
    r_1 <= _T_1
    out_1 <= r_1
    node _T_2 = rem(invalid, in_0)
    r_2 <= _T_2
    out_2 <= r_2
    node _T_3 = rem(invalid, invalid)
    r_3 <= _T_3
    out_3 <= r_3

    ; CHECK-LABEL: module rem
    ; CHECK:         r_0
    ; CHECK:         r_1
    ; CHECK-NOT:     r_2  <-- fixed; upstream to Scala FIRRTL impl?
    ; CHECK-NOT:     r_3

    ; // -----

FIRRTL version 4.0.0
circuit xorr :
  public module xorr :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out_0 : UInt<1>
    output out_1 : UInt<1>

    wire invalid : UInt<4>
    invalid is invalid
    reg r_0 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_0)
    reg r_1 : UInt<1>, clock with :
      reset => (UInt<1>(0h0), r_1)
    node _T = xorr(in)
    r_0 <= _T
    out_0 <= r_0
    node _T_1 = xorr(invalid)
    r_1 <= _T_1
    out_1 <= r_1

    ; CHECK-LABEL: module xorr
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3
