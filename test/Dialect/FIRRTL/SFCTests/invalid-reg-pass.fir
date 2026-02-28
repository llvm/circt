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
    invalidate invalid
    reg r_0 : UInt<5>, clock
    reg r_1 : UInt<5>, clock
    reg r_2 : UInt<5>, clock
    reg r_3 : UInt<5>, clock
    node _T = add(in_1, in_0)
    node _T_1 = tail(_T, 1)
    connect r_0, _T_1
    connect out_0, r_0
    node _T_2 = add(in_1, invalid)
    node _T_3 = tail(_T_2, 1)
    connect r_1, _T_3
    connect out_1, r_1
    node _T_4 = add(invalid, in_0)
    node _T_5 = tail(_T_4, 1)
    connect r_2, _T_5
    connect out_2, r_2
    node _T_6 = add(invalid, invalid)
    node _T_7 = tail(_T_6, 1)
    connect r_3, _T_7
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = and(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = and(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = and(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = and(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : AsyncReset, clock
    reg r_1 : AsyncReset, clock
    node _T = asAsyncReset(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = asAsyncReset(invalid)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : Clock, clock
    reg r_1 : Clock, clock
    node _T = asClock(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = asClock(invalid)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : SInt<5>, clock
    reg r_1 : SInt<5>, clock
    node _T = cvt(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = cvt(invalid)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = eq(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = eq(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = eq(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = eq(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<5>, clock
    reg r_1 : UInt<5>, clock
    node _T = sub(UInt<1>(0h0), in)
    node _T_1 = tail(_T, 1)
    connect r_0, _T_1
    connect out_0, r_0
    node _T_2 = sub(UInt<1>(0h0), invalid)
    node _T_3 = tail(_T_2, 1)
    connect r_1, _T_3
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = neq(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = neq(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = neq(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = neq(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = or(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = or(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = or(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = or(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<2>, clock
    reg r_1 : UInt<2>, clock
    node _T = pad(in, 2)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = pad(invalid, 2)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = xor(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = xor(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = xor(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = xor(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    node _T = andr(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = andr(invalid)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : SInt<2>, clock
    reg r_1 : SInt<2>, clock
    node _T = asSInt(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = asSInt(invalid)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<2>, clock
    reg r_1 : UInt<2>, clock
    node _T = asUInt(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = asUInt(invalid)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<2>, clock
    reg r_1 : UInt<2>, clock
    node _T = bits(in, 3, 2)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = bits(invalid, 3, 2)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<4>, clock
    reg r_1 : UInt<4>, clock
    reg r_2 : UInt<4>, clock
    reg r_3 : UInt<4>, clock
    node _T = cat(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = cat(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = cat(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = cat(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<5>, clock
    reg r_1 : UInt<5>, clock
    reg r_2 : UInt<5>, clock
    reg r_3 : UInt<5>, clock
    node _T = dshl(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = dshl(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = dshl(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = dshl(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<2>, clock
    reg r_1 : UInt<2>, clock
    reg r_2 : UInt<2>, clock
    reg r_3 : UInt<2>, clock
    node _T = dshr(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = dshr(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = dshr(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = dshr(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<2>, clock
    reg r_1 : UInt<2>, clock
    node _T = head(in, 2)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = head(invalid, 2)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = lt(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = lt(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = lt(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = lt(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = gt(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = gt(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = gt(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = gt(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = leq(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = leq(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = leq(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = leq(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    reg r_2 : UInt<1>, clock
    reg r_3 : UInt<1>, clock
    node _T = geq(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = geq(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = geq(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = geq(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<8>, clock
    reg r_1 : UInt<8>, clock
    reg r_2 : UInt<8>, clock
    reg r_3 : UInt<8>, clock
    node _T = mul(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = mul(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = mul(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = mul(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<4>, clock
    reg r_1 : UInt<4>, clock
    node _T = not(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = not(invalid)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    node _T = orr(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = orr(invalid)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<4>, clock
    reg r_1 : UInt<4>, clock
    node _T = shl(in, 2)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = shl(invalid, 2)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<2>, clock
    reg r_1 : UInt<2>, clock
    node _T = shr(in, 2)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = shr(invalid, 2)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<5>, clock
    reg r_1 : UInt<5>, clock
    reg r_2 : UInt<5>, clock
    reg r_3 : UInt<5>, clock
    node _T = sub(in_1, in_0)
    node _T_1 = tail(_T, 1)
    connect r_0, _T_1
    connect out_0, r_0
    node _T_2 = sub(in_1, invalid)
    node _T_3 = tail(_T_2, 1)
    connect r_1, _T_3
    connect out_1, r_1
    node _T_4 = sub(invalid, in_0)
    node _T_5 = tail(_T_4, 1)
    connect r_2, _T_5
    connect out_2, r_2
    node _T_6 = sub(invalid, invalid)
    node _T_7 = tail(_T_6, 1)
    connect r_3, _T_7
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<2>, clock
    reg r_1 : UInt<2>, clock
    node _T = tail(in, 2)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = tail(invalid, 2)
    connect r_1, _T_1
    connect out_1, r_1

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
    invalidate invalid
    reg r_0 : UInt<4>, clock
    reg r_1 : UInt<4>, clock
    reg r_2 : UInt<4>, clock
    reg r_3 : UInt<4>, clock
    node _T = div(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = div(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = div(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = div(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<4>, clock
    reg r_1 : UInt<4>, clock
    reg r_2 : UInt<4>, clock
    reg r_3 : UInt<4>, clock
    node _T = rem(in_1, in_0)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = rem(in_1, invalid)
    connect r_1, _T_1
    connect out_1, r_1
    node _T_2 = rem(invalid, in_0)
    connect r_2, _T_2
    connect out_2, r_2
    node _T_3 = rem(invalid, invalid)
    connect r_3, _T_3
    connect out_3, r_3

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
    invalidate invalid
    reg r_0 : UInt<1>, clock
    reg r_1 : UInt<1>, clock
    node _T = xorr(in)
    connect r_0, _T
    connect out_0, r_0
    node _T_1 = xorr(invalid)
    connect r_1, _T_1
    connect out_1, r_1

    ; CHECK-LABEL: module xorr
    ; CHECK:         r_0
    ; CHECK-NOT:     r_1
    ; CHECK-NOT:     r_2
    ; CHECK-NOT:     r_3
