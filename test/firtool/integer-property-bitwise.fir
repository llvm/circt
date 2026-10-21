; RUN: firtool %s --format=fir --parse-only | FileCheck %s

FIRRTL version 7.0.0
circuit IntegerPropertyBitwise:
  public module IntegerPropertyBitwise:
    output placeholder : UInt<1>
    connect placeholder, UInt<1>(0)

  class Properties:
    output andResult : Integer
    output orResult : Integer
    output notResult : Integer

    ; These operands have different significant widths. In particular, -2 is
    ; sign-extended when combined with the wider positive value 16.
    ; CHECK: firrtl.integer.and
    propassign andResult, integer_and(Integer(-2), Integer(16))

    ; CHECK: firrtl.integer.or
    propassign orResult, integer_or(Integer(-2), Integer(16))

    ; CHECK: firrtl.integer.not
    propassign notResult, integer_not(Integer(0))
