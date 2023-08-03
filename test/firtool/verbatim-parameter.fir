; RUN: firtool %s | FileCheck %s

FIRRTL version 3.1.0

circuit Top :
  ; CHECK-LABEL module Top
  module Top :
    output o : UInt<8>
    ; CHECK:       Blah #(
    ; CHECK-NEXT:    .foo(test test test 123)
    ; CHECK-NEXT:  ) e (
    ; CHECK-NEXT:    .o (o)
    ; CHECK-NEXT:  )
    inst e of MyExt
    connect o, e.o

  extmodule MyExt :
    output o : UInt<8>
    defname = Blah
    parameter foo = 'test test test 123'
