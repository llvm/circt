FIRRTL version 4.2.0
circuit Foo:
  public module Foo:

  public module HalfAdder:
    input a : UInt<1>
    input b : UInt<1>
    output s : UInt<1>
    output co : UInt<1>

    contract sum = cat(and(a, b), xor(a, b)):
      intrinsic(circt_verif_ensure, eq(sum, add(a, b)))

    connect s, bits(sum, 0, 0)
    connect co, bits(sum, 1, 1)
