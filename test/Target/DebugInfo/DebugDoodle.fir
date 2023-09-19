; RUN: true

circuit Foo :
  module Foo :
    input a : UInt<32>
    output b : UInt<32>

    inst b0 of Bar
    b0.x <= a

    wire c : UInt<32>
    c <= b0.y

    inst b1 of Bar
    b1.x <= c

    node d = b1.y

    b <= d

  module Bar :
    input x : UInt<32>
    output y : UInt<32>

    y <= tail(mul(x, x), 32)
