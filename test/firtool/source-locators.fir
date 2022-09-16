; RUN: firtool -O release -lowering-options=wireSpillingHeuristic=spillConsecutiveNamedLines %s | FileCheck %s

circuit Top :
  module Top :
    input clock : Clock
    input reset : UInt<1>
    input in0 : UInt<8>
    input in1 : UInt<8>
    input in2 : UInt<8>
    output out : UInt<8>
    node _inner0_T = add(in0, in1) @[example.scala 15:20]
    node _inner0_T_1 = tail(_inner0_T, 1) @[example.scala 15:20]
    node _inner0_T_2 = add(in0, in2) @[example.scala 15:32]
    node _inner0_T_3 = tail(_inner0_T_2, 1) @[example.scala 15:32]

    ; CHECK: inner0 = in0 + in1 | in0 + in2;
    node inner0 = or(_inner0_T_1, _inner0_T_3) @[example.scala 15:26]

    ; CHECK: inner1 = in2 ^ in1;
    node inner1 = xor(in2, in1) @[example.scala 16:20]

    ; CHECK: inner2 = {255'h0, inner0} << inner1;
    node inner2 = dshl(inner0, inner1) @[example.scala 17:23]

    out <= inner2 @[example.scala 19:7]
