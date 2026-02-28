; RUN: firtool --preserve-aggregate=1d-vec %s | FileCheck %s

FIRRTL version 5.1.0

; End-to-end test ensuring that 1d vector preservation works correctly with
; lower-layers and the vector subaccess op.

; CHECK: module Foo_Verification_Assert();
; CHECK:   reg  hasBeenResetReg;
; CHECK:   initial
; CHECK:     hasBeenResetReg = 1'bx;
; CHECK:   always @(posedge Foo.clock) begin
; CHECK:     if (Foo.reset)
; CHECK:       hasBeenResetReg <= 1'h1;
; CHECK:   end // always @(posedge)
; CHECK:   wire _GEN = ~(hasBeenResetReg === 1'h1 & Foo.reset === 1'h0);
; CHECK:   assert property (@(posedge Foo.clock) disable iff (_GEN) Foo.in[Foo.addr] == 4'h0);
; CHECK: endmodule

circuit Foo :
  layer Verification, bind, "verification" :
    layer Assert, bind, "verification/assert" :

  public module Foo :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>[4]
    input addr : UInt<2>
    output out : UInt<4>[4]

    connect out[0], in[0]
    connect out[1], in[1]
    connect out[2], in[2]
    connect out[3], in[3]

    node _T = eq(in[addr], UInt<1>(0h0))
    node has_been_reset = intrinsic(circt_has_been_reset : UInt<1>, clock, reset)
    node disable = eq(has_been_reset, UInt<1>(0h0))

    layerblock Verification :
      layerblock Assert :
        node ltl_clock = intrinsic(circt_ltl_clock : UInt<1>, _T, clock)
        node _T_1 = eq(disable, UInt<1>(0h0))
        intrinsic(circt_verif_assert, ltl_clock, _T_1)
