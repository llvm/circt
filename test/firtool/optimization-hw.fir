; RUN: firtool %s -lower-to-hw | FileCheck %s --check-prefix=OPT

; Ensure that HW-related optimizations cause only a single always block to be
; generated even if LowerToHW generates multiple always blocks.
;
; See: https://github.com/llvm/circt/issues/1594
circuit test_cse_reg_reset :
  module test_cse_reg_reset :
    input clock : Clock
    input reset : UInt<1>
    input a: UInt<1>
    output b : UInt<1>

    wire reset_n: UInt<1>
    reset_n <= not(reset)
    reg r : UInt<1>, clock with: (reset => (reset_n, UInt(0)))
    r <= a
    b <= r

    ; OPT-LABEL: @test_cse_reg_reset
    ; OPT-COUNT-1: sv.alwaysff
    ; OPT-NOT: sv.alwaysff
