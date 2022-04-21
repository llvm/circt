; REQUIRES: verilator
; RUN: firtool %s -verilog -o %t1.sv
; RUN: circt-rtl-sim.py %t1.sv --cycles 32 2>&1 | FileCheck %s

circuit top :
  module top :
    input clk: Clock
    input rstn: UInt<1>

    wire rst : UInt<1>
    rst <= not(rstn)

    reg a : UInt<32>, clk with :
      reset => (rst, UInt<1>("h0"))
    reg b : UInt<32>, clk with :
      reset => (rst, UInt<1>("h0"))
    reg c : UInt<32>, clk with :
      reset => (rst, UInt<1>("h0"))

    a <= add(a, UInt<1>("h1"))
    b <= add(b, a)
    c <= add(c, b)

    printf(clk, rstn, "%d %d %d\n", a, b, c)

; CHECK: 0          0          0
; CHECK: 1          0          0
; CHECK: 2          1          0
; CHECK: 3          3          1
; CHECK: 4          6          4
; CHECK: 5         10         10
; CHECK: 6         15         20
; CHECK: 7         21         35
