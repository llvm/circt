; REQUIRES: yosys
; RUN: split-file %s %t
; RUN: firtool %t/test_mod.fir --format=fir -verilog -o %t/test_mod.fir.v
; RUN: /home/andrew/src/circt/equiv_yosys.sh %t/test_mod.fir.v %t/test_mod.v test_mod

;--- test_mod.fir

circuit test_mod :
  module test_mod :
    input a: UInt<1>
    output b: UInt<1>
    b <= a

;--- test_mod.v

module test_mod(
  input   a,
  output  b
);
  assign b = a;
endmodule

