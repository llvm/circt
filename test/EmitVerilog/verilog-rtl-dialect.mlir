// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

firrtl.circuit "Circuit" {
  firrtl.module @M1(%x : !firrtl.uint<8>,
                    %y : !firrtl.flip<uint<8>>,
                    %z : i8) {
    %c42 = rtl.constant (42 : i8) : i8
    %c5 = rtl.constant (5 : i8) : i8
    %a = rtl.add %z, %c42 : i8
    %b = rtl.mul %a, %c5 : i8
    %c = firrtl.stdIntCast %b : (i8) -> !firrtl.uint<8>
    firrtl.connect %y, %c : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %d = rtl.mul %z, %z : i8
    %e = rtl.mod %d, %c5 : i8
    %f = rtl.concat %e, %z, %d : (i8, i8, i8) -> i8
    %g = firrtl.stdIntCast %f : (i8) -> !firrtl.uint<8>
    firrtl.connect %y, %g : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }

  // CHECK-LABEL: module M1(
  // CHECK-NEXT:    input  [7:0] x,
  // CHECK-NEXT:    output [7:0] y,
  // CHECK-NEXT:    input  [7:0] z);
  // CHECK-EMPTY:
  // CHECK-NEXT:    assign y = (z + 8'h2A) * 8'h5;
  // CHECK-NEXT:    wire [7:0] _T = z * z;
  // CHECK-NEXT:    assign y = {_T % 8'h5, z, _T};
  // CHECK-NEXT:  endmodule


   firrtl.module @M2(%x : i8,
                     %y : i8,
                     %z : i8) {
    %c42 = rtl.constant (42 : i8) : i8
    rtl.connect %x, %c42 : i8

    %w1 = rtl.wire { name = "foo" } : i8
    rtl.connect %w1, %y : i8
    rtl.connect %z, %w1 : i8
  }

  // CHECK-LABEL: module M2(
  // CHECK-NEXT:    input [7:0] x, y, z);
  // CHECK-EMPTY:
  // CHECK-NEXT:    wire [7:0] foo;
  // CHECK-EMPTY:
  // CHECK-NEXT:    assign x = 8'h2A;
  // CHECK-NEXT:    assign foo = y;
  // CHECK-NEXT:    assign z = foo;
  // CHECK-NEXT:  endmodule
}
