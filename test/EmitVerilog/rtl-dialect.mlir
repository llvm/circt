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

    %d = rtl.mul %z, %z, %z : i8
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
  // CHECK-NEXT:    wire [7:0] _T = z * z * z;
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

  firrtl.module @M3(%x : !firrtl.uint<8>,
                    %y : !firrtl.flip<uint<8>>,
                    %z : i8, %q : i16) {
    %c42 = rtl.constant (42 : i8) : i8
    %c5 = rtl.constant (5 : i8) : i8
    %ext_z = rtl.extract %q from 8 : (i16) -> i8
    %a = rtl.add %z, %c42 : i8
    %v1 = rtl.and %a, %c42, %c5 : i8
    %v2 = rtl.or %a, %v1 : i8
    %v3 = rtl.xor %v1, %v2, %c42, %ext_z : i8
    %c = firrtl.stdIntCast %v3 : (i8) -> !firrtl.uint<8>
    firrtl.connect %y, %c : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }

  // CHECK-LABEL: module M3(
  // CHECK-NEXT:  input  [7:0]  x,
  // CHECK-NEXT:  output [7:0]  y,
  // CHECK-NEXT:  input  [7:0]  z,
  // CHECK-NEXT:  input  [15:0] q);
  // CHECK-EMPTY:
  // CHECK-NEXT:  wire [7:0] _T = z + 8'h2A;
  // CHECK-NEXT:  wire [7:0] _T_0 = _T & 8'h2A & 8'h5;
  // CHECK-NEXT:  assign y = _T_0 ^ (_T | _T_0) ^ 8'h2A ^ q[15:8];
  // CHECK-NEXT:endmodule

  // The "_T" value is singly used, but Verilog can't bit extract out of a not,
  // so an explicit temporary is required.

  // CHECK-LABEL: module ExtractCrash(
  // CHECK: wire [3:0] _T = ~a;
  // CHECK-NEXT: assign b = _T[3:2];
  firrtl.module @ExtractCrash(%a: !firrtl.uint<4>, %b: !firrtl.flip<uint<1>>) {
    %26 = firrtl.not %a : (!firrtl.uint<4>) -> !firrtl.uint<4>
    %27 = firrtl.stdIntCast %26 : (!firrtl.uint<4>) -> i4
    %28 = rtl.extract %27 from 2 : (i4) -> i2
    %29 = firrtl.stdIntCast %b : (!firrtl.flip<uint<1>>) -> i1
    %30 = firrtl.stdIntCast %28 : (i2) -> !firrtl.uint<2>
    firrtl.connect %b, %30 : !firrtl.flip<uint<1>>, !firrtl.uint<2>
  }
}


