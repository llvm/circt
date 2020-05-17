// RUN: cirt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

firrtl.circuit "Circuit" {
  firrtl.module @M1(%x : !firrtl.uint<8> { firrtl.name = "y"},
                    %y : !firrtl.flip<uint<8>>) {
    firrtl.connect %y, %x : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }
  // CHECK-LABEL: module M1(
  // CHECK-NEXT:    input  [7:0] y,
  // CHECK-NEXT:    output [7:0] y_0);
  // CHECK-EMPTY:
  // CHECK-NEXT:    assign y_0 = y;
  // CHECK-NEXT:  endmodule

  firrtl.module @M2(%in : !firrtl.uint<8> { firrtl.name = "some name"},
                    %out : !firrtl.uint<7> { firrtl.name = "88^42"}) {
  }
  // CHECK-LABEL: module M2(
  // CHECK-NEXT:    input [7:0] some_name,
  // CHECK-NEXT:    input [6:0] _885E42);
  // CHECK-EMPTY:
  // CHECK-NEXT:  endmodule

  firrtl.module @M3(%x : !firrtl.uint<8>,
                    %y : !firrtl.flip<uint<8>>,
                    %z : i8) {
    firrtl.connect %y, %x : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %c42_ui8 = firrtl.constant(42 : ui8) : !firrtl.uint<8>
    firrtl.connect %y, %c42_ui8 : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %c42 = constant 42 : i8
    %a = firrtl.stdIntCast %c42 : (i8) -> !firrtl.uint<8>
    firrtl.connect %y, %a : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %b = firrtl.stdIntCast %x : (!firrtl.uint<8>) -> i8
    %c = firrtl.stdIntCast %b : (i8) -> !firrtl.uint<8>
    firrtl.connect %y, %c : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %d = firrtl.stdIntCast %z : (i8) -> !firrtl.uint<8>
    firrtl.connect %y, %d : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }

  // CHECK-LABEL: module M3(
  // CHECK-NEXT:    input  [7:0] x,
  // CHECK-NEXT:    output [7:0] y,
  // CHECK-NEXT:    input  [7:0] z);
  // CHECK-EMPTY:
  // CHECK-NEXT:    assign y = x;
  // CHECK-NEXT:    assign y = 8'h2A;
  // CHECK-NEXT:    assign y = 8'h2A;
  // CHECK-NEXT:    assign y = x;
  // CHECK-NEXT:    assign y = z;
  // CHECK-NEXT:  endmodule
}
