// RUN: circt-translate %s -emit-firrtl-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

firrtl.circuit "M1" {
  firrtl.module @M1(%x : !firrtl.uint<8> { firrtl.name = "y"},
                    %y : !firrtl.flip<uint<8>>) {
    firrtl.connect %y, %x : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %a = firrtl.asPassive %y : (!firrtl.flip<uint<8>>) -> !firrtl.uint<8>
    %b = firrtl.not %a : (!firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.connect %y, %b : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }

  // CHECK-LABEL: module M1(
  // CHECK-NEXT:    input  [7:0] y,
  // CHECK-NEXT:    output [7:0] y_0);
  // CHECK-EMPTY:
  // CHECK-NEXT:    assign y_0 = y;
  // CHECK-NEXT:    assign y_0 = ~y_0;
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
                    %y : !firrtl.flip<uint<8>>) {
    firrtl.connect %y, %x : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %c42_ui8 = firrtl.constant(42 : ui8) : !firrtl.uint<8>
    firrtl.connect %y, %c42_ui8 : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %b = firrtl.stdIntCast %x : (!firrtl.uint<8>) -> i8
    %c = firrtl.stdIntCast %b : (i8) -> !firrtl.uint<8>
    firrtl.connect %y, %c : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %d = firrtl.stdIntCast %b : (i8) -> !firrtl.uint<8>
    firrtl.connect %y, %d : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }

  // CHECK-LABEL: module M3(
  // CHECK-NEXT:    input  [7:0] x,
  // CHECK-NEXT:    output [7:0] y);
  // CHECK-EMPTY:
  // CHECK-NEXT:    assign y = x;
  // CHECK-NEXT:    assign y = 8'h2A;
  // CHECK-NEXT:    wire [7:0] _T = x;
  // CHECK-NEXT:    assign y = _T;
  // CHECK-NEXT:    assign y = _T;
  // CHECK-NEXT:  endmodule
}
