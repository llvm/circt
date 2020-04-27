// RUN: spt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

firrtl.circuit "Circuit" {
  firrtl.module @M1(%x : !firrtl.uint<8> { firrtl.name = "y"},
                    %y : !firrtl.flip<uint<8>>) {
    firrtl.connect %y, %x : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }
  // CHECK: module M1(
  // CHECK:   input  [7:0] y,
  // CHECK:   output [7:0] y_0);
  // CHECK:   assign y_0 = y;
  // CHECK: endmodule

  firrtl.module @M2(%in : !firrtl.uint<8> { firrtl.name = "some name"},
                    %out : !firrtl.uint<7> { firrtl.name = "88^42"}) {
  }
  // CHECK: module M2(
  // CHECK:   input [7:0] some_name,
  // CHECK:   input [6:0] _885E42);
  // CHECK: endmodule
}
