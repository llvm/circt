// RUN: cirt-translate -emit-verilog -verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false %s

func @foo() { // expected-error {{unknown operation}}
}

// -----

firrtl.circuit "Top" {
  // expected-error @+1 {{value has an unsupported verilog type '!firrtl.uint'}}
  firrtl.module @Top(%out: !firrtl.uint) {
  }
}

// -----

firrtl.circuit "Top" {

  firrtl.module @M3(%x : !firrtl.uint<8>,
                    %y : !firrtl.flip<uint<8>>) {
    firrtl.connect %y, %x : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %c42_ui8 = firrtl.constant(42 : ui8) : !firrtl.uint<8>
    firrtl.connect %y, %c42_ui8 : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %c42 = constant 42 : i8  // expected-error {{cannot}}
    %a = firrtl.stdIntCast %c42 : (i8) -> !firrtl.uint<8> // expected-error {{cannot}}
    firrtl.connect %y, %a : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    %b = firrtl.stdIntCast %x : (!firrtl.uint<8>) -> i8// expected-error {{cannot}}
    %c = firrtl.stdIntCast %b : (i8) -> !firrtl.uint<8>// expected-error {{cannot}}
    firrtl.connect %y, %c : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }

  // CHEC-LABEL: module M3(
  // CHEC-NEXT:    input  [7:0] x,
  // CHEC-NEXT:    output [7:0] y);
  // CHEC-EMPTY:
  // CHEC-NEXT:    assign y = x;
  // CHEC-NEXT:    assign y = 8'h2A;
  // CHEC-NEXT:  endmodule
}
