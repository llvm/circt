// RUN: firtool %s --format=mlir -mlir    | spt-opt | FileCheck %s --check-prefix=MLIR
// RUN: firtool %s --format=mlir -verilog |           FileCheck %s --check-prefix=VERILOG

firrtl.circuit "Top" {
  firrtl.module @MyModule(%in : !firrtl.uint<8>,
                          %out : !firrtl.flip<uint<8>>) {
    firrtl.connect %out, %in : !firrtl.flip<uint<8>>, !firrtl.uint<8>
  }
}

// MLIR-LABEL: firrtl.module @MyModule(%in: !firrtl.uint<8>, %out: !firrtl.flip<uint<8>>) {
// MLIR-NEXT:    firrtl.connect %out, %in : !firrtl.flip<uint<8>>, !firrtl.uint<8>
// MLIR-NEXT:  }

// VERILOG-LABEL: module MyModule(
// VERILOG-NEXT :   input  [7:0] in,
// VERILOG-NEXT :   output [7:0] out);
// VERILOG-NEXT :   assign out = in;
// VERILOG-NEXT : endmodule
