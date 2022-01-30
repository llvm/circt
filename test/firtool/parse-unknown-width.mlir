// RUN: firtool %s --format=mlir --ir-fir    | circt-opt | FileCheck %s --check-prefix=MLIR

firrtl.circuit "Example" {
  firrtl.module @Example(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %in: !firrtl.uint<2>, out %out: !firrtl.uint<2>) {
    %_GEN_0 = firrtl.add %in, %in : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint
    %_out_T = firrtl.node %_GEN_0 : !firrtl.uint
    %_GEN_1 = firrtl.add %in, %in : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint
    %_GEN_2 = firrtl.tail %_out_T, 1 : (!firrtl.uint) -> !firrtl.uint
    %_out_T_1 = firrtl.node %_GEN_2 : !firrtl.uint
    %_GEN_3 = firrtl.tail %_out_T, 1 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %_out_T_1 : !firrtl.uint<2>, !firrtl.uint
  }
}

// MLIR-LABEL: module  {
// MLIR-NEXT :   firrtl.circuit "Example"   {
// MLIR-NEXT :     firrtl.module @Example(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %in: !firrtl.uint<2>, out %out: !firrtl.uint<2>) {
// MLIR-NEXT :       %_out_T = firrtl.add %in, %in {name = "_out_T"} : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<3>
// MLIR-NEXT :       %_out_T_1 = firrtl.bits %_out_T 1 to 0 {name = "_out_T_1"} : (!firrtl.uint<3>) -> !firrtl.uint<2>
// MLIR-NEXT :       firrtl.connect %out, %_out_T_1 : !firrtl.uint<2>, !firrtl.uint<2>
// MLIR-NEXT :     }
// MLIR-NEXT :   }
// MLIR-NEXT : }
