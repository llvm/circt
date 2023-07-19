module {
  func.func @main() {
    %alloca_1 = memref.alloca() : memref<40xi32>
    %c0_11 = arith.constant 0 : index
    %c40_12 = arith.constant 40 : index
    %c1_13 = arith.constant 1 : index
    %c2_32 = arith.constant 2 : i32
    scf.for %arg0 = %c0_11 to %c40_12 step %c1_13 {
      %0 = memref.load %alloca_1[%arg0] : memref<40xi32>
      %2 = arith.addi %0, %c2_32 : i32
      memref.store %2, %alloca_1[%arg0] : memref<40xi32>
    }
    return
  }
}
