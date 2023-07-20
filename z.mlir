module {
  func.func @main() {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %alloca = memref.alloca() : memref<20x25xi32>
    %alloca_0 = memref.alloca() : memref<30x25xi32>
    %alloca_1 = memref.alloca() : memref<20x30xi32>
    %c0 = arith.constant 0 : index
    %c20 = arith.constant 20 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c20 step %c1 {
      %c0_2 = arith.constant 0 : index
      %c25 = arith.constant 25 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg1 = %c0_2 to %c25 step %c1_3 {
        %0 = memref.load %alloca[%arg0, %arg1] : memref<20x25xi32>
        %1 = arith.muli %0, %c2_i32 : i32
        memref.store %1, %alloca[%arg0, %arg1] : memref<20x25xi32>
      }
      %c0_4 = arith.constant 0 : index
      %c30 = arith.constant 30 : index
      %c1_5 = arith.constant 1 : index
      scf.for %arg1 = %c0_4 to %c30 step %c1_5 {
        %0 = memref.load %alloca_1[%arg0, %arg1] : memref<20x30xi32>
        %1 = arith.muli %0, %c3_i32 : i32
        %c0_6 = arith.constant 0 : index
        %c25_7 = arith.constant 25 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg2 = %c0_6 to %c25_7 step %c1_8 {
          %2 = memref.load %alloca_0[%arg1, %arg2] : memref<30x25xi32>
          %3 = arith.muli %1, %2 : i32
          %4 = memref.load %alloca[%arg0, %arg2] : memref<20x25xi32>
          %5 = arith.addi %4, %3 : i32
          memref.store %5, %alloca[%arg0, %arg2] : memref<20x25xi32>
        }
      }
    }
    return
  }
}
