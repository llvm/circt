module {
  func.func @main() {
    %c30 = arith.constant 30 : index
    %c25 = arith.constant 25 : index
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %alloca = memref.alloca() : memref<20x25xi32>
    %alloca_0 = memref.alloca() : memref<30x25xi32>
    %alloca_1 = memref.alloca() : memref<20x30xi32>
    %0 = scf.while (%arg0 = %c0) : (index) -> index {
      %1 = arith.cmpi slt, %arg0, %c20 : index
      scf.condition(%1) %arg0 : index
    } do {
    ^bb0(%arg0: index):
      %1 = arith.addi %arg0, %c1 : index
      %2 = scf.while (%arg1 = %c0) : (index) -> index {
        %4 = arith.cmpi slt, %arg1, %c25 : index
        scf.condition(%4) %arg1 : index
      } do {
      ^bb0(%arg1: index):
        %4 = arith.addi %arg1, %c1 : index
        %5 = memref.load %alloca[%arg0, %arg1] : memref<20x25xi32>
        %6 = arith.muli %5, %c2_i32 : i32
        memref.store %6, %alloca[%arg0, %arg1] : memref<20x25xi32>
        scf.yield %4 : index
      }
      %3 = scf.while (%arg1 = %c0) : (index) -> index {
        %4 = arith.cmpi slt, %arg1, %c30 : index
        scf.condition(%4) %arg1 : index
      } do {
      ^bb0(%arg1: index):
        %4 = arith.addi %arg1, %c1 : index
        %5 = memref.load %alloca_1[%arg0, %arg1] : memref<20x30xi32>
        %6 = arith.muli %5, %c3_i32 : i32
        %7 = scf.while (%arg2 = %c0) : (index) -> index {
          %8 = arith.cmpi slt, %arg2, %c25 : index
          scf.condition(%8) %arg2 : index
        } do {
        ^bb0(%arg2: index):
          %8 = arith.addi %arg2, %c1 : index
          %9 = memref.load %alloca_0[%arg1, %arg2] : memref<30x25xi32>
          %10 = arith.muli %6, %9 : i32
          %11 = memref.load %alloca[%arg0, %arg2] : memref<20x25xi32>
          %12 = arith.addi %11, %10 : i32
          memref.store %12, %alloca[%arg0, %arg2] : memref<20x25xi32>
          scf.yield %8 : index
        }
        scf.yield %4 : index
      }
      scf.yield %1 : index
    }
    return
  }
}

