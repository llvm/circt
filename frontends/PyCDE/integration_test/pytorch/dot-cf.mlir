module attributes {torch.debug_module_name = "DotModule"} {
  func.func @forward(%arg0: memref<5xi32, strided<[?], offset: ?>>, %arg1: memref<5xi32, strided<[?], offset: ?>>) -> memref<i32> {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<i32>
    memref.store %c0_i32, %alloc[] : memref<i32>
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %c5 : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %2 = memref.load %arg0[%0] : memref<5xi32, strided<[?], offset: ?>>
    %3 = memref.load %arg1[%0] : memref<5xi32, strided<[?], offset: ?>>
    %4 = memref.load %alloc[] : memref<i32>
    %5 = arith.muli %2, %3 : i32
    %6 = arith.addi %4, %5 : i32
    memref.store %6, %alloc[] : memref<i32>
    %7 = arith.addi %0, %c1 : index
    cf.br ^bb1(%7 : index)
  ^bb3:  // pred: ^bb1
    return %alloc : memref<i32>
  }
}
