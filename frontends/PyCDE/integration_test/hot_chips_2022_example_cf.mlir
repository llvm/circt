#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
module attributes {torch.debug_module_name = "DotModule"} {
  func.func @forward(%arg0: memref<5xi32, #map>, %arg1: memref<5xi32, #map>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0, %c0_i32 : index, i32)
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %0, %c5 : index
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %3 = memref.load %arg0[%0] : memref<5xi32, #map>
    %4 = memref.load %arg1[%0] : memref<5xi32, #map>
    %5 = arith.muli %3, %4 : i32
    %6 = arith.addi %1, %5 : i32
    %7 = arith.addi %0, %c1 : index
    cf.br ^bb1(%7, %6 : index, i32)
  ^bb3:  // pred: ^bb1
    return %1 : i32
  }
}

