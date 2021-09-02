// RUN: circt-opt %s -split-input-file -test-bitwidth-analysis | FileCheck %s

func @f1(%arg0: index) -> (i8) {
  // CHECK:      %0 = memref.alloc() {"result bits" = [32]} : memref<10xi8>
  // CHECK-NEXT: %c1_i8 = constant  {"result bits" = [1]} 1 : i8
  // CHECK-NEXT: memref.store %c1_i8, %0[%arg0] : memref<10xi8>
  // CHECK-NEXT: %1 = memref.load %0[%arg0] {"result bits" = [8]} : memref<10xi8>
  %0 = memref.alloc() : memref<10xi8>
  %c1 = constant 1 : i8
  memref.store %c1, %0[%arg0] : memref<10xi8>
  %1 = memref.load %0[%arg0] : memref<10xi8>
  return %1 : i8
}

// -----

func @f2() -> index {
  // CHECK:      %c1 = constant  {"result bits" = [1]} 1 : index
  // CHECK-NEXT: %c42 = constant  {"result bits" = [6]} 42 : index
  // CHECK-NEXT: %c63 = constant  {"result bits" = [6]} 63 : index
  // CHECK-NEXT: %c1_0 = constant  {"result bits" = [1]} 1 : index
  %c1 = constant 1 : index
  %c42 = constant 42 : index
  %c63 = constant 63 : index
  %c1_0 = constant 1 : index
  br ^bb1(%c1 : index)
^bb1(%0: index):	// 2 preds: ^bb0, ^bb2
  %1 = cmpi slt, %0, %c42 : index
  // CHECK:      %2 = addi %c63, %c1_0 {"result bits" = [7]} : index
  %2 = addi %c63, %c1_0 : index
  return %2: index
}

// -----

func @f3() -> index {
  // CHECK:       %c1 = constant  {"result bits" = [1]} 1 : index
  // CHECK-NEXT:  %c42 = constant  {"result bits" = [6]} 42 : index
  // CHECK-NEXT:  %c1_0 = constant  {"result bits" = [1]} 1 : index
  %c1 = constant 1 : index
  %c42 = constant 42 : index
  %c1_0 = constant 1 : index
  br ^bb1(%c1 : index)
^bb1(%0: index):	// 2 preds: ^bb0, ^bb2
  // CHECK:    %1 = cmpi slt, %0, %c42 {"result bits" = [1]} : index
  %1 = cmpi slt, %0, %c42 : index
  cond_br %1, ^bb2, ^bb3
^bb2:	// pred: ^bb1
  // CHECK:     %2 = addi %0, %c1_0 {"result bits" = [32]} : index
  %2 = addi %0, %c1_0 : index   // NOTE: Add inside loop with ops defined inside loop; default to type width
  // CHECK:     %3 = addi %c42, %c1_0 {"result bits" = [7]} : index
  %3 = addi %c42, %c1_0 : index // NOTE: Add inside loop, but ops defined outside loop
  br ^bb1(%2 : index)
^bb3:	// pred: ^bb1
  return %0 : index
}

// -----

func @f4() -> i32 {
  // CHECK:       %c1_i32 = constant  {"result bits" = [1]} 1 : i32
  // CHECK-NEXT:  %c42_i32 = constant  {"result bits" = [6]} 42 : i32
  // CHECK-NEXT:  %c1_i32_0 = constant  {"result bits" = [1]} 1 : i32
  %c1 = constant 1 : i32
  %c42 = constant 42 : i32
  %c1_0 = constant 1 : i32
  br ^bb1(%c1 : i32)
^bb1(%0: i32):	// 2 preds: ^bb0, ^bb2
  // CHECK:     %1 = cmpi slt, %0, %c42_i32 {"result bits" = [1]} : i32
  %1 = cmpi slt, %0, %c42 : i32
  cond_br %1, ^bb2, ^bb3
^bb2:	// pred: ^bb1
  // CHECK:     %2 = addi %0, %c1_i32_0 {"result bits" = [32]} : i32
  %2 = addi %0, %c1_0 : i32   // NOTE: Add inside loop with ops defined inside loop; default to type width
  // CHECK:     %3 = addi %c42_i32, %c1_i32_0 {"result bits" = [7]} : i32
  %3 = addi %c42, %c1_0 : i32 // NOTE: Add inside loop, but ops defined outside loop
  br ^bb1(%2 : i32)
^bb3:	// pred: ^bb1
  return %0 : i32
}

// -----

func @f5(%arg0 : i32) -> i32{
  // CHECK:      %c5_i32 = constant  {"result bits" = [3]} 5 : i32
  // CHECK-NEXT: %c42_i32 = constant  {"result bits" = [6]} 42 : i32
  // CHECK-NEXT: %0 = shift_left %arg0, %c5_i32 {"result bits" = [37]} : i32
  // CHECK-NEXT: %1 = shift_right_signed %arg0, %c5_i32 {"result bits" = [27]} : i32
  // CHECK-NEXT: %2 = shift_right_unsigned %arg0, %c5_i32 {"result bits" = [27]} : i32
  // CHECK-NEXT: %3 = shift_right_unsigned %arg0, %c42_i32 {"result bits" = [1]} : i32
  // CHECK-NEXT: %4 = shift_right_signed %arg0, %arg0 {"result bits" = [32]} : i32
  // CHECK-NEXT: %5 = shift_left %arg0, %arg0 {"result bits" = [32]} : i32
  // CHECK-NEXT: %6 = shift_left %arg0, %c42_i32 {"result bits" = [74]} : i32
  %c5 = constant 5 : i32
  %c42 = constant 42 : i32
  %0 = shift_left %arg0, %c5 : i32
  %1 = shift_right_signed %arg0, %c5 : i32
  %2 = shift_right_unsigned %arg0, %c5 : i32 
  %3 = shift_right_unsigned %arg0, %c42 : i32 
  %4 = shift_right_signed %arg0, %arg0 : i32  // rshift by unbounded int; default to lhs width
  %5 = shift_left %arg0, %arg0 : i32          // lshift by unbounded int; default to lhs width
  %6 = shift_left %arg0, %c42 : i32           // lshift by statically determined constant
  return %6 : i32
}
