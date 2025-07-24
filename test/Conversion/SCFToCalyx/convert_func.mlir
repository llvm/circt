// RUN: circt-opt %s --lower-scf-to-calyx="top-level-function=main" -canonicalize -split-input-file | FileCheck %s

// CHECK:         calyx.invoke @func_instance[](%[[VAL_0:.*]] = %[[VAL_1:.*]]) -> (i32)

module {
  func.func @func(%0 : i32) -> i32 {
    return %0 : i32
  }

  func.func @main() -> i32 {
    %0 = arith.constant 0 : i32
    %1 = func.call @func(%0) : (i32) -> i32 
    func.return %1 : i32
  } 
}

// -----

// CHECK:         calyx.invoke @func_instance[](%[[VAL_0:.*]] = %[[VAL_1:.*]]) -> (i32)

module {
  func.func @func(%0 : i32) -> i32 {
    return %0 : i32
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>  
    scf.while(%arg0 = %c0) : (index) -> (index) {
      %cond = arith.cmpi ult, %arg0, %c64 : index
      scf.condition(%cond) %arg0 : index
    } do {
    ^bb0(%arg1: index):
      %v = memref.load %0[%arg1] : memref<64xi32>
      %c = func.call @func(%v) : (i32) -> i32
      memref.store %c, %1[%arg1] : memref<64xi32>
      %inc = arith.addi %arg1, %c1 : index
      scf.yield %inc : index
    }
    return
  }
}

// -----

// CHECK:         calyx.invoke @fun_instance[](%[[VAL_0:.*]] = %[[VAL_1:.*]]) -> (i32)

module {
  func.func @fun(%0 : i32) -> i32 {
    return %0 : i32
  }

  func.func @main() {
    %alloca = memref.alloca() : memref<40xi32>
    %c0 = arith.constant 0 : index
    %c40 = arith.constant 40 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c40 step %c1 {
      %0 = memref.load %alloca[%arg0] : memref<40xi32>
      %1 = func.call @fun(%0) : (i32) -> i32 
      memref.store %1, %alloca[%arg0] : memref<40xi32>
    }
    return
  }
}

// -----

// CHECK:         calyx.invoke @func_instance[](%[[VAL_0:.*]] = %[[VAL_1:.*]]) -> (i32)
// CHECK:         calyx.invoke @func_instance[](%[[VAL_2:.*]] = %[[VAL_3:.*]]) -> (i32)

module {
  func.func @func(%0 : i32) -> i32 { 
    return %0 : i32
  }

  func.func @main(%a0 : i32, %a1 : i32, %a2 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.addi %0, %a1 : i32
    %b = arith.cmpi uge, %1, %a2 : i32
    cf.cond_br %b, ^bb1, ^bb2
  ^bb1:
    %ret0 = func.call @func(%0) : (i32) -> i32
    return %ret0 : i32
  ^bb2:
    %ret1 = func.call @func(%1) : (i32) -> i32
    return %ret1 : i32
  }
}

// -----

// Test non-top-level function has external memory allocation.

// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.invoke @callee_instance[arg_mem_0 = mem_0, arg_mem_1 = mem_1]() -> ()
// CHECK:             }
// CHECK:           }

module {
  func.func @callee(%arg0 : memref<5xi32>) {
    %idx = arith.constant 0 : index
    %alloc = memref.alloc() : memref<6xi32>
    %val = memref.load %alloc[%idx] : memref<6xi32>
    memref.store %val, %arg0[%idx] : memref<5xi32>
    return
  }
  func.func @main(%arg0: memref<5xi32>) {
    func.call @callee(%arg0) : (memref<5xi32>) -> ()
    return
  }
}
