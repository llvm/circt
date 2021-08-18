// RUN: circt-opt -convert-fsm-to-standard %s | FileCheck %s

// CHECK: func @foo(%arg0: i1, %arg1: i8, %arg2: memref<i8>, %arg3: memref<i2>) -> i1 {
// CHECK:   %0 = memref.load %arg2[] : memref<i8>
// CHECK:   %c0_i2 = constant 0 : i2
// CHECK:   %c1_i2 = constant 1 : i2
// CHECK:   %1 = memref.load %arg3[] : memref<i2>
// CHECK:   %2 = cmpi eq, %1, %c0_i2 : i2
// CHECK:   scf.if %2 {
// CHECK:     scf.if %arg0 {
// CHECK:       memref.store %c1_i2, %arg3[] : memref<i2>
// CHECK:       memref.store %arg1, %arg2[] : memref<i8>
// CHECK:     } else {
// CHECK:       memref.store %c0_i2, %arg3[] : memref<i2>
// CHECK:     }
// CHECK:   } else {
// CHECK:     %6 = cmpi eq, %1, %c1_i2 : i2
// CHECK:     scf.if %6 {
// CHECK:       memref.store %c1_i2, %arg3[] : memref<i2>
// CHECK:     } else {
// CHECK:       %true = constant true
// CHECK:       assert %true, "invalid state"
// CHECK:     }
// CHECK:   }
// CHECK:   %3 = memref.load %arg3[] : memref<i2>
// CHECK:   %4 = cmpi eq, %3, %c0_i2 : i2
// CHECK:   %5 = scf.if %4 -> (i1) {
// CHECK:     %true = constant true
// CHECK:     scf.yield %true : i1
// CHECK:   } else {
// CHECK:     %6 = cmpi eq, %3, %c1_i2 : i2
// CHECK:     %7 = scf.if %6 -> (i1) {
// CHECK:       %false = constant false
// CHECK:       scf.yield %false : i1
// CHECK:     } else {
// CHECK:       %true = constant true
// CHECK:       assert %true, "invalid state"
// CHECK:       %false = constant false
// CHECK:       scf.yield %false : i1
// CHECK:     }
// CHECK:     scf.yield %7 : i1
// CHECK:   }
// CHECK:   return %5 : i1
// CHECK: }
// CHECK: func @bar() {
// CHECK:   %0 = memref.alloca() : memref<i8>
// CHECK:   %c0_i8 = constant 0 : i8
// CHECK:   memref.store %c0_i8, %0[] : memref<i8>
// CHECK:   %1 = memref.alloca() : memref<i2>
// CHECK:   %c0_i2 = constant 0 : i2
// CHECK:   memref.store %c0_i2, %1[] : memref<i2>
// CHECK:   %true = constant true
// CHECK:   %c16_i8 = constant 16 : i8
// CHECK:   %2 = call @foo(%true, %c16_i8, %0, %1) : (i1, i8, memref<i8>, memref<i2>) -> i1
// CHECK:   %false = constant false
// CHECK:   %c0_i8_0 = constant 0 : i8
// CHECK:   %3 = call @foo(%false, %c0_i8_0, %0, %1) : (i1, i8, memref<i8>, memref<i2>) -> i1
// CHECK:   return
// CHECK: }

fsm.machine @foo(%i_valid: i1, %i_len: i8) -> i1 attributes {stateType = i2} {
  %counter = fsm.variable "counter" {initValue = 0 : i8} : i8

  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY guard  {
      fsm.return %i_valid : i1
    } action  {
      fsm.update %counter, %i_len : i8
    }
    fsm.transition @IDLE guard  {
    } action  {
    }
  }

  fsm.state "BUSY" output  {
    %false = constant false
    fsm.output %false : i1
  } transitions  {
    fsm.transition @BUSY guard  {
    } action  {
    }
  }
}

func @bar() {
  %foo_inst = fsm.instance "foo_inst" @foo

  %i_valid0 = constant true
  %i_len0 = constant 16 : i8
  %o_ready0 = fsm.trigger %foo_inst(%i_valid0, %i_len0) : (i1, i8) -> i1

  %i_valid1 = constant false
  %i_len1 = constant 0 : i8
  %o_ready1 = fsm.trigger %foo_inst(%i_valid1, %i_len1) : (i1, i8) -> i1
  return
}
