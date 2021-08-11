// RUN: circt-opt -convert-fsm-to-standard %s | FileCheck %s

// CHECK: func @foo([[ARG0:%.+]]: i1, [[ARG1:%.+]]: i8, [[ARG2:%.+]]: memref<i1>, [[ARG3:%.+]]: memref<i8>, [[ARG4:%.+]]: memref<i2>) -> i1 {
// CHECK:   [[VAL0:%.+]] = memref.load [[ARG2]][] : memref<i1>
// CHECK:   [[VAL1:%.+]] = memref.load [[ARG3]][] : memref<i8>
// CHECK:   %true = constant true
// CHECK:   %false = constant false
// CHECK:   %c0_i2 = constant 0 : i2
// CHECK:   %c1_i2 = constant 1 : i2
// CHECK:   [[VAL2:%.+]] = memref.load [[ARG4]][] : memref<i2>
// CHECK:   [[VAL3:%.+]] = cmpi eq, [[VAL2]], %c0_i2 : i2
// CHECK:   scf.if [[VAL3]] {
// CHECK:     scf.if [[ARG0]] {
// CHECK:       memref.store %c1_i2, [[ARG4]][] : memref<i2>
// CHECK:       memref.store %false, [[ARG2]][] : memref<i1>
// CHECK:       memref.store [[ARG1]], [[ARG3]][] : memref<i8>
// CHECK:     } else {
// CHECK:       memref.store %c0_i2, [[ARG4]][] : memref<i2>
// CHECK:       memref.store %false, [[ARG2]][] : memref<i1>
// CHECK:       memref.store %true, [[ARG2]][] : memref<i1>
// CHECK:     }
// CHECK:   } else {
// CHECK:     [[VAL4:%.+]] = cmpi eq, [[VAL2]], %c1_i2 : i2
// CHECK:     scf.if [[VAL4]] {
// CHECK:       memref.store %c1_i2, [[ARG4:%.+]][] : memref<i2>
// CHECK:     }
// CHECK:   }
// CHECK:   return [[VAL0]] : i1
// CHECK: }
// CHECK: func @bar() {
// CHECK:   [[VAL0:%.+]] = memref.alloca() : memref<i1>
// CHECK:   [[VAL1:%.+]] = memref.alloca() : memref<i8>
// CHECK:   [[VAL2:%.+]] = memref.alloca() : memref<i2>
// CHECK:   %true = constant true
// CHECK:   %c16_i8 = constant 16 : i8
// CHECK:   [[VAL3:%.+]] = call @foo(%true, %c16_i8, [[VAL0]], [[VAL1]], [[VAL2]]) : (i1, i8, memref<i1>, memref<i8>, memref<i2>) -> i1
// CHECK:   %false = constant false
// CHECK:   %c0_i8 = constant 0 : i8
// CHECK:   [[VAL4:%.+]] = call @foo(%false, %c0_i8, [[VAL0]], [[VAL1]], [[VAL2]]) : (i1, i8, memref<i1>, memref<i8>, memref<i2>) -> i1
// CHECK:   return
// CHECK: }

fsm.machine @foo(%i_valid: i1, %i_len: i8) -> i1 attributes {stateType = i2} {
  %o_ready = fsm.variable "o_ready" {initValue = false} : i1
  %counter = fsm.variable "counter" {initValue = 0 : i8} : i8

  %true = constant true
  %false = constant false

  fsm.state "IDLE" entry  {
    fsm.update %o_ready, %true : i1
  } exit  {
    fsm.update %o_ready, %false : i1
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

  fsm.state "BUSY" entry  {
  } exit  {
  } transitions  {
    fsm.transition @BUSY guard  {
    } action  {
    }
  }
  fsm.output %o_ready : i1
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
