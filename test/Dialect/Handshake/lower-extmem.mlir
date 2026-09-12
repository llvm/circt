// RUN: circt-opt -handshake-lower-extmem-to-hw %s | FileCheck %s

// CHECK-LABEL:   handshake.func @main(
// CHECK-SAME:          %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: none, %[[VAL_5:.*]]: none, ...) -> (none, i4, !hw.struct<address: i4, data: i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = join %[[VAL_6]]#1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_9:.*]] : index to i4
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_11:.*]] : index to i4
// CHECK:           %[[VAL_12:.*]] = hw.struct_create (%[[VAL_10]], %[[VAL_13:.*]]) : !hw.struct<address: i4, data: i32>
// CHECK:           %[[VAL_14:.*]]:2 = fork [2] %[[VAL_5]] : none
// CHECK:           %[[VAL_15:.*]], %[[VAL_9]] = load {{\[}}%[[VAL_0]]] %[[VAL_6]]#0, %[[VAL_14]]#0 : index, i32
// CHECK:           %[[VAL_13]], %[[VAL_11]] = store {{\[}}%[[VAL_1]]] %[[VAL_2]], %[[VAL_14]]#1 : index, i32
// CHECK:           sink %[[VAL_15]] : i32
// CHECK:           %[[VAL_16:.*]] = join %[[VAL_4]], %[[VAL_7]] : none, none
// CHECK:           return %[[VAL_16]], %[[VAL_8]], %[[VAL_12]] : none, i4, !hw.struct<address: i4, data: i32>
// CHECK:         }

handshake.func @main(%arg0: index, %arg1: index, %v: i32, %mem : memref<10xi32>, %argCtrl: none) -> none {
  %ldData, %stCtrl, %ldCtrl = handshake.extmemory[ld=1, st=1](%mem : memref<10xi32>)(%storeData, %storeAddr, %loadAddr) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %fCtrl:2 = fork [2] %argCtrl : none
  %loadData, %loadAddr = load [%arg0] %ldData, %fCtrl#0 : index, i32
  %storeData, %storeAddr = store [%arg1] %v, %fCtrl#1 : index, i32
  sink %loadData : i32
  %finCtrl = join %stCtrl, %ldCtrl : none, none
  return %finCtrl : none
}

// CHECK-LABEL: handshake.func @i0
// CHECK: %[[JOIN:.*]] = join
// CHECK: constant %[[JOIN]] {value = 0 : i0} : i0
handshake.func @i0(%c : memref<1xi32>) {
  %0 = source
  %addr = constant %0 {value = 0 : index} : index
  %data = constant %0 {value = 0 : i32} : i32
  %2 = extmemory[ld = 0, st = 1] (%c : memref<1xi32>) (%data, %addr) {id = 2 : i32} : (i32, index) -> none
  return
}

// Two memories where the first one is both read and written. Every memory that
// has already been lowered shifts the argument list by (numPorts - 1), so
// indexing it by the original argument index erased the wrong argument here -
// which tripped "Cannot destroy a value that still has uses!".
// CHECK-LABEL:   handshake.func @multiple_memories(
// CHECK-SAME:        %[[A0:.*]]: index, %[[A1:.*]]: index, %[[V:.*]]: i32, %[[M0LD:.*]]: i32, %[[M0ST:.*]]: none, %[[M1LD:.*]]: i32, %[[CTRL:.*]]: none, ...) -> (none, i4, !hw.struct<address: i4, data: i32>, i4)
// CHECK-SAME:    argNames = ["a0", "a1", "v", "m0_ld0.data", "m0_st0.done", "m1_ld0.data", "ctrl"]
// CHECK-SAME:    resNames = ["out0", "m0_ld0.addr", "m0_st0", "m1_ld0.addr"]
handshake.func @multiple_memories(%a0: index, %a1: index, %v: i32,
                                  %m0: memref<10xi32>, %m1: memref<10xi32>,
                                  %ctrl: none) -> none {
  %ld0, %st0c, %ld0c = handshake.extmemory[ld=1, st=1](%m0 : memref<10xi32>)(%sd, %sa, %la0) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %ld1, %ld1c = handshake.extmemory[ld=1, st=0](%m1 : memref<10xi32>)(%la1) {id = 1 : i32} : (index) -> (i32, none)
  %f:3 = fork [3] %ctrl : none
  %d0, %la0 = load [%a0] %ld0, %f#0 : index, i32
  %sd, %sa = store [%a1] %v, %f#1 : index, i32
  %d1, %la1 = load [%a0] %ld1, %f#2 : index, i32
  sink %d0 : i32
  sink %d1 : i32
  %fin = join %st0c, %ld0c, %ld1c : none, none, none
  return %fin : none
}
