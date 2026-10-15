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

// CHECK-LABEL:   handshake.func @two_extmems(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, %[[VAL_3:.*]]: none, ...) -> (none, i4, !hw.struct<address: i4, data: i32>, !hw.struct<address: i4, data: i32>)
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]] {value = 0 : index} : index
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_3]] {value = 0 : index} : index
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_3]] {value = 0 : index} : index
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_3]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_3]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_0]] : i32
// CHECK:           %[[VAL_10:.*]] = join %[[VAL_9]]#1 : i32
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_4]] : index to i4
// CHECK:           %[[VAL_12:.*]] = arith.index_cast %[[VAL_5]] : index to i4
// CHECK:           %[[VAL_13:.*]] = hw.struct_create (%[[VAL_12]], %[[VAL_7]]) : !hw.struct<address: i4, data: i32>
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_6]] : index to i4
// CHECK:           %[[VAL_15:.*]] = hw.struct_create (%[[VAL_14]], %[[VAL_8]]) : !hw.struct<address: i4, data: i32>
// CHECK:           sink %[[VAL_9]]#0 : i32
// CHECK:           %[[VAL_16:.*]] = join %[[VAL_1]], %[[VAL_10]], %[[VAL_2]] : none, none, none
// CHECK:           return %[[VAL_16]], %[[VAL_11]], %[[VAL_13]], %[[VAL_15]] : none, i4, !hw.struct<address: i4, data: i32>, !hw.struct<address: i4, data: i32>
// CHECK:         }
handshake.func @two_extmems(%mem0: memref<10xi32>, %mem1: memref<10xi32>, %argCtrl: none) -> none {
  %ldAddr = constant %argCtrl {value = 0 : index} : index
  %stAddr0 = constant %argCtrl {value = 0 : index} : index
  %stAddr1 = constant %argCtrl {value = 0 : index} : index
  %stData0 = constant %argCtrl {value = 0 : i32} : i32
  %stData1 = constant %argCtrl {value = 0 : i32} : i32
  %ldData, %stCtrl0, %ldCtrl = handshake.extmemory[ld=1, st=1](%mem0 : memref<10xi32>)(%stData0, %stAddr0, %ldAddr) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %stCtrl1 = handshake.extmemory[ld=0, st=1](%mem1 : memref<10xi32>)(%stData1, %stAddr1) {id = 1 : i32} : (i32, index) -> none
  sink %ldData : i32
  %finCtrl = join %stCtrl0, %ldCtrl, %stCtrl1 : none, none, none
  return %finCtrl : none
}
