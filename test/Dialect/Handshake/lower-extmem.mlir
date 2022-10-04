// RUN: circt-opt -handshake-lower-extmem-to-hw %s | FileCheck %s

// CHECK-LABEL:   handshake.func @main(
// CHECK-SAME:           %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: none, %[[VAL_5:.*]]: none, ...) -> (none, index, !hw.struct<data: i32, addr: index>) attributes {argNames = ["arg0", "arg1", "v", "mem_ld0.data", "mem_st0.done", "argCtrl"], resNames = ["out0", "mem_ld0.addr", "mem_st0"]} {
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = join %[[VAL_6]]#1 : i32
// CHECK:           %[[VAL_8:.*]] = hw.struct_create (%[[VAL_9:.*]], %[[VAL_10:.*]]) : !hw.struct<data: i32, addr: index>
// CHECK:           %[[VAL_11:.*]]:2 = fork [2] %[[VAL_5]] : none
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = load {{\[}}%[[VAL_0]]] %[[VAL_6]]#0, %[[VAL_11]]#0 : index, i32
// CHECK:           %[[VAL_9]], %[[VAL_10]] = store {{\[}}%[[VAL_1]]] %[[VAL_2]], %[[VAL_11]]#1 : index, i32
// CHECK:           sink %[[VAL_12]] : i32
// CHECK:           %[[VAL_14:.*]] = join %[[VAL_4]], %[[VAL_7]] : none, none
// CHECK:           return %[[VAL_14]], %[[VAL_13]], %[[VAL_8]] : none, index, !hw.struct<data: i32, addr: index>
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
