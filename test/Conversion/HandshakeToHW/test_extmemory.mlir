// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_load_in_ui64_ui32_out_ui32_ui64(
// CHECK-SAME:          %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: !esi.channel<i32>, %[[VAL_2:.*]]: !esi.channel<none>) -> (dataOut: !esi.channel<i32>, addrOut0: !esi.channel<i64>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i64
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_8:.*]] : i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_5]] : none
// CHECK:           %[[VAL_11:.*]], %[[VAL_8]] = esi.wrap.vr %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = esi.wrap.vr %[[VAL_3]], %[[VAL_14:.*]] : i64
// CHECK:           %[[VAL_14]] = comb.and %[[VAL_4]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_13]], %[[VAL_14]] : i1
// CHECK:           hw.output %[[VAL_11]], %[[VAL_12]] : !esi.channel<i32>, !esi.channel<i64>
// CHECK:         }

// CHECK-LABEL:   hw.module @handshake_store_in_ui64_ui32_out_ui32_ui64(
// CHECK-SAME:         %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: !esi.channel<i32>, %[[VAL_2:.*]]: !esi.channel<none>) -> (dataToMem: !esi.channel<i32>, addrOut0: !esi.channel<i64>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i64
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_5]] : none
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.wrap.vr %[[VAL_6]], %[[VAL_12:.*]] : i32
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = esi.wrap.vr %[[VAL_3]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_15:.*]] = comb.and %[[VAL_11]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_12]] = comb.and %[[VAL_7]], %[[VAL_4]], %[[VAL_9]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_15]], %[[VAL_12]] : i1
// CHECK:           hw.output %[[VAL_10]], %[[VAL_13]] : !esi.channel<i32>, !esi.channel<i64>
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
