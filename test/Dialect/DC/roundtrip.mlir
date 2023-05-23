// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL:   hw.module @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                   %[[VAL_1:.*]]: !dc.value<i1>,
// CHECK-SAME:                   %[[VAL_2:.*]]: i32) attributes {argNames = ["", "", ""]} {
// CHECK:           %[[VAL_3:.*]] = dc.buffer[2] %[[VAL_0]] : !dc.token
// CHECK:           %[[VAL_4:.*]] = dc.buffer[2] %[[VAL_1]] [1, 2] : !dc.value<i1>
// CHECK:           %[[VAL_5:.*]]:2 = dc.fork [2] %[[VAL_0]]
// CHECK:           %[[VAL_6:.*]] = dc.pack %[[VAL_0]]{{\[}}%[[VAL_2]]] : i32
// CHECK:           %[[VAL_7:.*]] = dc.merge %[[VAL_3]], %[[VAL_0]]
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i1>
// CHECK:           hw.output
// CHECK:         }
hw.module @foo(%0 : !dc.token, %1 : !dc.value<i1>, %2 : i32) {
  %buffer = dc.buffer [2] %0 : !dc.token
  %bufferInit = dc.buffer [2] %1 [1, 2] : !dc.value<i1>
  %f1, %f2 = dc.fork [2] %0
  %pack = dc.pack %0 [%2] : i32
  %merge = dc.merge %buffer, %0
  %unpack_token, %unpack_value = dc.unpack %1 : !dc.value<i1>
}
