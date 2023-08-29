// RUN: circt-opt --dc-test-scf-to-dc %s | FileCheck %s

// CHECK-LABEL:   hw.module @scf.if.yield(
// CHECK-SAME:                            %[[VAL_0:.*]]: !dc.value<i1>,
// CHECK-SAME:                            %[[VAL_1:.*]]: !dc.value<i32>,
// CHECK-SAME:                            %[[VAL_2:.*]]: !dc.value<i32>) -> (out0: !dc.value<i32>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i32>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = dc.unpack %[[VAL_2]] : !dc.value<i32>
// CHECK:           %[[VAL_9:.*]] = dc.join %[[VAL_3]], %[[VAL_5]], %[[VAL_7]]
// CHECK:           %[[VAL_10:.*]] = dc.pack %[[VAL_9]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = dc.branch %[[VAL_10]]
// CHECK:           %[[VAL_13:.*]] = dc.merge %[[VAL_11]], %[[VAL_12]]
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = dc.unpack %[[VAL_13]] : !dc.value<i1>
// CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_4]], %[[VAL_6]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_17:.*]] = dc.pack %[[VAL_14]], %[[VAL_16]] : i32
// CHECK:           hw.output %[[VAL_17]] : !dc.value<i32>
// CHECK:         }
func.func @scf.if.yield(%cond : i1, %a : i32, %b : i32) -> i32 {
  %0 = scf.if %cond -> i32 {
    scf.yield %a : i32
  } else {
    scf.yield %b : i32
  }
  return %0 : i32
}

// CHECK-LABEL:   hw.module @scf.while(
// CHECK-SAME:                         %[[VAL_0:.*]]: !dc.value<index>,
// CHECK-SAME:                         %[[VAL_1:.*]]: !dc.value<index>,
// CHECK-SAME:                         %[[VAL_2:.*]]: !dc.value<index>) -> (out0: !dc.value<i32>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_0]] : !dc.value<index>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_1]] : !dc.value<index>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = dc.unpack %[[VAL_2]] : !dc.value<index>
// CHECK:           %[[VAL_9:.*]] = dc.join %[[VAL_3]], %[[VAL_5]], %[[VAL_7]]
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_11:.*]] = dc.buffer[1] %[[VAL_12:.*]] [0 : i32] : !dc.value<i1>
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = dc.unpack %[[VAL_11]] : !dc.value<i1>
// CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_16:.*]], %[[VAL_4]] : index
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_14]], %[[VAL_18:.*]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_19:.*]] = dc.select %[[VAL_11]], %[[VAL_20:.*]], %[[VAL_9]]
// CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_6]] : index
// CHECK:           %[[VAL_12]] = dc.pack %[[VAL_19]], %[[VAL_21]] : i1
// CHECK:           %[[VAL_20]], %[[VAL_22:.*]] = dc.branch %[[VAL_12]]
// CHECK:           %[[VAL_16]] = arith.addi %[[VAL_15]], %[[VAL_8]] : index
// CHECK:           %[[VAL_18]] = arith.addi %[[VAL_17]], %[[VAL_17]] : i32
// CHECK:           %[[VAL_23:.*]] = dc.pack %[[VAL_22]], %[[VAL_17]] : i32
// CHECK:           hw.output %[[VAL_23]] : !dc.value<i32>
// CHECK:         }
func.func @scf.while(%arg0 : index, %arg1 : index, %arg2 : index) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %0:2 = scf.while (%arg3 = %arg0, %arg4 = %c1_i32) : (index, i32) -> (index, i32) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg3, %arg4 : index, i32
  } do {
  ^bb0(%arg3: index, %arg4: i32):
    %1 = arith.addi %arg3, %arg2 : index
    %2 = arith.addi %arg4, %arg4 : i32
    scf.yield %1, %2 : index, i32
  }
  return %0#1 : i32
}

// CHECK-LABEL:   hw.module @scf.while.nested(
// CHECK-SAME:                                %[[VAL_0:.*]]: !dc.value<index>,
// CHECK-SAME:                                %[[VAL_1:.*]]: !dc.value<index>,
// CHECK-SAME:                                %[[VAL_2:.*]]: !dc.value<index>) -> (out0: !dc.value<i32>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_0]] : !dc.value<index>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_1]] : !dc.value<index>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = dc.unpack %[[VAL_2]] : !dc.value<index>
// CHECK:           %[[VAL_9:.*]] = dc.join %[[VAL_3]], %[[VAL_5]], %[[VAL_7]]
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_11:.*]] = dc.buffer[1] %[[VAL_12:.*]] [0 : i32] : !dc.value<i1>
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = dc.unpack %[[VAL_11]] : !dc.value<i1>
// CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_16:.*]], %[[VAL_4]] : index
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_14]], %[[VAL_18:.*]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_19:.*]] = dc.select %[[VAL_11]], %[[VAL_20:.*]], %[[VAL_9]]
// CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_6]] : index
// CHECK:           %[[VAL_12]] = dc.pack %[[VAL_19]], %[[VAL_21]] : i1
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = dc.branch %[[VAL_12]]
// CHECK:           %[[VAL_16]] = arith.addi %[[VAL_15]], %[[VAL_8]] : index
// CHECK:           %[[VAL_24:.*]] = dc.buffer[1] %[[VAL_25:.*]] [0 : i32] : !dc.value<i1>
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = dc.unpack %[[VAL_24]] : !dc.value<i1>
// CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_29:.*]], %[[VAL_4]] : index
// CHECK:           %[[VAL_18]] = arith.select %[[VAL_27]], %[[VAL_30:.*]], %[[VAL_17]] : i32
// CHECK:           %[[VAL_31:.*]] = dc.select %[[VAL_24]], %[[VAL_32:.*]], %[[VAL_22]]
// CHECK:           %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_28]], %[[VAL_6]] : index
// CHECK:           %[[VAL_25]] = dc.pack %[[VAL_31]], %[[VAL_33]] : i1
// CHECK:           %[[VAL_32]], %[[VAL_20]] = dc.branch %[[VAL_25]]
// CHECK:           %[[VAL_29]] = arith.addi %[[VAL_28]], %[[VAL_8]] : index
// CHECK:           %[[VAL_30]] = arith.addi %[[VAL_18]], %[[VAL_18]] : i32
// CHECK:           %[[VAL_34:.*]] = dc.pack %[[VAL_23]], %[[VAL_17]] : i32
// CHECK:           hw.output %[[VAL_34]] : !dc.value<i32>
// CHECK:         }
func.func @scf.while.nested(%arg0 : index, %arg1 : index, %arg2 : index) -> i32{
  %c1_i32 = arith.constant 1 : i32
  %0:2 = scf.while (%arg3 = %arg0, %arg4 = %c1_i32) : (index, i32) -> (index, i32) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg3, %arg4 : index, i32
  } do {
  ^bb0(%arg3: index, %arg4: i32):
    %1 = arith.addi %arg3, %arg2 : index
    %2:2 = scf.while (%arg5 = %arg0, %arg6 = %arg4) : (index, i32) -> (index, i32) {
      %3 = arith.cmpi slt, %arg5, %arg1 : index
      scf.condition(%3) %arg5, %arg6 : index, i32
    } do {
    ^bb0(%arg5: index, %arg6: i32):
      %3 = arith.addi %arg5, %arg2 : index
      %4 = arith.addi %arg6, %arg6 : i32
      scf.yield %3, %4 : index, i32
    }
    scf.yield %1, %2#1 : index, i32
  }
  return %0#1 : i32
}
