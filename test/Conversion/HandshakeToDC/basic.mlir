// RUN: circt-opt %s --lower-handshake-to-dc | FileCheck %s

// CHECK:   hw.module @test_fork(in %[[VAL_0:.*]] : !dc.token, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.token, out out1 : !dc.token) {
// CHECK:           %[[VAL_1:.*]]:2 = dc.fork [2] %[[VAL_0]]
// CHECK:           hw.output %[[VAL_1]]#0, %[[VAL_1]]#1 : !dc.token, !dc.token
// CHECK:         }
handshake.func @test_fork(%arg0: none) -> (none, none) {
  %0:2 = fork [2] %arg0 : none
  return %0#0, %0#1 : none, none
}

// CHECK:   hw.module @test_fork_data(in %[[VAL_0:.*]] : !dc.value<i32>, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i32>) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i32>
// CHECK:           %[[VAL_3:.*]]:2 = dc.fork [2] %[[VAL_1]]
// CHECK:           %[[VAL_4:.*]] = dc.pack %[[VAL_3]]#0, %[[VAL_2]] : i32
// CHECK:           %[[VAL_5:.*]] = dc.pack %[[VAL_3]]#1, %[[VAL_2]] : i32
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = dc.unpack %[[VAL_4]] : !dc.value<i32>
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = dc.unpack %[[VAL_5]] : !dc.value<i32>
// CHECK:           %[[VAL_10:.*]] = dc.join %[[VAL_6]], %[[VAL_8]]
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_12:.*]] = dc.pack %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:           hw.output %[[VAL_12]] : !dc.value<i32>
// CHECK:         }
handshake.func @test_fork_data(%arg0: i32) -> (i32) {
  %0:2 = fork [2] %arg0 : i32
  %1 = arith.addi %0#0, %0#1 : i32
  return %1 : i32
}

// CHECK:   hw.module @top(in %[[VAL_0:.*]] : !dc.value<i64>, in %[[VAL_1:.*]] : !dc.value<i64>, in %[[VAL_2:.*]] : !dc.token, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i64>, out out1 : !dc.token) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i64>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i64>
// CHECK:           %[[VAL_7:.*]] = dc.join %[[VAL_3]], %[[VAL_5]]
// CHECK:           %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_4]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_9:.*]] = dc.pack %[[VAL_7]], %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = dc.unpack %[[VAL_9]] : !dc.value<i1>
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i64>
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i64>
// CHECK:           %[[VAL_16:.*]] = dc.join %[[VAL_10]], %[[VAL_12]], %[[VAL_14]]
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_11]], %[[VAL_13]], %[[VAL_15]] : i64
// CHECK:           %[[VAL_18:.*]] = dc.pack %[[VAL_16]], %[[VAL_17]] : i64
// CHECK:           hw.output %[[VAL_18]], %[[VAL_2]] : !dc.value<i64>, !dc.token
// CHECK:         }
handshake.func @top(%arg0: i64, %arg1: i64, %arg8: none, ...) -> (i64, none) {
    %0 = arith.cmpi slt, %arg0, %arg1 : i64
    %1 = arith.select %0, %arg1, %arg0 : i64
    return %1, %arg8 : i64, none
}

// CHECK-LABEL:   hw.module @constant(in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i32>) {
// CHECK:           %[[VAL_1:.*]] = dc.source
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_3:.*]] = dc.pack %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           hw.output %[[VAL_3]] : !dc.value<i32>
// CHECK:         }
handshake.func @constant() -> (i32) {
  %0 = arith.constant 42 : i32
  return %0 : i32
}

// CHECK:   hw.module @mux(in %[[VAL_0:.*]] : !dc.value<i1>, in %[[VAL_1:.*]] : !dc.value<i64>, in %[[VAL_2:.*]] : !dc.value<i64>, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i64>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i64>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = dc.unpack %[[VAL_2]] : !dc.value<i64>
// CHECK:           %[[VAL_9:.*]] = arith.constant false
// CHECK:           %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_9]] : i1
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_12:.*]] = dc.pack %[[VAL_3]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_13:.*]] = dc.select %[[VAL_12]], %[[VAL_7]], %[[VAL_5]]
// CHECK:           %[[VAL_14:.*]] = dc.pack %[[VAL_13]], %[[VAL_11]] : i64
// CHECK:           hw.output %[[VAL_14]] : !dc.value<i64>
// CHECK:         }
handshake.func @mux(%select : i1, %a : i64, %b : i64) -> i64{
  %0 = handshake.mux %select [%a, %b] : i1, i64
  return %0 : i64
}

// CHECK:   hw.module @mux4(in %[[VAL_0:.*]] : !dc.value<i2>, in %[[VAL_1:.*]] : !dc.value<i64>, in %[[VAL_2:.*]] : !dc.value<i64>, in %[[VAL_3:.*]] : !dc.value<i64>, in %[[VAL_4:.*]] : !dc.value<i64>, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i64>) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i2>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i64>
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = dc.unpack %[[VAL_2]] : !dc.value<i64>
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = dc.unpack %[[VAL_3]] : !dc.value<i64>
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = dc.unpack %[[VAL_4]] : !dc.value<i64>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : i2
// CHECK:           %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_15]] : i2
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_10]], %[[VAL_8]] : i64
// CHECK:           %[[VAL_18:.*]] = dc.pack %[[VAL_5]], %[[VAL_16]] : i1
// CHECK:           %[[VAL_19:.*]] = dc.select %[[VAL_18]], %[[VAL_9]], %[[VAL_7]]
// CHECK:           %[[VAL_20:.*]] = arith.constant 1 : i2
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_20]] : i2
// CHECK:           %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_12]], %[[VAL_17]] : i64
// CHECK:           %[[VAL_23:.*]] = dc.pack %[[VAL_5]], %[[VAL_21]] : i1
// CHECK:           %[[VAL_24:.*]] = dc.select %[[VAL_23]], %[[VAL_11]], %[[VAL_19]]
// CHECK:           %[[VAL_25:.*]] = arith.constant -2 : i2
// CHECK:           %[[VAL_26:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_25]] : i2
// CHECK:           %[[VAL_27:.*]] = arith.select %[[VAL_26]], %[[VAL_14]], %[[VAL_22]] : i64
// CHECK:           %[[VAL_28:.*]] = dc.pack %[[VAL_5]], %[[VAL_26]] : i1
// CHECK:           %[[VAL_29:.*]] = dc.select %[[VAL_28]], %[[VAL_13]], %[[VAL_24]]
// CHECK:           %[[VAL_30:.*]] = dc.pack %[[VAL_29]], %[[VAL_27]] : i64
// CHECK:           hw.output %[[VAL_30]] : !dc.value<i64>
// CHECK:         }
handshake.func @mux4(%select : i2, %a : i64, %b : i64, %c : i64, %d : i64) -> i64{
  %0 = handshake.mux %select [%a, %b, %c, %d] : i2, i64
  return %0 : i64
}

// CHECK:   hw.module @test_conditional_branch(in %[[VAL_0:.*]] : !dc.value<i1>, in %[[VAL_1:.*]] : !dc.value<index>, in %[[VAL_2:.*]] : !dc.token, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<index>, out out1 : !dc.value<index>, out out2 : !dc.token) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_1]] : !dc.value<index>
// CHECK:           %[[VAL_7:.*]] = dc.join %[[VAL_3]], %[[VAL_5]]
// CHECK:           %[[VAL_8:.*]] = dc.pack %[[VAL_7]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = dc.branch %[[VAL_8]]
// CHECK:           %[[VAL_11:.*]] = dc.pack %[[VAL_9]], %[[VAL_6]] : index
// CHECK:           %[[VAL_12:.*]] = dc.pack %[[VAL_10]], %[[VAL_6]] : index
// CHECK:           hw.output %[[VAL_11]], %[[VAL_12]], %[[VAL_2]] : !dc.value<index>, !dc.value<index>, !dc.token
// CHECK:         }
handshake.func @test_conditional_branch(%arg0: i1, %arg1: index, %arg2: none, ...) -> (index, index, none) {
  %0:2 = cond_br %arg0, %arg1 : index
  return %0#0, %0#1, %arg2 : index, index, none
}

// CHECK:   hw.module @test_conditional_branch_none(in %[[VAL_0:.*]] : !dc.value<i1>, in %[[VAL_1:.*]] : !dc.token, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.token, out out1 : !dc.token) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           %[[VAL_4:.*]] = dc.join %[[VAL_2]], %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]] = dc.pack %[[VAL_4]], %[[VAL_3]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = dc.branch %[[VAL_5]]
// CHECK:           hw.output %[[VAL_6]], %[[VAL_7]] : !dc.token, !dc.token
// CHECK:         }
handshake.func @test_conditional_branch_none(%arg0: i1, %arg1: none) -> (none, none) {
  %0:2 = cond_br %arg0, %arg1 : none
  return %0#0, %0#1 : none, none
}

// CHECK:   hw.module @test_constant(in %[[VAL_0:.*]] : !dc.token, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i32>) {
// CHECK:           %[[VAL_1:.*]] = dc.source
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_3:.*]] = dc.pack %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           hw.output %[[VAL_3]] : !dc.value<i32>
// CHECK:         }
handshake.func @test_constant(%arg0: none) -> (i32) {
  %1 = constant %arg0 {value = 42 : i32} : i32
  return %1: i32
}

// CHECK:   hw.module @test_control_merge(in %[[VAL_0:.*]] : !dc.token, in %[[VAL_1:.*]] : !dc.token, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.token, out out1 : !dc.value<index>) {
// CHECK:           %[[VAL_2:.*]] = dc.merge %[[VAL_0]], %[[VAL_1]]
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_2]] : !dc.value<i1>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_2]] : !dc.value<i1>
// CHECK:           %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : i1 to index
// CHECK:           %[[VAL_8:.*]] = dc.pack %[[VAL_5]], %[[VAL_7]] : index
// CHECK:           hw.output %[[VAL_3]], %[[VAL_8]] : !dc.token, !dc.value<index>
// CHECK:         }
handshake.func @test_control_merge(%arg0 : none, %arg1 : none) -> (none, index) {
  %out, %idx = control_merge %arg0, %arg1 : none, index
  return %out, %idx : none, index
}

// CHECK:   hw.module @test_control_merge_data(in %[[VAL_0:.*]] : !dc.value<i2>, in %[[VAL_1:.*]] : !dc.value<i2>, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i2>, out out1 : !dc.value<index>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i2>
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i2>
// CHECK:           %[[VAL_6:.*]] = dc.merge %[[VAL_2]], %[[VAL_4]]
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = dc.unpack %[[VAL_6]] : !dc.value<i1>
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_3]], %[[VAL_5]] : i2
// CHECK:           %[[VAL_10:.*]] = dc.pack %[[VAL_7]], %[[VAL_9]] : i2
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_8]] : i1 to index
// CHECK:           %[[VAL_12:.*]] = dc.pack %[[VAL_7]], %[[VAL_11]] : index
// CHECK:           hw.output %[[VAL_10]], %[[VAL_12]] : !dc.value<i2>, !dc.value<index>
// CHECK:         }
handshake.func @test_control_merge_data(%arg0 : i2, %arg1 : i2) -> (i2, index) {
  %out, %idx = control_merge %arg0, %arg1 : i2, index
  return %out, %idx : i2, index
}

// CHECK-LABEL:   hw.module @test_control_fixed_index_type(in 
// CHECK-SAME:                    %[[VAL_0:.*]] : !dc.value<i4>, in %[[VAL_1:.*]] : !dc.value<i4>, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i4>, out out1 : !dc.value<i32>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i4>
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i4>
// CHECK:           %[[VAL_6:.*]] = dc.merge %[[VAL_2]], %[[VAL_4]]
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = dc.unpack %[[VAL_6]] : !dc.value<i1>
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_3]], %[[VAL_5]] : i4
// CHECK:           %[[VAL_10:.*]] = dc.pack %[[VAL_7]], %[[VAL_9]] : i4
// CHECK:           %[[VAL_11:.*]] = arith.extui %[[VAL_8]] : i1 to i32
// CHECK:           %[[VAL_12:.*]] = dc.pack %[[VAL_7]], %[[VAL_11]] : i32
// CHECK:           hw.output %[[VAL_10]], %[[VAL_12]] : !dc.value<i4>, !dc.value<i32>
// CHECK:         }
handshake.func @test_control_fixed_index_type(%arg0 : i4, %arg1 : i4) -> (i4, i32) {
  %out, %idx = control_merge %arg0, %arg1 : i4, i32
  return %out, %idx : i4, i32
}

// CHECK:   hw.module @branch_and_merge(in %[[VAL_0:.*]] : !dc.value<i1>, in %[[VAL_1:.*]] : !dc.token, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.token, out out1 : !dc.value<index>) {
// CHECK:           %[[VAL_2:.*]] = dc.merge %[[VAL_3:.*]], %[[VAL_4:.*]]
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_2]] : !dc.value<i1>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = dc.unpack %[[VAL_2]] : !dc.value<i1>
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : i1 to index
// CHECK:           %[[VAL_10:.*]] = dc.pack %[[VAL_7]], %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           %[[VAL_13:.*]] = dc.join %[[VAL_11]], %[[VAL_1]]
// CHECK:           %[[VAL_14:.*]] = dc.pack %[[VAL_13]], %[[VAL_12]] : i1
// CHECK:           %[[VAL_3]], %[[VAL_4]] = dc.branch %[[VAL_14]]
// CHECK:           hw.output %[[VAL_5]], %[[VAL_10]] : !dc.token, !dc.value<index>
// CHECK:         }
handshake.func @branch_and_merge(%arg0 : i1, %arg1 : none) -> (none, index) {
  %out, %idx = control_merge %true, %false : none, index
  %true, %false = cond_br %arg0, %arg1 : none
  return %out, %idx : none, index
}

// CHECK:   hw.module @datamerge(in %[[VAL_0:.*]] : !dc.value<i1>, in %[[VAL_1:.*]] : !dc.value<i1>, in %[[VAL_2:.*]] : !dc.value<i1>, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i1>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i1>
// CHECK:           %[[VAL_7:.*]] = dc.merge %[[VAL_3]], %[[VAL_5]]
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = dc.unpack %[[VAL_7]] : !dc.value<i1>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_4]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_11:.*]] = dc.pack %[[VAL_8]], %[[VAL_10]] : i1
// CHECK:           hw.output %[[VAL_11]] : !dc.value<i1>
// CHECK:         }
handshake.func @datamerge(%arg0 : i1, %arg1 : i1, %arg2 : i1) -> i1 {
  %out = merge %arg0, %arg1 : i1
  return %out : i1
}

// CHECK:   hw.module @nonemerge(in %[[VAL_0:.*]] : !dc.token, in %[[VAL_1:.*]] : !dc.token, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.token) {
// CHECK:           %[[VAL_2:.*]] = dc.merge %[[VAL_0]], %[[VAL_1]]
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_2]] : !dc.value<i1>
// CHECK:           hw.output %[[VAL_3]] : !dc.token
// CHECK:         }
handshake.func @nonemerge(%arg0 : none, %arg1 : none) -> none {
  %out = merge %arg0, %arg1 : none
  return %out : none
}

// CHECK-LABEL:   hw.module @pack_unpack(in 
// CHECK-SAME:               %[[VAL_0:.*]] : !dc.value<i32>, in %[[VAL_1:.*]] : !dc.value<i1>, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i32>, out out1 : !dc.value<i1>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i32>
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i1>
// CHECK:           %[[VAL_6:.*]] = dc.join %[[VAL_2]], %[[VAL_4]]
// CHECK:           %[[VAL_7:.*]] = hw.struct_create (%[[VAL_3]], %[[VAL_5]]) : !hw.struct<field0: i32, field1: i1>
// CHECK:           %[[VAL_8:.*]] = dc.pack %[[VAL_6]], %[[VAL_7]] : !hw.struct<field0: i32, field1: i1>
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = dc.unpack %[[VAL_8]] : !dc.value<!hw.struct<field0: i32, field1: i1>>
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = hw.struct_explode %[[VAL_10]] : !hw.struct<field0: i32, field1: i1>
// CHECK:           %[[VAL_13:.*]] = dc.pack %[[VAL_9]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_14:.*]] = dc.pack %[[VAL_9]], %[[VAL_12]] : i1
// CHECK:           hw.output %[[VAL_13]], %[[VAL_14]] : !dc.value<i32>, !dc.value<i1>
// CHECK:         }
handshake.func @pack_unpack(%arg0 : i32, %arg1 : i1) -> (i32, i1) {
  %packed = handshake.pack %arg0, %arg1 : tuple<i32, i1>
  %a, %b = handshake.unpack %packed : tuple<i32, i1>
  return %a, %b : i32, i1
}

// CHECK-LABEL:  hw.module @esi_outer(in %clk : !seq.clock, in %rst : i1, in %arg0 : !esi.channel<i32>, in %arg1 : !esi.channel<i1>, out out : !esi.channel<i32>) {
// CHECK-NEXT:     [[R0:%.+]] = dc.from_esi %arg0 : <i32>
// CHECK-NEXT:     [[R1:%.+]] = dc.from_esi %arg1 : <i1>
// CHECK-NEXT:     %pack_unpack_inst.out0, %pack_unpack_inst.out1 = hw.instance "pack_unpack_inst" @pack_unpack(arg0: [[R0]]: !dc.value<i32>, arg1: [[R1]]: !dc.value<i1>, clk: %clk: !seq.clock, rst: %rst: i1) -> (out0: !dc.value<i32>, out1: !dc.value<i1>)
// CHECK-NEXT:     [[R2:%.+]] = dc.to_esi %pack_unpack_inst.out0 : !dc.value<i32>
// CHECK-NEXT:     [[R3:%.+]] = dc.to_esi %pack_unpack_inst.out1 : !dc.value<i1>
// CHECK-NEXT:     hw.output [[R2]] : !esi.channel<i32>
// CHECK-NEXT:   }
hw.module @esi_outer(in %clk: !seq.clock, in %rst: i1, in %arg0: !esi.channel<i32>, in %arg1: !esi.channel<i1>, out out: !esi.channel<i32>) {
  %ret0, %ret1 = handshake.esi_instance @pack_unpack "pack_unpack_inst" clk %clk rst %rst (%arg0, %arg1) : (!esi.channel<i32>, !esi.channel<i1>) -> (!esi.channel<i32>, !esi.channel<i1>)
  hw.output %ret0 : !esi.channel<i32>
}

// CHECK: sv.verbatim "// Blah"
sv.verbatim "// Blah"

// CHECK-LABEL:  hw.module @TestFunc(in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i8>) {
// CHECK-NEXT:     [[R0:%.+]] = dc.source
// CHECK-NEXT:     %c15_i8 = hw.constant 15 : i8
// CHECK-NEXT:     [[R1:%.+]] = dc.pack [[R0]], %c15_i8 : i8
// CHECK-NEXT:     hw.output [[R1]] : !dc.value<i8>
handshake.func @TestFunc() -> i8 {
  %c15_i8 = hw.constant 15 : i8
  return %c15_i8 : i8
}

// CHECK-LABEL:  hw.module @TestSync(in %in0 : !dc.value<i64>, in %clk : !seq.clock {dc.clock}, in %rst : i1 {dc.reset}, out out0 : !dc.value<i64>) {
// CHECK-NEXT:     %token, %output = dc.unpack %in0 : !dc.value<i64>
// CHECK-NEXT:     [[R0:%.+]] = dc.join %token
// CHECK-NEXT:     [[R1:%.+]] = dc.pack [[R0]], %output : i64
// CHECK-NEXT:     hw.output [[R1]] : !dc.value<i64>
handshake.func @TestSync(%arg0: i64) -> (i64) attributes {argNames = ["in0"], resNames = ["out0"]} {
  %1 = sync %arg0 : i64
  return %1 : i64
}
