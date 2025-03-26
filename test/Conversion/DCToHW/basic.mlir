// RUN: circt-opt --lower-dc-to-hw %s | FileCheck %s

// CHECK-LABEL:   hw.module @simple(
// CHECK: in %[[VAL_0:.*]] "" : !esi.channel<i0>, in %[[VAL_1:.*]] "" : !esi.channel<i64>, in %[[VAL_2:.*]] "" : i1, in %[[VAL_3:.*]] "" : !esi.channel<i1>, out out0 : !esi.channel<i0>, out out1 : !esi.channel<i64>,  out out2 : i1, out out3 : !esi.channel<i1>) {
// CHECK:           hw.output %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : !esi.channel<i0>, !esi.channel<i64>, i1, !esi.channel<i1>
// CHECK:         }
hw.module @simple(in %0 "" : !dc.token, in %1 "" : !dc.value<i64>, in %2 "" : i1, in %3 : !dc.value<i1>,
        out out0: !dc.token, out out1: !dc.value<i64>, out out2: i1, out out3: !dc.value<i1>) {
    hw.output %0, %1, %2, %3 : !dc.token, !dc.value<i64>, i1, !dc.value<i1>
}

// CHECK-LABEL:   hw.module @pack(
// CHECK:                    in %[[VAL_0:.*]] : !esi.channel<i0>, in %[[VAL_1:.*]] : i64, out out0 : !esi.channel<i64>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_4]] = esi.wrap.vr %[[VAL_1]], %[[VAL_3]] : i64
// CHECK:           hw.output %[[VAL_5]] : !esi.channel<i64>
hw.module @pack(in %token : !dc.token, in %v1 : i64, out out0: !dc.value<i64>) {
    %out = dc.pack %token, %v1 : i64
    hw.output %out : !dc.value<i64>
}

// CHECK-LABEL:   hw.module @unpack(
// CHECK:                      in %[[VAL_0:.*]] : !esi.channel<i64>, out out0 : !esi.channel<i0>, out out1 : i64) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i64
// CHECK:           %[[VAL_4:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_3]] = esi.wrap.vr %[[VAL_4]], %[[VAL_2]] : i0
// CHECK:           hw.output %[[VAL_5]], %[[VAL_1]] : !esi.channel<i0>, i64
hw.module @unpack(in %v : !dc.value<i64>, out out0: !dc.token, out out1: i64) {
    %out:2 = dc.unpack %v : !dc.value<i64>
    hw.output %out#0, %out#1 : !dc.token, i64
}

// CHECK-LABEL:   hw.module @join(
// CHECK:            in %[[VAL_0:.*]] : !esi.channel<i0>, in %[[VAL_1:.*]] : !esi.channel<i0>, out out0 : !esi.channel<i0>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : i0
// CHECK:           %[[VAL_7:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = esi.wrap.vr %[[VAL_7]], %[[VAL_10:.*]] : i0
// CHECK:           %[[VAL_10]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_9]], %[[VAL_10]] : i1
// CHECK:           hw.output %[[VAL_8]] : !esi.channel<i0>
// CHECK:         }
hw.module @join(in %t1 : !dc.token, in %t2 : !dc.token, out out0: !dc.token) {
    %out = dc.join %t1, %t2
    hw.output %out : !dc.token
}

// CHECK-LABEL:   hw.module @fork(
// CHECK:           in %[[VAL_0:.*]] : !esi.channel<i0>, in %[[VAL_1:.*]] : !seq.clock {dc.clock}, in %[[VAL_2:.*]] : i1 {dc.reset}, out out0 : !esi.channel<i0>, out out1 : !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i0
// CHECK:           %[[VAL_6:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_6]], %[[VAL_9:.*]] : i0
// CHECK:           %[[VAL_10:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = esi.wrap.vr %[[VAL_10]], %[[VAL_13:.*]] : i0
// CHECK:           %[[VAL_14:.*]] = hw.constant false
// CHECK:           %[[VAL_15:.*]] = hw.constant true
// CHECK:           %[[VAL_16:.*]] = comb.xor %[[VAL_5]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_17:.*]] = comb.and %[[VAL_18:.*]], %[[VAL_16]] : i1
// CHECK:           %[[VAL_19:.*]] = seq.compreg sym @emitted_0 %[[VAL_17]], %[[VAL_1]] reset %[[VAL_2]], %[[VAL_14]]  : i1
// CHECK:           %[[VAL_20:.*]] = comb.xor %[[VAL_19]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_9]] = comb.and %[[VAL_20]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.and %[[VAL_8]], %[[VAL_9]] : i1
// CHECK:           %[[VAL_18]] = comb.or %[[VAL_21]], %[[VAL_19]] {sv.namehint = "done0"} : i1
// CHECK:           %[[VAL_22:.*]] = comb.xor %[[VAL_5]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_23:.*]] = comb.and %[[VAL_24:.*]], %[[VAL_22]] : i1
// CHECK:           %[[VAL_25:.*]] = seq.compreg sym @emitted_1 %[[VAL_23]], %[[VAL_1]] reset %[[VAL_2]], %[[VAL_14]]  : i1
// CHECK:           %[[VAL_26:.*]] = comb.xor %[[VAL_25]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_13]] = comb.and %[[VAL_26]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_27:.*]] = comb.and %[[VAL_12]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_24]] = comb.or %[[VAL_27]], %[[VAL_25]] {sv.namehint = "done1"} : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_18]], %[[VAL_24]] {sv.namehint = "allDone"} : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_11]] : !esi.channel<i0>, !esi.channel<i0>
// CHECK:         }
hw.module @fork(in %t : !dc.token, in %clk : !seq.clock {"dc.clock"}, in %rst : i1 {"dc.reset"}, out out0 : !dc.token, out out1 : !dc.token) {
    %out:2 = dc.fork [2] %t
    hw.output %out#0, %out#1 : !dc.token, !dc.token
}

// CHECK-LABEL:   hw.module @bufferToken(
// CHECK:              in %[[VAL_0:.*]] : !esi.channel<i0>, in %[[VAL_1:.*]] : !seq.clock {dc.clock}, in %[[VAL_2:.*]] : i1 {dc.reset}, out out0 : !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]] = esi.buffer %[[VAL_1]], %[[VAL_2]], %[[VAL_0]] {stages = 2 : i64} : !esi.channel<i0> -> !esi.channel<i0>
// CHECK:           hw.output %[[VAL_3]] : !esi.channel<i0>
// CHECK:         }
hw.module @bufferToken(in %t1 : !dc.token, in %clk : !seq.clock {"dc.clock"}, in %rst : i1 {"dc.reset"}, out out0: !dc.token) {
    %out = dc.buffer [2] %t1 : !dc.token
    hw.output %out : !dc.token
}

// CHECK-LABEL:   hw.module @bufferValue(
// CHECK-SAME:              in %[[VAL_0:.*]] : !esi.channel<i64>, 
// CHECK-SAME:              in %[[VAL_1:.*]] : !seq.clock {dc.clock}, 
// CHECK-SAME:              in %[[VAL_2:.*]] : i1 {dc.reset},
// CHECK-SAME:              out out0 : !esi.channel<i64>) {
// CHECK:           %[[VAL_3:.*]] = esi.buffer %[[VAL_1]], %[[VAL_2]], %[[VAL_0]] {stages = 2 : i64} : !esi.channel<i64> -> !esi.channel<i64>
// CHECK:           hw.output %[[VAL_3]] : !esi.channel<i64>
// CHECK:         }
hw.module @bufferValue(in %v1 : !dc.value<i64>, in %clk : !seq.clock {"dc.clock"}, in %rst : i1 {"dc.reset"}, out out0: !dc.value<i64>) {
    %out = dc.buffer [2] %v1 : !dc.value<i64>
    hw.output %out : !dc.value<i64>
}

// CHECK-LABEL:   hw.module @branch(
// CHECK:                      in %[[VAL_0:.*]] : !esi.channel<i1>, out out0 : !esi.channel<i0>, out out1 : !esi.channel<i0>) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i1
// CHECK:           %[[VAL_4:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.wrap.vr %[[VAL_4]], %[[VAL_7:.*]] : i0
// CHECK:           %[[VAL_8:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.wrap.vr %[[VAL_8]], %[[VAL_11:.*]] : i0
// CHECK:           %[[VAL_7]] = comb.and %[[VAL_1]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_12:.*]] = hw.constant true
// CHECK:           %[[VAL_13:.*]] = comb.xor %[[VAL_1]], %[[VAL_12]] : i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_13]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_14:.*]] = comb.mux %[[VAL_1]], %[[VAL_6]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_3]] = comb.and %[[VAL_14]], %[[VAL_2]] : i1
// CHECK:           hw.output %[[VAL_5]], %[[VAL_9]] : !esi.channel<i0>, !esi.channel<i0>
// CHECK:         }
hw.module @branch(in %sel : !dc.value<i1>, out out0: !dc.token, out out1: !dc.token) {
    %true, %false = dc.branch %sel
    hw.output %true, %false : !dc.token, !dc.token
}

// CHECK-LABEL:   hw.module @select(
// CHECK:               in %[[VAL_0:.*]] : !esi.channel<i1>, in %[[VAL_1:.*]] : !esi.channel<i0>, in %[[VAL_2:.*]] : !esi.channel<i0>, out out0 : !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_8:.*]] : i0
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_11:.*]] : i0
// CHECK:           %[[VAL_12:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = esi.wrap.vr %[[VAL_12]], %[[VAL_15:.*]] : i0
// CHECK:           %[[VAL_16:.*]] = hw.constant false
// CHECK:           %[[VAL_17:.*]] = comb.concat %[[VAL_16]], %[[VAL_3]] : i1, i1
// CHECK:           %[[VAL_18:.*]] = hw.constant 1 : i2
// CHECK:           %[[VAL_19:.*]] = comb.shl %[[VAL_18]], %[[VAL_17]] : i2
// CHECK:           %[[VAL_20:.*]] = comb.mux %[[VAL_3]], %[[VAL_10]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_15]] = comb.and %[[VAL_20]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_15]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.extract %[[VAL_19]] from 0 : (i2) -> i1
// CHECK:           %[[VAL_8]] = comb.and %[[VAL_21]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_22:.*]] = comb.extract %[[VAL_19]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_22]], %[[VAL_5]] : i1
// CHECK:           hw.output %[[VAL_13]] : !esi.channel<i0>
// CHECK:         }
hw.module @select(in %sel : !dc.value<i1>, in %true : !dc.token, in %false : !dc.token, out out0: !dc.token) {
    %0 = dc.select %sel, %true, %false
    hw.output %0 : !dc.token
}

// CHECK-LABEL:   hw.module @to_from_esi_noop(
// CHECK:               in %[[VAL_0:.*]] : !esi.channel<i0>, in %[[VAL_1:.*]] : !esi.channel<i1>, out token : !esi.channel<i0>, out value : !esi.channel<i1>) {
// CHECK-NEXT:           hw.output %[[VAL_0]], %[[VAL_1]] : !esi.channel<i0>, !esi.channel<i1>
// CHECK-NEXT:         }
hw.module @to_from_esi_noop(in %token : !esi.channel<i0>, in %value : !esi.channel<i1>,
    out token : !esi.channel<i0>, out value : !esi.channel<i1>) {
    %token_dc = dc.from_esi %token : !esi.channel<i0>
    %value_dc = dc.from_esi %value : !esi.channel<i1>
    %token_esi = dc.to_esi %token_dc : !dc.token
    %value_esi = dc.to_esi %value_dc : !dc.value<i1>
    hw.output %token_esi, %value_esi : !esi.channel<i0>, !esi.channel<i1>
}

// CHECK-LABEL:   hw.module @sink(in 
// CHECK-SAME:                       %[[VAL_0:.*]] : !esi.channel<i0>) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i0
// CHECK:           %[[VAL_3]] = hw.constant true
// CHECK:           hw.output
// CHECK:         }
hw.module @sink(in %token : !dc.token) {
    dc.sink %token
}

// CHECK-LABEL:   hw.module @source(out token : !esi.channel<i0>) {
// CHECK:           %[[VAL_0:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.wrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i0
// CHECK:           %[[VAL_3]] = hw.constant true
// CHECK:           hw.output %[[VAL_1]] : !esi.channel<i0>
// CHECK:         }
hw.module @source(out token : !dc.token) {
    %token = dc.source
    hw.output %token : !dc.token
}

// CHECK-LABEL:   hw.module @merge(in 
// CHECK-SAME:                        %[[VAL_0:.*]] : !esi.channel<i0>, in
// CHECK-SAME:                        %[[VAL_1:.*]] : !esi.channel<i0>, out token : !esi.channel<i1>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_7:.*]] : i0
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = esi.wrap.vr %[[VAL_10:.*]], %[[VAL_11:.*]] : i1
// CHECK:           %[[VAL_11]] = comb.or %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_12:.*]] = hw.constant true
// CHECK:           %[[VAL_10]] = comb.xor %[[VAL_3]], %[[VAL_12]] : i1
// CHECK:           %[[VAL_13:.*]] = comb.and %[[VAL_11]], %[[VAL_9]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_13]], %[[VAL_3]] : i1
// CHECK:           %[[VAL_7]] = comb.and %[[VAL_13]], %[[VAL_10]] : i1
// CHECK:           hw.output %[[VAL_8]] : !esi.channel<i1>
// CHECK:         }
hw.module @merge(in %first : !dc.token, in %second : !dc.token, out token : !dc.value<i1>) {
    %selected = dc.merge %first, %second
    hw.output %selected : !dc.value<i1>
}

// CHECK:  hw.module.extern @ext(in %a : i32, out b : i32)
hw.module.extern @ext(in %a : i32, out b : i32)

// CHECK:  hw.module.extern @extDC(in %a : !esi.channel<i32>, out b : i32)
hw.module.extern @extDC(in %a : !dc.value<i32>, out b : i32)
