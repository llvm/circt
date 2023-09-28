// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @foo(
// CHECK-SAME:                   in %[[VAL_0:.*]] : !esi.channel<i32>,
// CHECK-SAME:                   in %[[CLOCK:.*]] : !seq.clock,
// CHECK-SAME:                   in %[[VAL_2:.*]] : i1, out out0 : !esi.channel<i32>) {
// CHECK:           hw.output %[[VAL_0]] : !esi.channel<i32>
// CHECK:         }

// CHECK:   hw.module @bar(in %[[VAL_0:.*]] : !esi.channel<i32>, in %[[CLOCK:.*]] : !seq.clock, in %[[VAL_2:.*]] : i1, out out0 : !esi.channel<i32>) {
// CHECK:           %[[VAL_3:.*]] = hw.instance "foo0" @foo(in: %[[VAL_0]]: !esi.channel<i32>, clock: %[[CLOCK]]: !seq.clock, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i32>)
// CHECK:           hw.output %[[VAL_3]] : !esi.channel<i32>
// CHECK:         }
handshake.func @foo(%in : i32) -> (i32) {
    handshake.return %in : i32
}

handshake.func @bar(%in : i32) -> (i32) {
    %out = handshake.instance @foo(%in) : (i32) -> (i32)
    handshake.return %out : i32
}

// -----

// CHECK:         hw.module.extern @foo(in %[[VAL_4:.*]] : !esi.channel<i32>, in %[[CLOCK:.*]] : !seq.clock, in %[[VAL_6:.*]] : i1, out out0 : !esi.channel<i32>)

// CHECK:   hw.module @bar(in %[[VAL_4]] : !esi.channel<i32>, in %[[CLOCK]] : !seq.clock, in %[[VAL_6]] : i1, out out0 : !esi.channel<i32>) {
// CHECK:           %[[VAL_0:.*]] = hw.instance "foo0" @foo(in: %[[VAL_4]]: !esi.channel<i32>, clock: %[[CLOCK]]: !seq.clock, reset: %[[VAL_6]]: i1) -> (out0: !esi.channel<i32>)
// CHECK:           hw.output %[[VAL_0]] : !esi.channel<i32>
// CHECK:         }
handshake.func @foo(%in : i32) -> (i32)

handshake.func @bar(%in : i32) -> (i32) {
    %out = handshake.instance @foo(%in) : (i32) -> (i32)
    handshake.return %out : i32
}
