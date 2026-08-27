// RUN: circt-opt --hw-eliminate-inout-ports %s | FileCheck %s

// CHECK-LABEL:   hw.module @read(
// CHECK-SAME:                    in %[[VAL_0:.*]] : i42, out out : i42) {
// CHECK:           hw.output %[[VAL_0]] : i42
// CHECK:         }
hw.module @read(inout %a: i42, out out: i42) {
  %a_net = sv.net.from_inout %a : !hw.inout<i42> -> !sv.net<i42>
  %aget = sv.read_inout %a_net: !sv.net<i42>
  hw.output %aget : i42
}

// CHECK-LABEL:   hw.module @write(out a_wr : i42) {
// CHECK:           %[[VAL_0:.*]] = hw.constant 0 : i42
// CHECK:           hw.output %[[VAL_0]] : i42
// CHECK:         }
hw.module @write(inout %a: i42) {
  %0 = hw.constant 0 : i42
  %a_net = sv.net.from_inout %a : !hw.inout<i42> -> !sv.net<i42>
  sv.assign %a_net, %0 : i42
}

// CHECK-LABEL:   hw.module @read_write(
// CHECK-SAME:                          in %[[VAL_0:.*]] : i42, out a_wr : i42, out out : i42) {
// CHECK:           hw.output %[[VAL_0]], %[[VAL_0]] : i42, i42
// CHECK:         }
hw.module @read_write(inout %a: i42, out out: i42) {
  %a_net = sv.net.from_inout %a : !hw.inout<i42> -> !sv.net<i42>
  %aget = sv.read_inout %a_net: !sv.net<i42>
  sv.assign %a_net, %aget : i42
  hw.output %aget : i42
}

// CHECK-LABEL:   hw.module @oneLevel() {
// CHECK:           %[[VAL_0:.*]] = sv.wire : !sv.net<i42>
// CHECK:           %[[VAL_1:.*]] = sv.inout.from_net %[[VAL_0]] : !sv.net<i42> -> !hw.inout<i42>
// CHECK:           %[[VAL_2:.*]] = sv.net.from_inout %[[VAL_1]] : !hw.inout<i42> -> !sv.net<i42>
// CHECK:           %[[VAL_3:.*]] = sv.read_inout %[[VAL_2]] : !sv.net<i42>
// CHECK:           %[[VAL_4:.*]] = hw.instance "read" @read(a_rd: %[[VAL_3]]: i42) -> (out: i42)
// CHECK:           %[[VAL_5:.*]] = sv.net.from_inout %[[VAL_1]] : !hw.inout<i42> -> !sv.net<i42>
// CHECK:           sv.assign %[[VAL_5]], %[[VAL_6:.*]] : i42
// CHECK:           %[[VAL_6]] = hw.instance "write" @write() -> (a_wr: i42)
// CHECK:           %[[VAL_7:.*]] = sv.net.from_inout %[[VAL_1]] : !hw.inout<i42> -> !sv.net<i42>
// CHECK:           %[[VAL_8:.*]] = sv.read_inout %[[VAL_7]] : !sv.net<i42>
// CHECK:           %[[VAL_9:.*]] = sv.net.from_inout %[[VAL_1]] : !hw.inout<i42> -> !sv.net<i42>
// CHECK:           sv.assign %[[VAL_9]], %[[VAL_10:.*]] : i42
// CHECK:           %[[VAL_10]], %[[VAL_11:.*]] = hw.instance "readWrite" @read_write(a_rd: %[[VAL_8]]: i42) -> (a_wr: i42, out: i42)
// CHECK:           hw.output
// CHECK:         }
hw.module @oneLevel() {
  %0 = sv.wire : !sv.net<i42>
  %wire_io = sv.inout.from_net %0 : !sv.net<i42> -> !hw.inout<i42>
  %read = hw.instance "read" @read(a : %wire_io : !hw.inout<i42>) -> (out: i42)
  hw.instance "write" @write(a : %wire_io : !hw.inout<i42>) -> ()
  %read_write = hw.instance "readWrite" @read_write(a : %wire_io : !hw.inout<i42>) -> (out: i42)
}

// CHECK-LABEL:   hw.module @passthrough(out a_wr : i42) {
// CHECK:           %[[VAL_0:.*]] = hw.instance "write" @write() -> (a_wr: i42)
// CHECK:           hw.output %[[VAL_0]] : i42
// CHECK:         }
hw.module @passthrough(inout %a : i42) {
  hw.instance "write" @write(a : %a : !hw.inout<i42>) -> ()
}

// CHECK-LABEL:   hw.module @passthroughTwoLevels() {
// CHECK:           %[[VAL_0:.*]] = sv.wire : !sv.net<i42>
// CHECK:           %[[VAL_1:.*]] = sv.inout.from_net %[[VAL_0]] : !sv.net<i42> -> !hw.inout<i42>
// CHECK:           %[[VAL_2:.*]] = sv.net.from_inout %[[VAL_1]] : !hw.inout<i42> -> !sv.net<i42>
// CHECK:           sv.assign %[[VAL_2]], %[[VAL_3:.*]] : i42
// CHECK:           %[[VAL_3]] = hw.instance "passthrough" @passthrough() -> (a_wr: i42)
// CHECK:           hw.output
// CHECK:         }
hw.module @passthroughTwoLevels() {
  %0 = sv.wire : !sv.net<i42>
  %wire_io = sv.inout.from_net %0 : !sv.net<i42> -> !hw.inout<i42>
  hw.instance "passthrough" @passthrough(a : %wire_io : !hw.inout<i42>) -> ()
}

// CHECK-LABEL:   hw.module @writeInput(
// CHECK-SAME:                          in %[[VAL_0:.*]] : i42, out a_wr : i42) {
// CHECK:           hw.output %[[VAL_0]] : i42
// CHECK:         }
hw.module @writeInput(inout %a: i42, in %in : i42) {
  %a_net = sv.net.from_inout %a : !hw.inout<i42> -> !sv.net<i42>
  sv.assign %a_net, %in : i42
}
