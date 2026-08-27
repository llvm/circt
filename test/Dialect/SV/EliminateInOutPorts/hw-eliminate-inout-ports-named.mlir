// RUN: circt-opt --hw-eliminate-inout-ports="read-suffix= write-suffix=" %s | FileCheck %s

// CHECK-LABEL:   hw.module @read(in %rd : i42, out out : i42)
hw.module @read(inout %rd: i42, out out: i42) {
  %rd_net = sv.net.from_inout %rd : !hw.inout<i42> -> !sv.net<i42>
  %aget = sv.read_inout %rd_net: !sv.net<i42>
  hw.output %aget : i42
}

// CHECK-LABEL: hw.module @write(out wr : i42)
hw.module @write(inout %wr: i42) {
  %0 = hw.constant 0 : i42
  %wr_net = sv.net.from_inout %wr : !hw.inout<i42> -> !sv.net<i42>
  sv.assign %wr_net, %0 : i42
}

// CHECK-LABEL: hw.module @oneLevel()
// CHECK:           %[[x:.*]] = hw.instance "read" @read(rd: %[[x:.*]]: i42) -> (out: i42)
// CHECK:           %[[x:.*]] = hw.instance "write" @write() -> (wr: i42)
hw.module @oneLevel() {
  %0 = sv.wire : !sv.net<i42>
  %wire_io = sv.inout.from_net %0 : !sv.net<i42> -> !hw.inout<i42>
  %read = hw.instance "read" @read(rd : %wire_io : !hw.inout<i42>) -> (out: i42)
  hw.instance "write" @write(wr : %wire_io : !hw.inout<i42>) -> ()
}
