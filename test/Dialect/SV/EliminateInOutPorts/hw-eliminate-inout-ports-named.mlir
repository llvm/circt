// RUN: circt-opt --hw-eliminate-inout-ports="read-suffix= write-suffix=" %s | FileCheck %s

// CHECK-LABEL:   hw.module @read(in %rd : i42, out out : i42)
hw.module @read(inout %rd: i42, out out: i42) {
  %aget = sv.read_inout %rd: !hw.inout<i42>
  hw.output %aget : i42
}

// CHECK-LABEL: hw.module @write(out wr : i42)
hw.module @write(inout %wr: i42) {
  %0 = hw.constant 0 : i42
  sv.assign %wr, %0 : i42
}

// CHECK-LABEL: hw.module @oneLevel()
// CHECK:           %[[x:.*]] = hw.instance "read" @read(rd: %[[x:.*]]: i42) -> (out: i42)
// CHECK:           %[[x:.*]] = hw.instance "write" @write() -> (wr: i42)
hw.module @oneLevel() {
  %0 = sv.wire : !hw.inout<i42>
  %read = hw.instance "read" @read(rd : %0 : !hw.inout<i42>) -> (out: i42)
  hw.instance "write" @write(wr : %0 : !hw.inout<i42>) -> ()
}
