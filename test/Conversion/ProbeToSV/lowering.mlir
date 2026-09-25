// RUN: circt-opt %s --lower-probe-to-sv | FileCheck %s

// CHECK: hw.hierpath private @[[LOCAL_PATH:[A-Za-z0-9_]+]] [@Local::@[[LOCAL_WIRE:[A-Za-z0-9_]+]]]
// CHECK: hw.hierpath private @[[CHILD_PATH:[A-Za-z0-9_]+]] [@Top::@[[CHILD_INST:[A-Za-z0-9_]+]], @Producer::@[[CHILD_WIRE:[A-Za-z0-9_]+]]]
// CHECK: hw.hierpath private @[[CHILD_PATH2:[A-Za-z0-9_]+]] [@Top::@[[CHILD_INST2:[A-Za-z0-9_]+]], @Producer::@[[CHILD_WIRE]]]

// CHECK-LABEL: hw.module @Local
hw.module @Local(in %in: i8, out out: i8) {
  %sum = comb.add %in, %in : i8
  %probe = probe.send %sum : i8
  %observed = probe.read %probe : <i8>
  hw.output %observed : i8
}
// CHECK: %[[LOCAL_VALUE:.+]] = comb.add
// CHECK: hw.wire %[[LOCAL_VALUE]] sym @[[LOCAL_WIRE]] : i8
// CHECK: %[[LOCAL_XMR:.+]] = sv.xmr.ref @[[LOCAL_PATH]] : !hw.inout<i8>
// CHECK: %[[LOCAL_READ:.+]] = sv.read_inout %[[LOCAL_XMR]] : !hw.inout<i8>
// CHECK: hw.output %[[LOCAL_READ]] : i8
// CHECK-NOT: probe.

// CHECK-LABEL: hw.module private @Producer
// CHECK-SAME: in %in : i8 {hw.verilogName = "kept_in"}
// CHECK-SAME: out passthrough : i8 {hw.verilogName = "kept_out"}
hw.module private @Producer(
    in %in: i8 {hw.verilogName = "kept_in"},
    out passthrough: i8 {hw.verilogName = "kept_out"},
    out observed: !probe.ref<i8>) {
  %value = comb.xor %in, %in : i8
  %probe = probe.send %value : i8
  hw.output %in, %probe : i8, !probe.ref<i8>
}
// CHECK: %[[CHILD_VALUE:.+]] = comb.xor
// CHECK: hw.wire %[[CHILD_VALUE]] sym @[[CHILD_WIRE]] : i8
// CHECK: hw.output %in : i8
// CHECK-NOT: probe.

// CHECK-LABEL: hw.module @Top
hw.module @Top(in %in: i8, out first: i8, out second: i8,
               out passthrough: i8) {
  %pass0, %probe0 = hw.instance "producer0" @Producer(in: %in: i8) ->
      (passthrough: i8, observed: !probe.ref<i8>)
  %pass1, %probe1 = hw.instance "producer1" @Producer(in: %in: i8) ->
      (passthrough: i8, observed: !probe.ref<i8>) {test.attribute = 42 : i64}
  %first = probe.read %probe0 : <i8>
  %firstAgain = probe.read %probe0 : <i8>
  %second = probe.read %probe1 : <i8>
  %combined = comb.xor %first, %firstAgain : i8
  hw.output %combined, %second, %pass1 : i8, i8, i8
}
// CHECK: %[[PASS0:.+]] = hw.instance "producer0" sym @[[CHILD_INST]] @Producer(in: %in: i8) -> (passthrough: i8)
// CHECK: %[[PASS1:.+]] = hw.instance "producer1" sym @[[CHILD_INST2]] @Producer(in: %in: i8) -> (passthrough: i8) {test.attribute = 42 : i64}
// CHECK: %[[XMR0:.+]] = sv.xmr.ref @[[CHILD_PATH]] : !hw.inout<i8>
// CHECK: sv.read_inout %[[XMR0]] : !hw.inout<i8>
// CHECK: %[[XMR0_AGAIN:.+]] = sv.xmr.ref @[[CHILD_PATH]] : !hw.inout<i8>
// CHECK: sv.read_inout %[[XMR0_AGAIN]] : !hw.inout<i8>
// CHECK: %[[XMR1:.+]] = sv.xmr.ref @[[CHILD_PATH2]] : !hw.inout<i8>
// CHECK: sv.read_inout %[[XMR1]] : !hw.inout<i8>
// CHECK: hw.output {{.*}}, {{.*}}, %[[PASS1]] : i8, i8, i8
// CHECK-NOT: probe.

// CHECK-LABEL: hw.module private @UnusedProducer
// CHECK-NOT: hw.wire
// CHECK-NOT: probe.
hw.module private @UnusedProducer(in %in: i8,
                                  out unused: !probe.ref<i8>) {
  %probe = probe.send %in : i8
  hw.output %probe : !probe.ref<i8>
}

// CHECK-LABEL: hw.module @UnusedInstance
// CHECK: hw.instance "unused" @UnusedProducer(in: %in: i8) -> () {doNotPrint}
hw.module @UnusedInstance(in %in: i8) {
  %unused = hw.instance "unused" @UnusedProducer(in: %in: i8) ->
      (unused: !probe.ref<i8>) {doNotPrint}
}

// CHECK-LABEL: hw.module @Aggregate
hw.module @Aggregate(
    in %in: !hw.struct<a: i8, b: !hw.array<2xi1>>,
    out out: !hw.struct<a: i8, b: !hw.array<2xi1>>) {
  %probe = probe.send %in : !hw.struct<a: i8, b: !hw.array<2xi1>>
  %read = probe.read %probe : <!hw.struct<a: i8, b: !hw.array<2xi1>>>
  hw.output %read : !hw.struct<a: i8, b: !hw.array<2xi1>>
}
// CHECK: hw.wire %in sym @{{[A-Za-z0-9_]+}} : !hw.struct<a: i8, b: !hw.array<2xi1>>
// CHECK: sv.xmr.ref @{{[A-Za-z0-9_]+}} : !hw.inout<struct<a: i8, b: !hw.array<2xi1>>>
// CHECK: sv.read_inout {{.*}} : !hw.inout<struct<a: i8, b: !hw.array<2xi1>>>
// CHECK-NOT: probe.
