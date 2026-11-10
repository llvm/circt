// RUN: circt-opt %s --canonicalize | circt-opt | FileCheck %s
// RUN: circt-opt --lower-esi-bundles %s | circt-opt | FileCheck %s --check-prefix=LOWER

!bundleType = !esi.bundle<[
  !esi.channel<i8, ValidReady> to "data",
  !esi.channel<none, ValidReady> from ack,
  !esi.channel<i32, ValidReady> to addr]>

// CHECK-LABEL: hw.module @Receiver(in %foo : !esi.bundle<[!esi.channel<i8, ValidReady> to "data", !esi.channel<none, ValidReady> from "ack", !esi.channel<i32, ValidReady> to "addr"]>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<none, ValidReady>
// CHECK-NEXT:     %data, %addr = esi.bundle.unpack [[R0]] from %foo : !esi.bundle<[!esi.channel<i8, ValidReady> to "data", !esi.channel<none, ValidReady> from "ack", !esi.channel<i32, ValidReady> to "addr"]>

// LOWER-LABEL:  hw.module @Receiver(in %foo_data : !esi.channel<i8, ValidReady>, in %foo_addr : !esi.channel<i32, ValidReady>, out foo_ack : !esi.channel<none, ValidReady>)
// LOWER-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<none, ValidReady>
// LOWER-NEXT:     hw.output [[R0]] : !esi.channel<none, ValidReady>
hw.module @Receiver(in %foo: !bundleType) {
  %ack = esi.null : !esi.channel<none, ValidReady>
  %data, %addr = esi.bundle.unpack %ack from %foo : !bundleType
}

// CHECK-LABEL:  hw.module @Sender(out foo : !esi.bundle<[!esi.channel<i8, ValidReady> to "data", !esi.channel<none, ValidReady> from "ack", !esi.channel<i32, ValidReady> to "addr"]>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<i8, ValidReady>
// CHECK-NEXT:     [[R1:%.+]] = esi.null : !esi.channel<i32, ValidReady>
// CHECK-NEXT:     %bundle, %ack = esi.bundle.pack [[R0]], [[R1]] : !esi.bundle<[!esi.channel<i8, ValidReady> to "data", !esi.channel<none, ValidReady> from "ack", !esi.channel<i32, ValidReady> to "addr"]>
// CHECK-NEXT:     hw.output %bundle : !esi.bundle<[!esi.channel<i8, ValidReady> to "data", !esi.channel<none, ValidReady> from "ack", !esi.channel<i32, ValidReady> to "addr"]>

// LOWER-LABEL:  hw.module @Sender(in %foo_ack : !esi.channel<none, ValidReady>, out foo_data : !esi.channel<i8, ValidReady>, out foo_addr : !esi.channel<i32, ValidReady>) {
// LOWER-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<i8, ValidReady>
// LOWER-NEXT:     [[R1:%.+]] = esi.null : !esi.channel<i32, ValidReady>
// LOWER-NEXT:     hw.output [[R0]], [[R1]] : !esi.channel<i8, ValidReady>, !esi.channel<i32, ValidReady>
hw.module @Sender(out foo: !bundleType) {
  %data = esi.null : !esi.channel<i8, ValidReady>
  %addr = esi.null : !esi.channel<i32, ValidReady>
  %bundle, %ack = esi.bundle.pack %data, %addr : !bundleType
  hw.output %bundle : !bundleType
}

// CHECK-LABEL:  hw.module @Top() {
// CHECK-NEXT:     %sender.foo = hw.instance "sender" @Sender() -> (foo: !esi.bundle<[!esi.channel<i8, ValidReady> to "data", !esi.channel<none, ValidReady> from "ack", !esi.channel<i32, ValidReady> to "addr"]>)
// CHECK-NEXT:     hw.instance "receiver" @Receiver(foo: %sender.foo: !esi.bundle<[!esi.channel<i8, ValidReady> to "data", !esi.channel<none, ValidReady> from "ack", !esi.channel<i32, ValidReady> to "addr"]>) -> ()

// LOWER-LABEL:  hw.module @Top() {
// LOWER-NEXT:     %sender.foo_data, %sender.foo_addr = hw.instance "sender" @Sender(foo_ack: %receiver.foo_ack: !esi.channel<none, ValidReady>) -> (foo_data: !esi.channel<i8, ValidReady>, foo_addr: !esi.channel<i32, ValidReady>)
// LOWER-NEXT:     %receiver.foo_ack = hw.instance "receiver" @Receiver(foo_data: %sender.foo_data: !esi.channel<i8, ValidReady>, foo_addr: %sender.foo_addr: !esi.channel<i32, ValidReady>) -> (foo_ack: !esi.channel<none, ValidReady>)
hw.module @Top() {
  %b = hw.instance "sender" @Sender() -> (foo: !bundleType)
  hw.instance "receiver" @Receiver(foo: %b: !bundleType) -> ()
}

// CHECK-LABEL:  hw.module.extern @ResettableBundleModule(in %foo : !esi.bundle<[!esi.channel<i8, ValidReady> to "data", !esi.channel<none, ValidReady> from "ack"] reset >)
// LOWER-LABEL:  hw.module.extern @ResettableBundleModule(in %foo_data : !esi.channel<i8, ValidReady>, out foo_ack : !esi.channel<none, ValidReady>)
hw.module.extern @ResettableBundleModule(in %foo: !esi.bundle<[
  !esi.channel<i8, ValidReady> to "data",
  !esi.channel<none, ValidReady> from ack] reset>)

// LOWER-LABEL:  hw.module @BundleTest(in %s1_in : !esi.channel<i32, ValidReady>, in %b_send_resp : !esi.channel<i1, ValidReady>, out b_send_req : !esi.channel<i32, ValidReady>, out i1_out : !esi.channel<i1, ValidReady>) {
// LOWER-NEXT:     hw.output %s1_in, %b_send_resp : !esi.channel<i32, ValidReady>, !esi.channel<i1, ValidReady>
hw.module @BundleTest(in %s1_in : !esi.channel<i32, ValidReady>, out b_send : !esi.bundle<[!esi.channel<i32, ValidReady> to "req", !esi.channel<i1, ValidReady> from "resp"]>, out i1_out : !esi.channel<i1, ValidReady>) {
  %bundle, %resp = esi.bundle.pack %s1_in : !esi.bundle<[!esi.channel<i32, ValidReady> to "req", !esi.channel<i1, ValidReady> from "resp"]>
  hw.output %bundle, %resp: !esi.bundle<[!esi.channel<i32, ValidReady> to "req", !esi.channel<i1, ValidReady> from "resp"]>, !esi.channel<i1, ValidReady>
}

hw.module @TestNullFold() {
  %c0_i0 = hw.constant 0 : i0
  %false = hw.constant false
  %chanOutput, %ready = esi.wrap.vr %c0_i0, %false: !esi.channel<i0, ValidReady>
}
