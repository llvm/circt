// RUN: circt-opt %s --canonicalize | circt-opt | FileCheck %s
// RUN: circt-opt --lower-esi-bundles %s | circt-opt | FileCheck %s --check-prefix=LOWER

!bundleType = !esi.bundle<[
  !esi.channel<i8> to "data",
  !esi.channel<none> from ack,
  !esi.channel<i32> to addr]>

// CHECK-LABEL: hw.module @Receiver(in %foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack", !esi.channel<i32> to "addr"]>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<none>
// CHECK-NEXT:     %data, %addr = esi.bundle.unpack [[R0]] from %foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack", !esi.channel<i32> to "addr"]>

// LOWER-LABEL:  hw.module @Receiver(in %foo_data : !esi.channel<i8>, in %foo_addr : !esi.channel<i32>, out foo_ack : !esi.channel<none>)
// LOWER-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<none>
// LOWER-NEXT:     hw.output [[R0]] : !esi.channel<none>
hw.module @Receiver(in %foo: !bundleType) {
  %ack = esi.null : !esi.channel<none>
  %data, %addr = esi.bundle.unpack %ack from %foo : !bundleType
}

// CHECK-LABEL:  hw.module @Sender(out foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack", !esi.channel<i32> to "addr"]>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<i8>
// CHECK-NEXT:     [[R1:%.+]] = esi.null : !esi.channel<i32>
// CHECK-NEXT:     %bundle, %ack = esi.bundle.pack [[R0]], [[R1]] : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack", !esi.channel<i32> to "addr"]>
// CHECK-NEXT:     hw.output %bundle : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack", !esi.channel<i32> to "addr"]>

// LOWER-LABEL:  hw.module @Sender(in %foo_ack : !esi.channel<none>, out foo_data : !esi.channel<i8>, out foo_addr : !esi.channel<i32>) {
// LOWER-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<i8>
// LOWER-NEXT:     [[R1:%.+]] = esi.null : !esi.channel<i32>
// LOWER-NEXT:     hw.output [[R0]], [[R1]] : !esi.channel<i8>, !esi.channel<i32>
hw.module @Sender(out foo: !bundleType) {
  %data = esi.null : !esi.channel<i8>
  %addr = esi.null : !esi.channel<i32>
  %bundle, %ack = esi.bundle.pack %data, %addr : !bundleType
  hw.output %bundle : !bundleType
}

// CHECK-LABEL:  hw.module @Top() {
// CHECK-NEXT:     %sender.foo = hw.instance "sender" @Sender() -> (foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack", !esi.channel<i32> to "addr"]>)
// CHECK-NEXT:     hw.instance "receiver" @Receiver(foo: %sender.foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack", !esi.channel<i32> to "addr"]>) -> ()

// LOWER-LABEL:  hw.module @Top() {
// LOWER-NEXT:     %sender.foo_data, %sender.foo_addr = hw.instance "sender" @Sender(foo_ack: %receiver.foo_ack: !esi.channel<none>) -> (foo_data: !esi.channel<i8>, foo_addr: !esi.channel<i32>)
// LOWER-NEXT:     %receiver.foo_ack = hw.instance "receiver" @Receiver(foo_data: %sender.foo_data: !esi.channel<i8>, foo_addr: %sender.foo_addr: !esi.channel<i32>) -> (foo_ack: !esi.channel<none>)
hw.module @Top() {
  %b = hw.instance "sender" @Sender() -> (foo: !bundleType)
  hw.instance "receiver" @Receiver(foo: %b: !bundleType) -> ()
}

// CHECK-LABEL:  hw.module.extern @ResettableBundleModule(in %foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"] reset >)
// LOWER-LABEL:  hw.module.extern @ResettableBundleModule(in %foo_data : !esi.channel<i8>, out foo_ack : !esi.channel<none>)
hw.module.extern @ResettableBundleModule(in %foo: !esi.bundle<[
  !esi.channel<i8> to "data",
  !esi.channel<none> from ack] reset>)

// LOWER-LABEL:  hw.module @BundleTest(in %s1_in : !esi.channel<i32>, in %b_send_resp : !esi.channel<i1>, out b_send_req : !esi.channel<i32>, out i1_out : !esi.channel<i1>) {
// LOWER-NEXT:     hw.output %s1_in, %b_send_resp : !esi.channel<i32>, !esi.channel<i1>
hw.module @BundleTest(in %s1_in : !esi.channel<i32>, out b_send : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, out i1_out : !esi.channel<i1>) {
  %bundle, %resp = esi.bundle.pack %s1_in : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>
  hw.output %bundle, %resp: !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, !esi.channel<i1>
}
