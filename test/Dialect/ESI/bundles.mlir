// RUN: circt-opt %s | circt-opt | FileCheck %s

!bundleType = !esi.bundle<[
  !esi.channel<i8> to "data",
  !esi.channel<none> from ack]>

// CHECK-LABEL: hw.module @Receiver(in %foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>, out data : !esi.channel<i8>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<none>
// CHECK-NEXT:     %data = esi.bundle.unpack [[R0]] from %foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>
hw.module @Receiver(in %foo: !bundleType, out data: !esi.channel<i8>) {
  %ack = esi.null : !esi.channel<none>
  %data = esi.bundle.unpack %ack from %foo : !bundleType
  hw.output %data : !esi.channel<i8>
}

// CHECK-LABEL:  hw.module @Sender(out foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>, out ack : !esi.channel<none>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<i8>
// CHECK-NEXT:     %bundle, %ack = esi.bundle.pack [[R0]] : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>
// CHECK-NEXT:     hw.output %bundle, %ack : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>, !esi.channel<none>
hw.module @Sender(out foo: !bundleType, out ack: !esi.channel<none>) {
  %data = esi.null : !esi.channel<i8>
  %bundle, %ack = esi.bundle.pack %data : !bundleType
  hw.output %bundle, %ack : !bundleType, !esi.channel<none>
}

// CHECK-LABEL:  hw.module @Top() {
// CHECK-NEXT:     %sender.foo, %sender.ack = hw.instance "sender" @Sender() -> (foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>, ack: !esi.channel<none>)
// CHECK-NEXT:     %receiver.data = hw.instance "receiver" @Receiver(foo: %sender.foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>) -> (data: !esi.channel<i8>)
hw.module @Top() {
  %b, %ack = hw.instance "sender" @Sender() -> (foo: !bundleType, ack: !esi.channel<none>)
  %receiver.data = hw.instance "receiver" @Receiver(foo: %b: !bundleType) -> (data: !esi.channel<i8>)
}

// CHECK-LABEL:  hw.module.extern @ResettableBundleModule(in %foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"] reset >)
hw.module.extern @ResettableBundleModule(in %foo: !esi.bundle<[
  !esi.channel<i8> to "data",
  !esi.channel<none> from ack] reset>)
