// RUN: circt-opt %s | circt-opt | FileCheck %s

!bundleType = !esi.bundle<[
  !esi.channel<i8> to "data",
  !esi.channel<none> from ack]>

// CHECK-LABEL: hw.module @Receiver(%foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<none>
// CHECK-NEXT:     [[R1:%.+]] = esi.bundle.unpack [[R0]] from %foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>
hw.module @Receiver(%foo: !bundleType) {
  %ack = esi.null : !esi.channel<none>
  %data = esi.bundle.unpack %ack from %foo : !bundleType
}

// CHECK-LABEL:  hw.module @Sender() -> (foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<i8>
// CHECK-NEXT:     %bundle, %fromChannels = esi.bundle.pack [[R0]] : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>
// CHECK-NEXT:     hw.output %bundle : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>
hw.module @Sender() -> (foo: !bundleType) {
  %data = esi.null : !esi.channel<i8>
  %bundle, %ack = esi.bundle.pack %data : !bundleType
  hw.output %bundle : !bundleType
}

// CHECK-LABEL:  hw.module @Top() {
// CHECK-NEXT:     %sender.foo = hw.instance "sender" @Sender() -> (foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>)
// CHECK-NEXT:     hw.instance "receiver" @Receiver(foo: %sender.foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>) -> ()
hw.module @Top() {
  %b = hw.instance "sender" @Sender() -> (foo: !bundleType)
  hw.instance "receiver" @Receiver(foo: %b: !bundleType) -> ()
}

// CHECK-LABEL:  hw.module.extern @ResettableBundleModule(%foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"] reset >)
hw.module.extern @ResettableBundleModule(%foo: !esi.bundle<[
  !esi.channel<i8> to "data",
  !esi.channel<none> from ack] reset>)
