// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt --lower-esi-bundles %s | circt-opt | FileCheck %s --check-prefix=LOWER

!bundleType = !esi.bundle<[
  !esi.channel<i8> to "data",
  !esi.channel<none> from ack]>

// CHECK-LABEL: hw.module @Receiver(%foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<none>
// CHECK-NEXT:     %data = esi.bundle.unpack [[R0]] from %foo : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>

// LOWER-LABEL:  hw.module @Receiver(%foo_data: !esi.channel<i8>) -> (foo_ack: !esi.channel<none>) {
// LOWER-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<none>
// LOWER-NEXT:     hw.output [[R0] : !esi.channel<none>
hw.module @Receiver(%foo: !bundleType) -> () {
  %ack = esi.null : !esi.channel<none>
  %data = esi.bundle.unpack %ack from %foo : !bundleType
}

// CHECK-LABEL:  hw.module @Sender() -> (foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>) {
// CHECK-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<i8>
// CHECK-NEXT:     %bundle, %ack = esi.bundle.pack [[R0]] : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>
// CHECK-NEXT:     hw.output %bundle : !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>

// LOWER-LABEL:  hw.module @Sender(%foo_ack: !esi.channel<none>) -> (foo_data: !esi.channel<i8>) {
// LOWER-NEXT:     [[R0:%.+]] = esi.null : !esi.channel<i8>
// LOWER-NEXT:     hw.output [[R0]] : !esi.channel<i8>
hw.module @Sender() -> (foo: !bundleType) {
  %data = esi.null : !esi.channel<i8>
  %bundle, %ack = esi.bundle.pack %data : !bundleType
  hw.output %bundle : !bundleType
}

// CHECK-LABEL:  hw.module @Top() {
// CHECK-NEXT:     %sender.foo, %sender.ack = hw.instance "sender" @Sender() -> (foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>, ack: !esi.channel<none>)
// CHECK-NEXT:     %receiver.data = hw.instance "receiver" @Receiver(foo: %sender.foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"]>) -> (data: !esi.channel<i8>)

// LOWER-LABEL:  hw.module @Top() {
// LOWER-NEXT:     %sender.foo_data, %sender.ack = hw.instance "sender" @Sender(foo_ack: %receiver.foo_ack: !esi.channel<none>) -> (foo_data: !esi.channel<i8>, ack: !esi.channel<none>)
// LOWER-NEXT:     %receiver.foo_ack, %receiver.data = hw.instance "receiver" @Receiver(foo_data: %sender.foo_data: !esi.channel<i8>) -> (foo_ack: !esi.channel<none>, data: !esi.channel<i8>)
hw.module @Top() {
  %b = hw.instance "sender" @Sender() -> (foo: !bundleType)
  hw.instance "receiver" @Receiver(foo: %b: !bundleType) -> ()
}

// CHECK-LABEL:  hw.module.extern @ResettableBundleModule(%foo: !esi.bundle<[!esi.channel<i8> to "data", !esi.channel<none> from "ack"] reset >)
// LOWER-LABEL:  hw.module.extern @ResettableBundleModule(%foo_data: !esi.channel<i8>) -> (foo_ack: !esi.channel<none>)
hw.module.extern @ResettableBundleModule(%foo: !esi.bundle<[
  !esi.channel<i8> to "data",
  !esi.channel<none> from ack] reset>)
