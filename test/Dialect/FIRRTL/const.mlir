// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "ConstTypes" {
firrtl.module @ConstTypes() {}

// CHECK-LABEL: firrtl.module @ConstUInt(in %a: !firrtl.const.uint<2>) {
firrtl.module @ConstUInt(in %a: !firrtl.const.uint<2>) {}

// CHECK-LABEL: firrtl.module @ConstSInt(in %a: !firrtl.const.sint<2>) {
firrtl.module @ConstSInt(in %a: !firrtl.const.sint<2>) {}

// CHECK-LABEL: firrtl.module @ConstAnalog(in %a: !firrtl.const.analog<2>) {
firrtl.module @ConstAnalog(in %a: !firrtl.const.analog<2>) {}

// CHECK-LABEL: firrtl.module @ConstClock(in %a: !firrtl.const.clock) {
firrtl.module @ConstClock(in %a: !firrtl.const.clock) {}

// CHECK-LABEL: firrtl.module @ConstReset(in %a: !firrtl.const.reset) {
firrtl.module @ConstReset(in %a: !firrtl.const.reset) {}

// CHECK-LABEL: firrtl.module @ConstAsyncReset(in %a: !firrtl.const.asyncreset) {
firrtl.module @ConstAsyncReset(in %a: !firrtl.const.asyncreset) {}

// CHECK-LABEL: firrtl.module @ConstEnum(in %a: !firrtl.enum<a: uint<1>, b: uint<2>>) {
firrtl.module @ConstEnum(in %a: !firrtl.enum<a: uint<1>, b: uint<2>>) {}

// CHECK-LABEL: firrtl.module @ConstVec(in %a: !firrtl.const.vector<uint<1>, 3>) {
firrtl.module @ConstVec(in %a: !firrtl.const.vector<uint<1>, 3>) {}

// CHECK-LABEL: firrtl.module @ConstVecExplicitElements(in %a: !firrtl.const.vector<const.uint<1>, 3>) {
firrtl.module @ConstVecExplicitElements(in %a: !firrtl.const.vector<const.uint<1>, 3>) {}

// CHECK-LABEL: firrtl.module @ConstBundle(in %a: !firrtl.const.bundle<a: uint<1>, b: sint<2>>) {
firrtl.module @ConstBundle(in %a: !firrtl.const.bundle<a: uint<1>, b: sint<2>>) {}

// CHECK-LABEL: firrtl.module @MixedConstBundle(in %a: !firrtl.bundle<a: uint<1>, b: const.sint<2>>) {
firrtl.module @MixedConstBundle(in %a: !firrtl.bundle<a: uint<1>, b: const.sint<2>>) {}

// CHECK-LABEL: firrtl.module @ConstBundleExplicitElements(in %a: !firrtl.const.bundle<a: const.uint<1>, b: const.sint<2>>) {
firrtl.module @ConstBundleExplicitElements(in %a: !firrtl.const.bundle<a: const.uint<1>, b: const.sint<2>>) {}

// Subaccess of a const vector should always have a const result
// CHECK-LABEL: firrtl.module @ConstSubindex
firrtl.module @ConstSubindex(in %a: !firrtl.const.vector<uint<1>, 3>, out %b: !firrtl.const.uint<1>) {
  // CHECK-NEXT: [[VAL:%.+]] = firrtl.subindex %a[1] : !firrtl.const.vector<uint<1>, 3>
  // CHECK-NEXT: firrtl.connect %b, [[VAL]] : !firrtl.const.uint<1>, !firrtl.const.uint<1>
  %0 = firrtl.subindex %a[1] : !firrtl.const.vector<uint<1>, 3>
  firrtl.connect %b, %0 : !firrtl.const.uint<1>, !firrtl.const.uint<1>
}

// Subaccess of a const vector should be const only if the index is const
// CHECK-LABEL: firrtl.module @ConstSubaccess
firrtl.module @ConstSubaccess(in %a: !firrtl.const.vector<uint<1>, 3>, in %constIndex: !firrtl.const.uint<4>, in %dynamicIndex: !firrtl.uint<4>, out %constOut: !firrtl.const.uint<1>, out %dynamicOut: !firrtl.uint<1>) {
  // CHECK-NEXT: [[VAL0:%.+]] = firrtl.subaccess %a[%constIndex] : !firrtl.const.vector<uint<1>, 3>, !firrtl.const.uint<4>
  // CHECK-NEXT: [[VAL1:%.+]] = firrtl.subaccess %a[%dynamicIndex] : !firrtl.const.vector<uint<1>, 3>, !firrtl.uint<4>
  // CHECK-NEXT: firrtl.connect %constOut, [[VAL0]] : !firrtl.const.uint<1>, !firrtl.const.uint<1>
  // CHECK-NEXT: firrtl.connect %dynamicOut, [[VAL1]] : !firrtl.uint<1>, !firrtl.uint<1>
  %0 = firrtl.subaccess %a[%constIndex] : !firrtl.const.vector<uint<1>, 3>, !firrtl.const.uint<4>
  %1 = firrtl.subaccess %a[%dynamicIndex] : !firrtl.const.vector<uint<1>, 3>, !firrtl.uint<4>
  firrtl.connect %constOut, %0 : !firrtl.const.uint<1>, !firrtl.const.uint<1>
  firrtl.connect %dynamicOut, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @ConstSubtag
firrtl.module @ConstSubtag(in %in : !firrtl.const.enum<a: uint<1>, b: uint<2>>,
                           out %out : !firrtl.const.uint<2>) {
  // CHECK-NEXT: [[VAL:%.+]] = firrtl.subtag %in[b] : !firrtl.const.enum<a: uint<1>, b: uint<2>>
  // CHECK-NEXT: firrtl.strictconnect %out, [[VAL]] : !firrtl.const.uint<2>
  %0 = firrtl.subtag %in[b] : !firrtl.const.enum<a: uint<1>, b: uint<2>>
  firrtl.strictconnect %out, %0 : !firrtl.const.uint<2>
}

}
