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

// CHECK-LABEL: firrtl.module @ConstVec(in %a: !firrtl.const.vector<uint<1>, 3>) {
firrtl.module @ConstVec(in %a: !firrtl.const.vector<uint<1>, 3>) {}

// CHECK-LABEL: firrtl.module @ConstBundle(in %a: !firrtl.const.bundle<a: uint<1>, b: sint<2>>) {
firrtl.module @ConstBundle(in %a: !firrtl.const.bundle<a: uint<1>, b: sint<2>>) {}

// CHECK-LABEL: firrtl.module @MixedConstBundle(in %a: !firrtl.bundle<a: uint<1>, b: const.sint<2>>) {
firrtl.module @MixedConstBundle(in %a: !firrtl.bundle<a: uint<1>, b: const.sint<2>>) {}

}
