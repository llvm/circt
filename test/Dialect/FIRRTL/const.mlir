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

// Subfield of a const bundle should always have a const result
// CHECK-LABEL: firrtl.module @ConstSubfield
firrtl.module @ConstSubfield(in %a: !firrtl.const.bundle<a: uint<1>, b: sint<2>>, out %b: !firrtl.const.uint<1>) {
  // CHECK-NEXT: [[VAL:%.+]] = firrtl.subfield %a[a] : !firrtl.const.bundle<a: uint<1>, b: sint<2>>
  // CHECK-NEXT: firrtl.connect %b, [[VAL]] : !firrtl.const.uint<1>, !firrtl.const.uint<1>
  %0 = firrtl.subfield %a[a] : !firrtl.const.bundle<a: uint<1>, b: sint<2>>
  firrtl.connect %b, %0 : !firrtl.const.uint<1>, !firrtl.const.uint<1>
}

// Subfield of a mixed const bundle should always the same constness as the field type
// CHECK-LABEL: firrtl.module @MixedConstSubfield
firrtl.module @MixedConstSubfield(in %a: !firrtl.bundle<a: uint<1>, b: const.sint<2>>, out %b: !firrtl.uint<1>, out %c: !firrtl.const.sint<2>) {
  // CHECK-NEXT: [[VAL0:%.+]] = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<1>, b: const.sint<2>>
  // CHECK-NEXT: [[VAL1:%.+]] = firrtl.subfield %a[b] : !firrtl.bundle<a: uint<1>, b: const.sint<2>>
  // CHECK-NEXT: firrtl.connect %b, [[VAL0]] : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %c, [[VAL1]] : !firrtl.const.sint<2>, !firrtl.const.sint<2>
  %0 = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<1>, b: const.sint<2>>
  %1 = firrtl.subfield %a[b] : !firrtl.bundle<a: uint<1>, b: const.sint<2>>
  firrtl.connect %b, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %c, %1 : !firrtl.const.sint<2>, !firrtl.const.sint<2>
}

// Subindex of a const vector should always have a const result
// CHECK-LABEL: firrtl.module @ConstSubindex
firrtl.module @ConstSubindex(in %a: !firrtl.const.vector<uint<1>, 3>, out %b: !firrtl.const.uint<1>) {
  // CHECK-NEXT: [[VAL:%.+]] = firrtl.subindex %a[1] : !firrtl.const.vector<uint<1>, 3>
  // CHECK-NEXT: firrtl.connect %b, [[VAL]] : !firrtl.const.uint<1>, !firrtl.const.uint<1>
  %0 = firrtl.subindex %a[1] : !firrtl.const.vector<uint<1>, 3>
  firrtl.connect %b, %0 : !firrtl.const.uint<1>, !firrtl.const.uint<1>
}

// Subindex of a non-const vector with a const element type should always have a const result
// CHECK-LABEL: firrtl.module @ConstElementSubindex
firrtl.module @ConstElementSubindex(in %a: !firrtl.vector<const.uint<1>, 3>, out %b: !firrtl.const.uint<1>) {
  // CHECK-NEXT: [[VAL:%.+]] = firrtl.subindex %a[1] : !firrtl.vector<const.uint<1>, 3>
  // CHECK-NEXT: firrtl.connect %b, [[VAL]] : !firrtl.const.uint<1>, !firrtl.const.uint<1>
  %0 = firrtl.subindex %a[1] : !firrtl.vector<const.uint<1>, 3>
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

// Subaccess of a non-const vector with a const element type should be const only if the index is const
// CHECK-LABEL: firrtl.module @ConstElementSubaccess
firrtl.module @ConstElementSubaccess(in %a: !firrtl.vector<const.uint<1>, 3>, in %constIndex: !firrtl.const.uint<4>, in %dynamicIndex: !firrtl.uint<4>, out %constOut: !firrtl.const.uint<1>, out %dynamicOut: !firrtl.uint<1>) {
  // CHECK-NEXT: [[VAL0:%.+]] = firrtl.subaccess %a[%constIndex] : !firrtl.vector<const.uint<1>, 3>, !firrtl.const.uint<4>
  // CHECK-NEXT: [[VAL1:%.+]] = firrtl.subaccess %a[%dynamicIndex] : !firrtl.vector<const.uint<1>, 3>, !firrtl.uint<4>
  // CHECK-NEXT: firrtl.connect %constOut, [[VAL0]] : !firrtl.const.uint<1>, !firrtl.const.uint<1>
  // CHECK-NEXT: firrtl.connect %dynamicOut, [[VAL1]] : !firrtl.uint<1>, !firrtl.uint<1>
  %0 = firrtl.subaccess %a[%constIndex] : !firrtl.vector<const.uint<1>, 3>, !firrtl.const.uint<4>
  %1 = firrtl.subaccess %a[%dynamicIndex] : !firrtl.vector<const.uint<1>, 3>, !firrtl.uint<4>
  firrtl.connect %constOut, %0 : !firrtl.const.uint<1>, !firrtl.const.uint<1>
  firrtl.connect %dynamicOut, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// Subaccess of a non-const vector with a nested const element type should preserve the subelement constness
// only if the index is const
// CHECK-LABEL: firrtl.module @ConstNestedElementSubaccess
firrtl.module @ConstNestedElementSubaccess(in %a: !firrtl.vector<bundle<a: const.uint<1>>, 3>, in %constIndex: !firrtl.const.uint<4>, in %dynamicIndex: !firrtl.uint<4>) {
  // CHECK-NEXT: [[VAL0:%.+]] = firrtl.subaccess %a[%constIndex] : !firrtl.vector<bundle<a: const.uint<1>>, 3>, !firrtl.const.uint<4>
  // CHECK-NEXT: [[VAL1:%.+]] = firrtl.subfield [[VAL0]][a] : !firrtl.bundle<a: const.uint<1>>
  %0 = firrtl.subaccess %a[%constIndex] : !firrtl.vector<bundle<a: const.uint<1>>, 3>, !firrtl.const.uint<4>
  %1 = firrtl.subfield %0[a] : !firrtl.bundle<a: const.uint<1>>

  // CHECK-NEXT: [[VAL2:%.+]] = firrtl.subaccess %a[%dynamicIndex] : !firrtl.vector<bundle<a: const.uint<1>>, 3>, !firrtl.uint<4>
  // CHECK-NEXT: [[VAL3:%.+]] = firrtl.subfield [[VAL2]][a] : !firrtl.bundle<a: uint<1>>
  %2 = firrtl.subaccess %a[%dynamicIndex] : !firrtl.vector<bundle<a: const.uint<1>>, 3>, !firrtl.uint<4>
  %3 = firrtl.subfield %2[a] : !firrtl.bundle<a: uint<1>>
}

// CHECK-LABEL: firrtl.module @ConstSubtag
firrtl.module @ConstSubtag(in %in : !firrtl.const.enum<a: uint<1>, b: uint<2>>,
                           out %out : !firrtl.const.uint<2>) {
  // CHECK-NEXT: [[VAL:%.+]] = firrtl.subtag %in[b] : !firrtl.const.enum<a: uint<1>, b: uint<2>>
  // CHECK-NEXT: firrtl.connect %out, [[VAL]] : !firrtl.const.uint<2>, !firrtl.const.uint<2>
  %0 = firrtl.subtag %in[b] : !firrtl.const.enum<a: uint<1>, b: uint<2>>
  firrtl.connect %out, %0 : !firrtl.const.uint<2>, !firrtl.const.uint<2>
}

// CHECK-LABEL: firrtl.module @MixedConstSubtag
firrtl.module @MixedConstSubtag(in %in : !firrtl.enum<a: uint<1>, b: const.uint<2>>,
                                out %a : !firrtl.uint<1>,
                                out %b : !firrtl.const.uint<2>) {
  // CHECK-NEXT: [[VAL0:%.+]] = firrtl.subtag %in[a] : !firrtl.enum<a: uint<1>, b: const.uint<2>>
  // CHECK-NEXT: [[VAL1:%.+]] = firrtl.subtag %in[b] : !firrtl.enum<a: uint<1>, b: const.uint<2>>
  // CHECK-NEXT: firrtl.connect %a, [[VAL0]] : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %b, [[VAL1]] : !firrtl.const.uint<2>, !firrtl.const.uint<2>
  %0 = firrtl.subtag %in[a] : !firrtl.enum<a: uint<1>, b: const.uint<2>>
  %1 = firrtl.subtag %in[b] : !firrtl.enum<a: uint<1>, b: const.uint<2>>
  firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.const.uint<2>, !firrtl.const.uint<2>
}

// CHECK-LABEL: firrtl.module @ConstRegResetValue
firrtl.module @ConstRegResetValue(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetValue: !firrtl.const.sint<1>) {
  %0 = firrtl.regreset %clock, %reset, %resetValue : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.sint<1>, !firrtl.sint<1>
}

// CHECK-LABEL: firrtl.module @ConstCast
firrtl.module @ConstCast(in %in: !firrtl.const.uint<1>, out %out: !firrtl.uint<1>) {
  %0 = firrtl.constCast %in : (!firrtl.const.uint<1>) -> !firrtl.uint<1>
  firrtl.strictconnect %out, %0 : !firrtl.uint<1> 
}

// CHECK-LABEL: firrtl.module @ConstCastToMixedConstBundle
firrtl.module @ConstCastToMixedConstBundle(in %in: !firrtl.const.bundle<a: uint<1>>, out %out: !firrtl.bundle<a: const.uint<1>>) {
  %0 = firrtl.constCast %in : (!firrtl.const.bundle<a: uint<1>>) -> !firrtl.bundle<a: const.uint<1>>
  firrtl.strictconnect %out, %0 : !firrtl.bundle<a: const.uint<1>>
}

// CHECK-LABEL: firrtl.module @ConstCastToMixedConstVector
firrtl.module @ConstCastToMixedConstVector(in %in: !firrtl.const.vector<uint<1>, 2>, out %out: !firrtl.vector<const.uint<1>, 2>) {
  %0 = firrtl.constCast %in : (!firrtl.const.vector<uint<1>, 2>) -> !firrtl.vector<const.uint<1>, 2>
  firrtl.strictconnect %out, %0 : !firrtl.vector<const.uint<1>, 2>
}

}
