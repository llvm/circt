// RUN: circt-opt %s --hw-flatten-modules | FileCheck %s

// Test inlining of an empty module
// CHECK-LABEL: hw.module @InlineEmpty
hw.module @InlineEmpty(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: hw.output %x
  hw.instance "empty" @EmptyModule() -> ()
  hw.output %x : i4
}
// CHECK-NOT: hw.module private @EmptyModule
hw.module private @EmptyModule() {
  hw.output
}

// Test inlining of a module with no outputs
// CHECK-LABEL: hw.module @InlineNoOutputs
hw.module @InlineNoOutputs(in %x: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: hw.output
  hw.instance "noout" @NoOutputModule(in: %x: i4) -> ()
  hw.output
}
// CHECK-NOT: hw.module private @NoOutputModule
hw.module private @NoOutputModule(in %in: i4) {
  %0 = comb.add %in, %in : i4
  hw.output
}

// Test that public modules are not inlined
// CHECK-LABEL: hw.module @DontInlinePublic
hw.module @DontInlinePublic(in %x: i4, out y: i4) {
  // CHECK-NEXT: hw.instance "pub" @PublicModule
  %0 = hw.instance "pub" @PublicModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK: hw.module @PublicModule
hw.module @PublicModule(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  hw.output %0 : i4
}

// Test inlining with single use
// CHECK-LABEL: hw.module @InlineSingleUse
hw.module @InlineSingleUse(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.mul %x, %x
  // CHECK-NEXT: hw.output %[[V0]]
  %0 = hw.instance "single" @SingleUseModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @SingleUseModule
hw.module private @SingleUseModule(in %a: i4, out b: i4) {
  %0 = comb.mul %a, %a : i4
  hw.output %0 : i4
}

// Test name prefixing during inlining
// CHECK-LABEL: hw.module @InlineWithNames
hw.module @InlineWithNames(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[C:.+]] = hw.constant 1 : i4 {name = "named/const_val"}
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %[[C]]
  // CHECK-NEXT: hw.output %[[V0]]
  %0 = hw.instance "named" @ModuleWithNames(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @ModuleWithNames
hw.module private @ModuleWithNames(in %a: i4, out b: i4) {
  %c = hw.constant 1 : i4 {name = "const_val"}
  %0 = comb.add %a, %c : i4
  hw.output %0 : i4
}

// Test that large modules with state are inlined by default
// CHECK-LABEL: hw.module @DontInlineLargeWithState
hw.module @DontInlineLargeWithState(in %clk: !seq.clock, in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: %[[V1:.+]] = comb.mul %[[V0]], %x
  // CHECK-NEXT: %[[V2:.+]] = comb.xor %[[V1]], %x
  // CHECK-NEXT: %[[V3:.+]] = comb.or %[[V2]], %x
  // CHECK-NEXT: %[[V4:.+]] = comb.and %[[V3]], %x
  // CHECK-NEXT: %[[V5:.+]] = comb.add %[[V4]], %x
  // CHECK-NEXT: %[[V6:.+]] = comb.mul %[[V5]], %x
  // CHECK-NEXT: %[[V7:.+]] = comb.xor %[[V6]], %x
  // CHECK-NEXT: %[[REG:.+]] = seq.firreg %[[V7]] clock %clk
  // CHECK-NEXT: hw.output %[[REG]]
  %0 = hw.instance "large" @LargeModuleWithState(clk: %clk: !seq.clock, a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @LargeModuleWithState
hw.module private @LargeModuleWithState(in %clk: !seq.clock, in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = comb.mul %0, %a : i4
  %2 = comb.xor %1, %a : i4
  %3 = comb.or %2, %a : i4
  %4 = comb.and %3, %a : i4
  %5 = comb.add %4, %a : i4
  %6 = comb.mul %5, %a : i4
  %7 = comb.xor %6, %a : i4
  %reg = seq.firreg %7 clock %clk : i4
  hw.output %reg : i4
}

// Test inlining with nested instances
// CHECK-LABEL: hw.module @InlineNested
hw.module @InlineNested(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: %[[V1:.+]] = comb.mul %[[V0]], %[[V0]]
  // CHECK-NEXT: hw.output %[[V1]]
  %0 = hw.instance "outer" @OuterModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @OuterModule
hw.module private @OuterModule(in %a: i4, out b: i4) {
  %0 = hw.instance "inner" @InnerModule(x: %a: i4) -> (y: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @InnerModule
hw.module private @InnerModule(in %x: i4, out y: i4) {
  %0 = comb.add %x, %x : i4
  %1 = comb.mul %0, %0 : i4
  hw.output %1 : i4
}

// Test multiple outputs
// CHECK-LABEL: hw.module @InlineMultiOutput
hw.module @InlineMultiOutput(in %x: i4, out y: i4, out z: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: %[[V1:.+]] = comb.mul %x, %x
  // CHECK-NEXT: hw.output %[[V0]], %[[V1]]
  %0, %1 = hw.instance "multi_out" @MultiOutputModule(a: %x: i4) -> (b: i4, c: i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NOT: hw.module private @MultiOutputModule
hw.module private @MultiOutputModule(in %a: i4, out b: i4, out c: i4) {
  %0 = comb.add %a, %a : i4
  %1 = comb.mul %a, %a : i4
  hw.output %0, %1 : i4, i4
}

// Test inner symbol renaming with multiple instances
// CHECK-LABEL: hw.module @InlineMultiInstanceInnerSym
hw.module @InlineMultiInstanceInnerSym(in %x: i4, in %y: i4, out a: i4, out b: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-DAG: %{{.+}} = hw.wire %{{.+}} sym @wire_sym{{(_[0-9]+)?}}
  // CHECK-DAG: %{{.+}} = hw.wire %{{.+}} sym @wire_sym{{(_[0-9]+)?}}
  // CHECK: hw.output
  %0 = hw.instance "inst1" @ModuleWithSym(a: %x: i4) -> (b: i4)
  %1 = hw.instance "inst2" @ModuleWithSym(a: %y: i4) -> (b: i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NOT: hw.module private @ModuleWithSym
hw.module private @ModuleWithSym(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = hw.wire %0 sym @wire_sym : i4
  hw.output %1 : i4
}

// Test inner symbol renaming with multiple symbols in same module
// CHECK-LABEL: hw.module @InlineMultipleInnerSyms
hw.module @InlineMultipleInnerSyms(in %x: i4, in %y: i4, out a: i4, out b: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-DAG: %{{.+}} = hw.wire %{{.+}} sym @wire1{{(_[0-9]+)?}}
  // CHECK-DAG: %{{.+}} = hw.wire %{{.+}} sym @wire2{{(_[0-9]+)?}}
  // CHECK-DAG: %{{.+}} = hw.wire %{{.+}} sym @wire1{{(_[0-9]+)?}}
  // CHECK-DAG: %{{.+}} = hw.wire %{{.+}} sym @wire2{{(_[0-9]+)?}}
  // CHECK: hw.output
  %0 = hw.instance "inst1" @ModuleWithMultipleSyms(a: %x: i4) -> (b: i4)
  %1 = hw.instance "inst2" @ModuleWithMultipleSyms(a: %y: i4) -> (b: i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NOT: hw.module private @ModuleWithMultipleSyms
hw.module private @ModuleWithMultipleSyms(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = hw.wire %0 sym @wire1 : i4
  %2 = comb.mul %1, %a : i4
  %3 = hw.wire %2 sym @wire2 : i4
  hw.output %3 : i4
}

// Test InnerRefAttr updating with single instance
// CHECK-LABEL: hw.module @InlineWithInnerRef
hw.module @InlineWithInnerRef(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: %[[W0:.+]] = hw.wire %[[V0]] sym @[[SYM:wire_op]]
  // CHECK-NEXT: sv.verbatim "ref to {{[{][{]}}0{{[}][}]}}" {symbols = [#hw.innerNameRef<@InlineWithInnerRef::@[[SYM]]>]}
  // CHECK-NEXT: hw.output %[[W0]]
  %0 = hw.instance "myinst" @ChildWithRef(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @ChildWithRef
hw.module private @ChildWithRef(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = hw.wire %0 sym @wire_op : i4
  sv.verbatim "ref to {{0}}" {symbols = [#hw.innerNameRef<@ChildWithRef::@wire_op>]}
  hw.output %1 : i4
}

// Test InnerRefAttr updating with multiple instances
// CHECK-LABEL: hw.module @InlineMultiInstanceInnerRef
hw.module @InlineMultiInstanceInnerRef(in %x: i4, in %y: i4, out a: i4, out b: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-DAG: %{{.+}} = hw.wire %{{.+}} sym @[[SYM1:wire_op(_[0-9]+)?]]
  // CHECK-DAG: sv.verbatim "ref to {{[{][{]}}0{{[}][}]}}" {symbols = [#hw.innerNameRef<@InlineMultiInstanceInnerRef::@[[SYM1]]>]}
  // CHECK-DAG: %{{.+}} = hw.wire %{{.+}} sym @[[SYM2:wire_op(_[0-9]+)?]]
  // CHECK-DAG: sv.verbatim "ref to {{[{][{]}}0{{[}][}]}}" {symbols = [#hw.innerNameRef<@InlineMultiInstanceInnerRef::@[[SYM2]]>]}
  // CHECK: hw.output
  %0 = hw.instance "inst1" @ChildWithRefMulti(a: %x: i4) -> (b: i4)
  %1 = hw.instance "inst2" @ChildWithRefMulti(a: %y: i4) -> (b: i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NOT: hw.module private @ChildWithRefMulti
hw.module private @ChildWithRefMulti(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = hw.wire %0 sym @wire_op : i4
  sv.verbatim "ref to {{0}}" {symbols = [#hw.innerNameRef<@ChildWithRefMulti::@wire_op>]}
  hw.output %1 : i4
}

// Test HierPath updating when inlining
// CHECK: hw.hierpath private @HierPathTest [@InlineWithHierPath::@[[WIRE_SYM:wire_sym]]]
hw.hierpath private @HierPathTest [@InlineWithHierPath::@inst, @ChildWithHierPath::@wire_sym]

// CHECK-LABEL: hw.module @InlineWithHierPath
hw.module @InlineWithHierPath(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: %[[W0:.+]] = hw.wire %[[V0]] sym @[[WIRE_SYM]]
  // CHECK-NEXT: hw.output %[[W0]]
  %0 = hw.instance "inst" sym @inst @ChildWithHierPath(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @ChildWithHierPath
hw.module private @ChildWithHierPath(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = hw.wire %0 sym @wire_sym : i4
  hw.output %1 : i4
}

// Test that modules targeted by module NLAs are not inlined
// CHECK: hw.hierpath private @ModuleNLA [@DontInlineModuleNLA]
hw.hierpath private @ModuleNLA [@DontInlineModuleNLA]

// CHECK-LABEL: hw.module @DontInlineModuleNLAParent
hw.module @DontInlineModuleNLAParent(in %x: i4, out y: i4) {
  // CHECK-NEXT: hw.instance "inst" @DontInlineModuleNLA
  %0 = hw.instance "inst" @DontInlineModuleNLA(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK: hw.module private @DontInlineModuleNLA
hw.module private @DontInlineModuleNLA(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  hw.output %0 : i4
}

