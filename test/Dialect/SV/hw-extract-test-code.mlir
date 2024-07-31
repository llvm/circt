// RUN:  circt-opt --sv-extract-test-code --split-input-file %s | FileCheck %s
// CHECK-LABEL: module attributes {firrtl.extract.assert = #hw.output_file<"dir3{{/|\\\\}}"
// CHECK-NEXT: sv.macro.decl @SYNTHESIS
// CHECK-NEXT: emit.fragment @some_fragment {
// CHECK-NEXT:   sv.verbatim "foo"
// CHECK-NEXT: }
// CHECK-NEXT: hw.module.extern @foo_cover
// CHECK-NOT: attributes
// CHECK-NEXT: hw.module.extern @foo_assume
// CHECK-NOT: attributes
// CHECK-NEXT: hw.module.extern @foo_assert
// CHECK-NOT: attributes
// CHECK: hw.module private @issue1246_assert(in %clock : i1) attributes {comment = "VCS coverage exclude_file", emit.fragments = [@some_fragment], output_file = #hw.output_file<"dir3{{/|\\\\}}", excludeFromFileList, includeReplicatedOps>}
// CHECK: sv.assert
// CHECK: sv.error "Assertion failed"
// CHECK: sv.error "assert:"
// CHECK: sv.error "assertNotX:"
// CHECK: sv.error "check [verif-library-assert] is included"
// CHECK: sv.fatal 1
// CHECK: foo_assert
// CHECK: hw.module private @issue1246_assume(in %clock : i1) attributes {
// CHECK-SAME: comment = "VCS coverage exclude_file"
// CHECK-SAME: emit.fragments = [@some_fragment]
// CHECK-SAME: }
// CHECK: sv.assume
// CHECK: foo_assume
// CHECK: hw.module private @issue1246_cover(in %clock : i1) attributes {
// CHECK-SAME: comment = "VCS coverage exclude_file"
// CHECK-SAME: emit.fragments = [@some_fragment]
// CHECK-SAME: }
// CHECK: sv.cover
// CHECK: foo_cover
// CHECK: hw.module @issue1246
// CHECK-NOT: sv.assert
// CHECK-NOT: sv.assume
// CHECK-NOT: sv.cover
// CHECK-NOT: foo_assert
// CHECK-NOT: foo_assume
// CHECK-NOT: foo_cover
// CHECK: sv.bind <@issue1246::@__ETC_issue1246_assert>
// CHECK: sv.bind <@issue1246::@__ETC_issue1246_assume> {output_file = #hw.output_file<"file4", excludeFromFileList>}
// CHECK: sv.bind <@issue1246::@__ETC_issue1246_cover>
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>, firrtl.extract.assume.bindfile = #hw.output_file<"file4", excludeFromFileList>} {
  sv.macro.decl @SYNTHESIS
  emit.fragment @some_fragment {
    sv.verbatim "foo"
  }
  hw.module.extern @foo_cover(in %a : i1) attributes {"firrtl.extract.cover.extra"}
  hw.module.extern @foo_assume(in %a : i1) attributes {"firrtl.extract.assume.extra"}
  hw.module.extern @foo_assert(in %a : i1) attributes {"firrtl.extract.assert.extra"}
  hw.module @issue1246(in %clock: i1) attributes {emit.fragments = [@some_fragment]} {
    sv.always posedge %clock  {
      sv.ifdef.procedural @SYNTHESIS {
      } else  {
        sv.if %2937  {
          sv.assert %clock, immediate
          sv.error "Assertion failed"
          sv.error "assert:"
          sv.error "assertNotX:"
          sv.error "check [verif-library-assert] is included"
          sv.fatal 1
          sv.assume %clock, immediate
          sv.cover %clock, immediate
        }
      }
    }
    %2937 = hw.constant 0 : i1
    hw.instance "bar_cover" @foo_cover(a: %clock : i1) -> ()
    hw.instance "bar_assume" @foo_assume(a: %clock : i1) -> ()
    hw.instance "bar_assert" @foo_assert(a: %clock : i1) -> ()
    hw.output
  }
}

// -----

// Check that a module that is already going to be extracted does not have its
// asserts also extracted.  This avoids a problem where certain simulators do
// not like to bind instances into bound instances.  See:
//   - https://github.com/llvm/circt/issues/2910
//
// CHECK-LABEL: @AlreadyExtracted
// CHECK-COUNT-1: doNotPrint
// CHECK-NOT:     doNotPrint
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>} {
  hw.module @AlreadyExtracted(in %clock: i1) {
    sv.always posedge %clock  {
      sv.assert %clock, immediate
    }
  }
  hw.module @Top(in %clock: i1) {
    hw.instance "submodule" @AlreadyExtracted(clock: %clock: i1) -> () {doNotPrint = true}
  }
}

// -----

// Check that we don't extract assertions from a module with "firrtl.extract.do_not_extract" attribute.
//
// CHECK-NOT:  hw.module private @ModuleInTestHarness_assert
// CHECK-NOT:  firrtl.extract.do_not_extract
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>} {
  hw.module @ModuleInTestHarness(in %clock: i1) attributes {"firrtl.extract.do_not_extract"} {
    sv.always posedge %clock  {
      sv.assert %clock, immediate
    }
  }
}

// -----
// Check extracted modules and their instantiations use same name

// CHECK-LABEL: @InstanceName(
// CHECK:      hw.instance "[[name:.+]]_assert" sym @{{[^ ]+}} @[[name]]_assert
// CHECK-NEXT: hw.instance "[[name:.+]]_assume" sym @{{[^ ]+}} @[[name]]_assume
// CHECK-NEXT: hw.instance "[[name:.+]]_cover"  sym @{{[^ ]+}} @[[name]]_cover
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>} {
  hw.module @InstanceName(in %clock: i1, in %cond: i1, in %cond2: i1) {
    sv.always posedge %clock  {
      sv.assert %cond, immediate
      sv.assume %cond, immediate
      sv.cover %cond, immediate
    }
  }
}


// -----
// Check wires are extracted once

// CHECK-LABEL: @MultiRead(
// CHECK: hw.instance "[[name:.+]]_cover"  sym @{{[^ ]+}} @[[name]]_cover(foo: %0: i1, clock: %clock: i1)
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>} {
  hw.module @MultiRead(in %clock: i1, in %cond: i1) {
    %foo = sv.wire : !hw.inout<i1>
    sv.assign %foo, %cond : i1
    %cond1 = sv.read_inout %foo : !hw.inout<i1>
    %cond2 = sv.read_inout %foo : !hw.inout<i1>
    %cond3 = sv.read_inout %foo : !hw.inout<i1>
    sv.always posedge %clock  {
      sv.cover %cond1, immediate
      sv.cover %cond2, immediate
      sv.cover %cond3, immediate
    }
  }
}

// -----
// Check extracted module ports take name of instance result when needed.

// CHECK-LABEL: @InstResult(
// CHECK: hw.instance "[[name:.+]]_cover"  sym @{{[^ ]+}} @[[name]]_cover(mem.result_name: %{{[^ ]+}}: i1, mem.1: %{{[^ ]+}}: i1, clock: %clock: i1)
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>} {
  hw.module @Mem(out result_name: i1, out "": i1) {
    %reg = sv.reg : !hw.inout<i1>
    %0 = sv.read_inout %reg : !hw.inout<i1>
    hw.output %0, %0 : i1, i1
  }
  // Dummy is needed to prevent the instance itself being extracted
  hw.module @Dummy(in %in1: i1, in %in2: i1) {}
  hw.module @InstResult(in %clock: i1) {
    %0, %1 = hw.instance "mem" @Mem() -> (result_name: i1, "": i1)
    hw.instance "dummy" sym @keep @Dummy(in1: %0 : i1, in2: %1 : i1) -> ()
    %2 = comb.and bin %0, %1 : i1
    sv.always posedge %clock  {
      sv.cover %2, immediate
    }
  }
}

// -----
// Check "empty" modules are inlined

// CHECK-NOT: @InputOnly(
// CHECK-DAG: @InputOnly_assert(
// CHECK-DAG: @InputOnly_cover(
// CHECK-DAG: @InputOnlySym_cover(
// CHECK-LABEL: @InputOnlySym(
// CHECK: hw.instance "{{[^ ]+}}" sym @[[input_only_sym_cover:[^ ]+]] @InputOnlySym_cover
// CHECK-LABEL: @Top
// CHECK-NOT: hw.instance {{.+}} @Top{{.*}}
// CHECK: hw.instance "{{[^ ]+}}" sym @[[input_only_assert:[^ ]+]] @InputOnly_assert
// CHECK: hw.instance "{{[^ ]+}}" sym @[[input_only_cover:[^ ]+]] @InputOnly_cover
// CHECK: hw.instance "{{[^ ]+}}" {{.+}} @InputOnlySym
// CHECK-NOT: %0 = comb.and %1
// CHECK-NOT: %1 = comb.and %0
// CHECK: hw.instance "{{[^ ]+}}" {{.+}} @InputOnlyCycle_cover
// CHECK: hw.instance {{.*}} sym @[[already_bound:[^ ]+]] @AlreadyBound
// CHECK-NOT: sv.bind <@InputOnly::
// CHECK-DAG: sv.bind <@Top::@[[input_only_assert]]>
// CHECK-DAG: sv.bind <@Top::@[[input_only_cover]]>
// CHECK-DAG: sv.bind <@InputOnlySym::@[[input_only_sym_cover]]>
// CHECK-DAG: sv.bind <@Top::@[[already_bound]]>
module {
  hw.module private @AlreadyBound() {}

  hw.module private @InputOnly(in %clock: i1, in %cond: i1) {
    sv.always posedge %clock  {
      sv.cover %cond, immediate
      sv.assert %cond, immediate
    }
  }

  hw.module private @InputOnlySym(in %clock: i1, in %cond: i1) {
    sv.always posedge %clock  {
      sv.cover %cond, immediate
    }
  }

  hw.module private @InputOnlyCycle(in %clock: i1, in %cond: i1) {
    // Arbitrary code that won't be extracted, should be dead in the input only module.
    // Make sure to delete them.
    %0 = comb.and %1 : i1
    %1 = comb.and %0 : i1

    sv.always posedge %clock  {
      sv.cover %cond, immediate
    }
  }

  hw.module private @InputOnlyBind(in %clock: i1, in %cond: i1) {
    hw.instance "already_bound" sym @already_bound @AlreadyBound() -> () {doNotPrint = true}
    sv.always posedge %clock  {
      sv.cover %cond, immediate
      sv.assert %cond, immediate
    }
  }

  hw.module @Top(in %clock: i1, in %cond: i1, out foo: i1) {
    hw.instance "input_only" @InputOnly(clock: %clock: i1, cond: %cond: i1) -> ()
    hw.instance "input_only_sym" sym @foo @InputOnlySym(clock: %clock: i1, cond: %cond: i1) -> ()
    hw.instance "input_only_cycle" @InputOnlyCycle(clock: %clock: i1, cond: %cond: i1) -> ()
    hw.instance "input_only_bind" @InputOnlyBind(clock: %clock: i1, cond: %cond: i1) -> ()
    hw.output %cond : i1
  }

  sv.bind <@InputOnlyBind::@already_bound>
}

// -----
// Check instance extraction

// In AllExtracted, instances foo, bar, and baz should be extracted.
// CHECK-LABEL: @AllExtracted_cover
// CHECK: hw.instance "foo"
// CHECK: hw.instance "bar"
// CHECK: hw.instance "baz"

// In SomeExtracted, only instance baz should be extracted.
// Check that a dead external module bozo and its operand are still alive.
// CHECK-LABEL: @SomeExtracted_cover
// CHECK-NOT: hw.instance "foo"
// CHECK-NOT: hw.instance "bar"
// CHECK-NOT: hw.instance "bozo"
// CHECK: hw.instance "baz"
// CHECK-LABEL: @SomeExtracted
// CHECK: comb.and
// CHECK: hw.instance "bozo"

// In CycleExtracted, instance foo should be extracted despite combinational cycle.
// CHECK-LABEL: @CycleExtracted_cover
// CHECK: hw.instance "foo"

// In ChildShouldInline, instance child should be inlined while it's instance foo is still extracted.
// CHECK-NOT: hw.module @ShouldBeInlined(
// CHECK-LABEL: @ShouldBeInlined_cover
// CHECK: hw.instance "foo"
// CHECK-LABEL: @ChildShouldInline
// CHECK-NOT: hw.instance "child"
// CHECK: hw.instance {{.+}} @ShouldBeInlined_cover

// In ChildShouldInline2, instance bozo should not be inlined, since it was also extracted.
// CHECK-LABEL: hw.module @ChildShouldInline2
// CHECK-NOT: hw.instance "bozo"

// In MultiResultExtracted, instance qux should be extracted without leaving null operands to the extracted instance
// CHECK-LABEL: @MultiResultExtracted_cover
// CHECK: hw.instance "qux"
// CHECK-LABEL: @MultiResultExtracted
// CHECK-SAME: (in %[[clock:.+]] : i1, in %[[in:.+]] : i1)
// CHECK: hw.instance {{.+}} @MultiResultExtracted_cover([[in]]: %[[in]]: i1, [[clock]]: %[[clock]]: i1)

// In SymNotExtracted, instance foo should not be extracted because it has a sym.
// CHECK-LABEL: @SymNotExtracted_cover
// CHECK-NOT: hw.instance "foo"
// CHECK-LABEL: @SymNotExtracted
// CHECK: hw.instance "foo"

// In NoExtraInput, instance foo should be extracted, and no extra input should be added for %0
// CHECK-LABEL: @NoExtraInput_cover
// CHECK: %[[or0:.+]] = comb.or
// CHECK: hw.instance "foo" @Foo(a: %[[or0]]: i1)
// CHECK-LABEL: @NoExtraInput
// CHECK-NOT: %{{.+}} = comb.or

// In InstancesWithCycles, the only_testcode instances should be extracted, but the non_testcode instances should not
// CHECK-LABEL: @InstancesWithCycles_cover
// CHECK: hw.instance "only_testcode_and_instance0"
// CHECK: hw.instance "only_testcode_and_instance1"
// CHECK-LABEL: @InstancesWithCycles
// CHECK-NOT: hw.instance "only_testcode_and_instance0"
// CHECK-NOT: hw.instance "only_testcode_and_instance1"
// CHECK: hw.instance "non_testcode_and_instance0"
// CHECK: hw.instance "non_testcode_and_instance1"

module {
  sv.macro.decl @SYNTHESIS

  hw.module private @Foo(in %a: i1, out b: i1) {
    hw.output %a : i1
  }

  hw.module.extern private @Bar(in %a: i1, out b: i1)

  hw.module.extern private @Baz(in %a: i1, out b: i1)

  hw.module.extern private @Qux(in %a: i1, out b: i1, out c: i1)

  hw.module.extern private @Bozo(in %a: i1, out b: i1)

  hw.module @AllExtracted(in %clock: i1, in %in: i1) {
    %foo.b = hw.instance "foo" @Foo(a: %in: i1) -> (b: i1)
    %bar.b = hw.instance "bar" @Bar(a: %in: i1) -> (b: i1)
    %baz.b = hw.instance "baz" @Baz(a: %in: i1) -> (b: i1)
    sv.always posedge %clock {
      sv.if %foo.b {
        sv.if %bar.b {
          sv.cover %foo.b, immediate
        }
      }
      sv.cover %bar.b, immediate
      sv.cover %baz.b, immediate
    }
  }

  hw.module @SomeExtracted(in %clock: i1, in %in: i1, out out0: i1, out out1: i1) {
    %foo.b = hw.instance "foo" @Foo(a: %in: i1) -> (b: i1)
    %bar.b = hw.instance "bar" @Bar(a: %in: i1) -> (b: i1)
    %baz.b = hw.instance "baz" @Baz(a: %in: i1) -> (b: i1)
    %and = comb.and %in, %clock: i1
    %bozo = hw.instance "bozo" @Bozo(a: %and: i1) -> (b: i1)
    sv.always posedge %clock {
      sv.cover %foo.b, immediate
      sv.cover %bar.b, immediate
      sv.cover %baz.b, immediate
      sv.cover %and, immediate
    }
    hw.output %foo.b, %bar.b : i1, i1
  }

  hw.module @CycleExtracted(in %clock: i1, in %in: i1) {
    %foo.b = hw.instance "foo" @Foo(a: %in: i1) -> (b: i1)
    %0 = comb.or %0, %foo.b : i1
    sv.always posedge %clock {
      sv.cover %0, immediate
    }
  }

  hw.module private @ShouldBeInlined(in %clock: i1, in %in: i1) {
    %foo.b = hw.instance "foo" @Foo(a: %in: i1) -> (b: i1)
    sv.always posedge %clock {
      sv.cover %foo.b, immediate
    }
  }

  hw.module private @ShouldBeInlined2(in %clock: i1, in %in: i1) {
    %bozo.b = hw.instance "bozo" @Bozo(a: %in: i1) -> (b: i1)
    sv.ifdef @SYNTHESIS {
    } else {
      sv.always posedge %clock {
        sv.if %bozo.b {
          sv.cover %bozo.b, immediate
        }
      }
    }
  }

  hw.module @ChildShouldInline(in %clock: i1, in %in: i1) {
    hw.instance "child" @ShouldBeInlined(clock: %clock: i1, in: %in: i1) -> ()
  }

  hw.module @ChildShouldInline2(in %clock: i1, in %in: i1) {
    hw.instance "child" @ShouldBeInlined2(clock: %clock: i1, in: %in: i1) -> ()
  }

  hw.module @MultiResultExtracted(in %clock: i1, in %in: i1) {
    %qux.b, %qux.c = hw.instance "qux" @Qux(a: %in: i1) -> (b: i1, c: i1)
    sv.always posedge %clock {
      sv.cover %qux.b, immediate
      sv.cover %qux.c, immediate
    }
  }

  hw.module @SymNotExtracted(in %clock: i1, in %in: i1) {
    %foo.b = hw.instance "foo" sym @foo @Foo(a: %in: i1) -> (b: i1)
    sv.always posedge %clock {
      sv.cover %foo.b, immediate
    }
  }

  hw.module @NoExtraInput(in %clock: i1, in %in: i1) {
    %0 = comb.or %in, %in : i1
    %foo.b = hw.instance "foo" @Foo(a: %0: i1) -> (b: i1)
    sv.always posedge %clock {
      sv.cover %0, immediate
      sv.cover %foo.b, immediate
    }
  }

  hw.module private @Passthrough(in %in: i1, out out: i1) {
    hw.output %in : i1
  }

  hw.module @InstancesWithCycles(in %clock: i1, in %in: i1, out out: i1) {
    %0 = hw.instance "non_testcode_and_instance0" @Passthrough(in: %1: i1) -> (out: i1)
    %1 = hw.instance "non_testcode_and_instance1" @Passthrough(in: %0: i1) -> (out: i1)

    %2 = hw.instance "only_testcode_and_instance0" @Passthrough(in: %3: i1) -> (out: i1)
    %3 = hw.instance "only_testcode_and_instance1" @Passthrough(in: %2: i1) -> (out: i1)
    %4 = comb.or %2, %3 : i1

    sv.always posedge %clock {
      sv.cover %1, immediate
      sv.cover %2, immediate
      sv.cover %4, immediate
    }

    hw.output %0 : i1
  }
}

// -----
// Check register extraction

module {
  // CHECK-LABEL: @RegExtracted_cover
  // CHECK-SAME: %designAndTestCode
  // CHECK: %testCode1 = seq.firreg
  // CHECK: %testCode2 = seq.firreg
  // CHECK-NOT: seq.firreg

  // CHECK-LABEL: @RegExtracted
  // CHECK: %symbol = seq.firreg
  // CHECK: %designAndTestCode = seq.firreg
  // CHECK-NOT: seq.firreg
  hw.module @RegExtracted(in %clock: !seq.clock, in %reset: i1, in %in: i1, out out: i1) {
    %muxed = comb.mux bin %reset, %in, %testCode1 : i1
    %testCode1 = seq.firreg %muxed clock %clock : i1
    %testCode2 = seq.firreg %testCode1 clock %clock : i1
    %symbol = seq.firreg %in clock %clock sym @foo : i1
    %designAndTestCode = seq.firreg %in clock %clock : i1
    %deadReg = seq.firreg %testCode1 clock %clock : i1

    %clk = seq.from_clock %clock
    sv.always posedge %clk {
      sv.cover %testCode1, immediate
      sv.cover %testCode2, immediate
      sv.cover %designAndTestCode, immediate
    }

    hw.output %designAndTestCode : i1
  }
}

// -----
// Check that constants are cloned freely.

module {
  // CHECK-LABEL: @ConstantCloned_cover(in %in : i1, in %clock : i1)
  // CHECK-NEXT:   %true = hw.constant true
  // CHECK-NEXT:   comb.xor bin %in, %true : i1
  hw.module @ConstantCloned(in %clock: i1, in %in: i1, out out: i1) {
    %true = hw.constant true
    %not = comb.xor bin %in, %true : i1

    sv.always posedge %clock {
      sv.cover %not, immediate
    }

    hw.output %true : i1
  }
}

// -----
// Check that input only modules are inlined properly.

module {
  // @ShouldNotBeInlined cannot be inlined because there is a wire with an inner sym
  // that is referred by hierpath op.
  hw.hierpath private @Foo [@ShouldNotBeInlined::@foo]
  hw.module private @ShouldNotBeInlined(in %clock: i1, in %a: i1) {
    %w = sv.wire sym @foo: !hw.inout<i1>
    sv.always posedge %clock {
      sv.if %a {
        sv.assert %a, immediate message "foo"
      }
    }
    hw.output
  }
  hw.module private @Assert(in %clock: i1, in %a: i1) {
    sv.always posedge %clock {
      sv.if %a {
        sv.assert %a, immediate message "foo"
      }
    }
    hw.output
  }

  // CHECK-LABEL: hw.module private @AssertWrapper(in %clock : i1, in %a : i1, out b : i1) {
  // CHECK-NEXT:  hw.instance "Assert_assert" sym @__ETC_Assert_assert @Assert_assert
  // CHECK-SAME:  doNotPrint = true
  hw.module private @AssertWrapper(in %clock: i1, in %a: i1, out b: i1) {
    hw.instance "a3" @Assert(clock: %clock: i1, a: %a: i1) -> ()
    hw.output %a: i1
  }

  // CHECK-NOT: @InputOnly
  hw.module private @InputOnly(in %clock: i1, in %a: i1) {
    hw.instance "a4" @Assert(clock: %clock: i1, a: %a: i1) -> ()
  }

  // CHECK-LABEL: hw.module @Top(in %clock : i1, in %a : i1, in %b : i1) {
  // CHECK-NEXT:  hw.instance "Assert_assert" sym @__ETC_Assert_assert_0 @Assert_assert
  // CHECK-SAME:  doNotPrint = true
  // CHECK-NEXT:  hw.instance "Assert_assert" sym @__ETC_Assert_assert @Assert_assert
  // CHECK-SAME:  doNotPrint = true
  // CHECK-NEXT:  hw.instance "Assert_assert" sym @__ETC_Assert_assert_1 @Assert_assert
  // CHECK-SAME:  doNotPrint = true
  // CHECK-NEXT:  hw.instance "should_not_be_inlined" @ShouldNotBeInlined
  // CHECK-NOT: doNotPrint
  hw.module @Top(in %clock: i1, in %a: i1, in %b: i1) {
    hw.instance "a1" @Assert(clock: %clock: i1, a: %a: i1) -> ()
    hw.instance "a2" @Assert(clock: %clock: i1, a: %b: i1) -> ()
    hw.instance "a3" @InputOnly(clock: %clock: i1, a: %b: i1) -> ()
    hw.instance "should_not_be_inlined" @ShouldNotBeInlined (clock: %clock: i1, a: %b: i1) -> ()
    hw.output
  }
  // CHECK:       sv.bind <@Top::@__ETC_Assert_assert>
  // CHECK-NEXT:  sv.bind <@Top::@__ETC_Assert_assert_0>
  // CHECK-NEXT:  sv.bind <@Top::@__ETC_Assert_assert_1>
  // CHECK-NEXT:  sv.bind <@AssertWrapper::@__ETC_Assert_assert>
}


// -----

// Check that the port order is [%clock, %in] and not the reverse.  This is
// trying to guarantee taht the generated port order is stable with additions or
// deletions of internal operations.  The critical thing in this test is the
// existence of the `%0 = seq.from_clock %clock` operation.  This operation was
// added and it caused the port order to become [%in, %clock].  This shouldn't
// happen.
//
// See: https://github.com/llvm/circt/issues/6072

module {
  // CHECK-LABEL: hw.module @PortOrder_6072
  hw.module @PortOrder_6072(in %clock : !seq.clock, in %in : i1) {
    hw.instance "dut" @Foo(clock: %clock: !seq.clock, in: %in: i1) -> ()
    hw.output
  }
  // CHECK: hw.module private @Foo_cover(in %clock : !seq.clock, in %in : i1)
  hw.module private @Foo(in %clock: !seq.clock, in %in: i1) {
    %0 = seq.from_clock %clock
    sv.cover.concurrent posedge %0, %in label "cover__hello"
    hw.output
  }
}


// -----

// Check that no anonymous ports are created and all the port names are unique.

module {
  hw.module @PortName(in %clock : !seq.clock, in %in : i1) {
    %x = hw.instance "pF" @PortNameFoo(clock: %clock: !seq.clock, "": %in: i1) -> (o: i1)
    hw.output
  }
  // CHECK-LABEL: hw.module private @PortNameFoo_cover
  // CHECK-SAME: (in %clock : !seq.clock, in %port_1 : i1, in %port_2 : i1)
  hw.module private @PortNameFoo(in %clock: !seq.clock, in %1: i1, out o : i1) {
    // CHECK: hw.instance "PortNameFoo_cover"
    // CHECK-SAME: @PortNameFoo_cover(clock: %clock: !seq.clock, port_1: %0: i1, port_2: %1: i1) -> ()
    %0 = seq.from_clock %clock
    %2 = comb.xor %1, %1 : i1
    sv.cover.concurrent posedge %0, %1 label "cover__hello1"
    sv.cover.concurrent posedge %0, %2 label "cover__hello2"
    hw.output %2 : i1
  }
}

// -----

// Check that verif ops are also extracted.

// CHECK: hw.module private @VerifOps_assert
// CHECK: hw.module private @VerifOps_assume
// CHECK: hw.module private @VerifOps_cover
hw.module @VerifOps(in %a : i1, in %b : i1) {
  %true = hw.constant true
  verif.assert %true : i1
  verif.assume %true : i1
  verif.cover %true : i1
  hw.output
}

// -----

// Check that clocked verif ops are also extracted.

// CHECK: hw.module private @ClockedVerifOps_assert
// CHECK: hw.module private @ClockedVerifOps_assume
// CHECK: hw.module private @ClockedVerifOps_cover
hw.module @ClockedVerifOps(in %a : i1, in %en : i1, in %clock: i1) {
  %true = hw.constant true
  verif.clocked_assert %a if %en, posedge %clock : i1
  verif.clocked_assume %a if %en, posedge %clock : i1
  verif.clocked_cover %a if %en, posedge %clock : i1
  hw.output
}
