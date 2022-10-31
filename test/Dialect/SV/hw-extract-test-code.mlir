// RUN:  circt-opt --sv-extract-test-code --split-input-file %s | FileCheck %s
// CHECK-LABEL: module attributes {firrtl.extract.assert = #hw.output_file<"dir3{{/|\\\\}}"
// CHECK-NEXT: hw.module.extern @foo_cover
// CHECK-NOT: attributes
// CHECK-NEXT: hw.module.extern @foo_assume
// CHECK-NOT: attributes
// CHECK-NEXT: hw.module.extern @foo_assert
// CHECK-NOT: attributes
// CHECK: hw.module @issue1246_assert(%clock: i1) attributes {comment = "VCS coverage exclude_file", output_file = #hw.output_file<"dir3{{/|\\\\}}", excludeFromFileList, includeReplicatedOps>}
// CHECK: sv.assert
// CHECK: sv.error "Assertion failed"
// CHECK: sv.error "assert:"
// CHECK: sv.error "assertNotX:"
// CHECK: sv.error "check [verif-library-assert] is included"
// CHECK: sv.fatal 1
// CHECK: foo_assert
// CHECK: hw.module @issue1246_assume(%clock: i1)
// CHECK-SAME: attributes {comment = "VCS coverage exclude_file"}
// CHECK: sv.assume
// CHECK: foo_assume
// CHECK: hw.module @issue1246_cover(%clock: i1)
// CHECK-SAME: attributes {comment = "VCS coverage exclude_file"}
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
  hw.module.extern @foo_cover(%a : i1) attributes {"firrtl.extract.cover.extra"}
  hw.module.extern @foo_assume(%a : i1) attributes {"firrtl.extract.assume.extra"}
  hw.module.extern @foo_assert(%a : i1) attributes {"firrtl.extract.assert.extra"}
  hw.module @issue1246(%clock: i1) -> () {
    sv.always posedge %clock  {
      sv.ifdef.procedural "SYNTHESIS"  {
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
  hw.module @AlreadyExtracted(%clock: i1) -> () {
    sv.always posedge %clock  {
      sv.assert %clock, immediate
    }
  }
  hw.module @Top(%clock: i1) -> () {
    hw.instance "submodule" @AlreadyExtracted(clock: %clock: i1) -> () {doNotPrint = true}
  }
}

// -----

// Check that we don't extract assertions from a module with "firrtl.extract.do_not_extract" attribute.
//
// CHECK-NOT:  hw.module @ModuleInTestHarness_assert
// CHECK-NOT:  firrtl.extract.do_not_extract
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>} {
  hw.module @ModuleInTestHarness(%clock: i1) -> () attributes {"firrtl.extract.do_not_extract"} {
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
  hw.module @InstanceName(%clock: i1, %cond: i1, %cond2: i1) -> () {
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
  hw.module @MultiRead(%clock: i1, %cond: i1) -> () {
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
// CHECK: %0 = comb.and %1
// CHECK: %1 = comb.and %0
// CHECK: hw.instance "{{[^ ]+}}" {{.+}} @InputOnlyCycle_cover
// CHECK-NOT: sv.bind <@InputOnly::
// CHECK-DAG: sv.bind <@Top::@[[input_only_assert]]>
// CHECK-DAG: sv.bind <@Top::@[[input_only_cover]]>
// CHECK-DAG: sv.bind <@InputOnlySym::@[[input_only_sym_cover]]>
module {
  hw.module private @InputOnly(%clock: i1, %cond: i1) -> () {
    sv.always posedge %clock  {
      sv.cover %cond, immediate
      sv.assert %cond, immediate
    }
  }

  hw.module private @InputOnlySym(%clock: i1, %cond: i1) -> () {
    sv.always posedge %clock  {
      sv.cover %cond, immediate
    }
  }

  hw.module private @InputOnlyCycle(%clock: i1, %cond: i1) -> () {
    // Arbitrary code that won't be extracted, should be inlined, and has a cycle.
    %0 = comb.and %1 : i1
    %1 = comb.and %0 : i1

    sv.always posedge %clock  {
      sv.cover %cond, immediate
    }
  }

  hw.module @Top(%clock: i1, %cond: i1) -> (foo: i1) {
    hw.instance "input_only" @InputOnly(clock: %clock: i1, cond: %cond: i1) -> ()
    hw.instance "input_only_sym" sym @foo @InputOnlySym(clock: %clock: i1, cond: %cond: i1) -> ()
    hw.instance "input_only_cycle" @InputOnlyCycle(clock: %clock: i1, cond: %cond: i1) -> ()
    hw.output %cond : i1
  }
}

// -----
// Check instance extraction

// All instances of Baz are extracted, so it should be output to the testbench.
// CHECK-LABEL: @Baz
// CHECK-SAME: output_file = #hw.output_file<"testbench{{/|\\\\}}", excludeFromFileList, includeReplicatedOps>

// All instances of Bozo are extracted, so it should be output to the testbench.
// CHECK-LABEL: @Bozo
// CHECK: #hw.output_file<"testbench

// In AllExtracted, instances foo, bar, and baz should be extracted.
// CHECK-LABEL: @AllExtracted_cover
// CHECK: hw.instance "foo"
// CHECK: hw.instance "bar"
// CHECK: hw.instance "baz"

// In SomeExtracted, only instance baz should be extracted.
// CHECK-LABEL: @SomeExtracted_cover
// CHECK-NOT: hw.instance "foo"
// CHECK-NOT: hw.instance "bar"
// CHECK: hw.instance "baz"

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
// CHECK-SAME: (%[[clock:.+]]: i1, %[[in:.+]]: i1)
// CHECK: hw.instance {{.+}} @MultiResultExtracted_cover([[clock]]: %[[clock]]: i1, [[in]]: %[[in]]: i1)

// In SymNotExtracted, instance foo should not be extracted because it has a sym.
// CHECK-LABEL: @SymNotExtracted_cover
// CHECK-NOT: hw.instance "foo"
// CHECK-LABEL: @SymNotExtracted
// CHECK: hw.instance "foo"

module attributes {
  firrtl.extract.testbench = #hw.output_file<"testbench/", excludeFromFileList, includeReplicatedOps>
} {
  hw.module private @Foo(%a: i1) -> (b: i1) {
    hw.output %a : i1
  }

  hw.module.extern private @Bar(%a: i1) -> (b: i1)

  hw.module.extern private @Baz(%a: i1) -> (b: i1)

  hw.module.extern private @Qux(%a: i1) -> (b: i1, c: i1)

  hw.module.extern private @Bozo(%a: i1) -> (b: i1)

  hw.module @AllExtracted(%clock: i1, %in: i1) {
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

  hw.module @SomeExtracted(%clock: i1, %in: i1) -> (out0: i1, out1: i1) {
    %foo.b = hw.instance "foo" @Foo(a: %in: i1) -> (b: i1)
    %bar.b = hw.instance "bar" @Bar(a: %in: i1) -> (b: i1)
    %baz.b = hw.instance "baz" @Baz(a: %in: i1) -> (b: i1)
    sv.always posedge %clock {
      sv.cover %foo.b, immediate
      sv.cover %bar.b, immediate
      sv.cover %baz.b, immediate
    }
    hw.output %foo.b, %bar.b : i1, i1
  }

  hw.module @CycleExtracted(%clock: i1, %in: i1) {
    %foo.b = hw.instance "foo" @Foo(a: %in: i1) -> (b: i1)
    %0 = comb.or %0, %foo.b : i1
    sv.always posedge %clock {
      sv.cover %0, immediate
    }
  }

  hw.module private @ShouldBeInlined(%clock: i1, %in: i1) {
    %foo.b = hw.instance "foo" @Foo(a: %in: i1) -> (b: i1)
    sv.always posedge %clock {
      sv.cover %foo.b, immediate
    }
  }

  hw.module private @ShouldBeInlined2(%clock: i1, %in: i1) {
    %bozo.b = hw.instance "bozo" @Bozo(a: %in: i1) -> (b: i1)
    sv.ifdef "SYNTHESIS" {
    } else {
      sv.always posedge %clock {
        sv.if %bozo.b {
          sv.cover %bozo.b, immediate
        }
      }
    }
  }

  hw.module @ChildShouldInline(%clock: i1, %in: i1) {
    hw.instance "child" @ShouldBeInlined(clock: %clock: i1, in: %in: i1) -> ()
  }

  hw.module @ChildShouldInline2(%clock: i1, %in: i1) {
    hw.instance "child" @ShouldBeInlined2(clock: %clock: i1, in: %in: i1) -> ()
  }

  hw.module @MultiResultExtracted(%clock: i1, %in: i1) {
    %qux.b, %qux.c = hw.instance "qux" @Qux(a: %in: i1) -> (b: i1, c: i1)
    sv.always posedge %clock {
      sv.cover %qux.b, immediate
      sv.cover %qux.c, immediate
    }
  }

  hw.module @SymNotExtracted(%clock: i1, %in: i1) {
    %foo.b = hw.instance "foo" sym @foo @Foo(a: %in: i1) -> (b: i1)
    sv.always posedge %clock {
      sv.cover %foo.b, immediate
    }
  }
}
