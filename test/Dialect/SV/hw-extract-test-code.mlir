// RUN:  circt-opt --sv-extract-test-code --split-input-file %s | FileCheck %s
// CHECK-LABEL: module attributes {firrtl.extract.assert = #hw.output_file<"dir3/"
// CHECK-NEXT: hw.module.extern @foo_cover
// CHECK-NOT: attributes
// CHECK-NEXT: hw.module.extern @foo_assume
// CHECK-NOT: attributes
// CHECK-NEXT: hw.module.extern @foo_assert
// CHECK-NOT: attributes
// CHECK: hw.module @issue1246_assert(%clock: i1) attributes {comment = "VCS coverage exclude_file", output_file = #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>}
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
