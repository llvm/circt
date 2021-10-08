// RUN:  circt-opt --sv-extract-test-code %s | FileCheck %s
// CHECK-LABEL: module attributes {firrtl.extract.assert = #hw.output_file<"dir3/"
// CHECK-NEXT: hw.module.extern @foo_cover
// CHECK-NOT: attributes
// CHECK-NEXT: hw.module.extern @foo_assume
// CHECK-NOT: attributes
// CHECK-NEXT: hw.module.extern @foo_assert
// CHECK-NOT: attributes
// CHECK: hw.module @issue1246_assert(%clock: i1) attributes {output_file = #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>}
// CHECK: sv.assert
// CHECK: foo_assert
// CHECK: hw.module @issue1246_assume(%clock: i1) 
// CHECK-NOT: attributes 
// CHECK: sv.assume
// CHECK: foo_assume
// CHECK: hw.module @issue1246_cover(%clock: i1) 
// CHECK-NOT: attributes 
// CHECK: sv.cover
// CHECK: foo_cover
// CHECK: hw.module @issue1246
// CHECK-NOT: sv.assert
// CHECK-NOT: sv.assume
// CHECK-NOT: sv.cover
// CHECK-NOT: foo_assert
// CHECK-NOT: foo_assume
// CHECK-NOT: foo_cover
// CHECK: sv.bind @__ETC_issue1246_assert in @issue1246
// CHECK: sv.bind @__ETC_issue1246_assume in @issue1246 {output_file = #hw.output_file<"file4", excludeFromFileList>}
// CHECK: sv.bind @__ETC_issue1246_cover in @issue1246
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>, firrtl.extract.assume.bindfile = #hw.output_file<"file4", excludeFromFileList>} {
  hw.module.extern @foo_cover(%a : i1) attributes {"firrtl.extract.cover.extra"} 
  hw.module.extern @foo_assume(%a : i1) attributes {"firrtl.extract.assume.extra"} 
  hw.module.extern @foo_assert(%a : i1) attributes {"firrtl.extract.assert.extra"} 
  hw.module @issue1246(%clock: i1) -> () {
    sv.always posedge %clock  {
      sv.ifdef.procedural "SYNTHESIS"  {
      } else  {
        sv.if %2937  {
          sv.assert  %clock : i1
          sv.assume %clock : i1
          sv.cover %clock : i1
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
