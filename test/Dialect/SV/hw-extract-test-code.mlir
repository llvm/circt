// RUN:  circt-opt --sv-extract-test-code %s | FileCheck %s
// CHECK-LABEL: hw.module @issue1246_assert(%clock: i1) attributes {output_file = #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>}
// CHECK: sv.assert
// CHECK: hw.module @issue1246_assume(%clock: i1) 
// CHECK-NOT: attributes 
// CHECK: sv.assume
// CHECK: hw.module @issue1246
// CHECK: sv.bind @__ETC_issue1246_assume in @issue1246 {output_file = #hw.output_file<"file4", excludeFromFileList>}
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>, firrtl.extract.assume.bindfile = #hw.output_file<"file4", excludeFromFileList>} {
  hw.module @issue1246(%clock: i1) -> () {
    sv.always posedge %clock  {
      sv.ifdef.procedural "SYNTHESIS"  {
      } else  {
        sv.if %2937  {
          sv.assert  %clock : i1
          sv.assume %clock : i1
        }
      }
    }
    %2937 = hw.constant 0 : i1
    hw.output
  }
}
