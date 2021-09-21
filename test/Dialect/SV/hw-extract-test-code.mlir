// RUN:  circt-opt --sv-extract-test-code %s | FileCheck %s
// CHECK-LABEL: hw.module @issue1246_assert(%clock: i1) attributes {output_file = {directory = "dir3", exclude_from_filelist = true, exclude_replicated_ops = false, name = ""}}
// CHECK: sv.assert
// CHECK: hw.module @issue1246_assume(%clock: i1) 
// CHECK-NOT: attributes 
// CHECK: sv.assume
// CHECK: hw.module @issue1246
// CHECK: sv.bind @__ETC_issue1246_assume in @issue1246 {output_file = {directory = "", exclude_from_filelist = true, exclude_replicated_ops = true, name = "file4"}}
module attributes {firrtl.extract.assert =  {directory = "dir3", exclude_from_filelist = true, exclude_replicated_ops = false, name = ""}, firrtl.extract.assume.bindfile =  {directory = "", exclude_from_filelist = true, exclude_replicated_ops = true, name = "file4"}} {
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
