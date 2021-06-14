// RUN:  circt-opt --sv-extract-test-code %s | FileCheck %s

// CHECK-LABEL: hw.module @issue1246_assert
// CHECK: sv.assert
// CHECK: hw.module @issue1246
  hw.module @issue1246(%clock: i1) -> () {
    sv.always posedge %clock  {
      sv.ifdef.procedural "SYNTHESIS"  {
      } else  {
        sv.if %2937  {
          sv.assert %clock : i1
        }
      }
    }
    %2937 = hw.constant 0 : i1
    hw.output
  }
