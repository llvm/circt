// RUN: circt-opt --lowering-options=spillWiresForNamehints --export-verilog %s | FileCheck %s
// RUN: circt-opt  --export-verilog %s | FileCheck %s --check-prefix=NORMAL

// CHECK-LABEL: module namehint
hw.module @namehint(%a: i5, %b: i5) -> (c: i5) {
    // CHECK:       wire [4:0] v1 = a + a;
    // CHECK-NEXT:  wire [4:0] v2 = b + v1;
    // CHECK-NEXT:  assign c = v2;
    // NORMAL:      assign c = b + a + a;
    %0 = comb.add %a, %a {sv.namehint = "v1"} : i5
    %1 = comb.add %b, %0 {sv.namehint = "v2"} : i5
    hw.output %1 : i5
}
