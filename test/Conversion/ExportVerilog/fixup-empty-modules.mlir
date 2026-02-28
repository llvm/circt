// RUN: circt-opt --test-apply-lowering-options='options=fixUpEmptyModules' --export-verilog %s | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: module empty1
hw.module @empty1() {
    // CHECK-NEXT: /* This wire is added to avoid emitting empty modules. See `fixUpEmptyModules` lowering option in CIRCT. */ 
    // CHECK-NEXT:  wire _GEN = 1'h1;
}

// CHECK-LABEL: module empty2
hw.module @empty2(in %in: i1, in %in2: i32) {
    // CHECK: /* This wire is added to avoid emitting empty modules. See `fixUpEmptyModules` lowering option in CIRCT. */ 
    // CHECK-NEXT:  wire _GEN = 1'h1;
}

// CHECK-LABEL: module not_empty
hw.module @not_empty(in %in: i1, out out: i1) {
    // CHECK:  assign out = in;
    hw.output %in : i1
}
