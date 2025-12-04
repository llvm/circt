// RUN: circt-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: hw.module @EmptyProcess(
hw.module @EmptyProcess() {
  llhd.process {
    llhd.halt
  }
  llhd.combinational {
    llhd.yield
  }
  // CHECK-NEXT: hw.output
}

// CHECK-LABEL: hw.module @DontRemoveEmptyProcessWithResults(
hw.module @DontRemoveEmptyProcessWithResults(in %a: i42, out z: i42) {
  // CHECK-NEXT: llhd.process -> i42 {
  // CHECK-NEXT:   llhd.halt
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output
  %0 = llhd.process -> i42 {
    llhd.halt %a : i42
  }
  hw.output %0 : i42
}

// CHECK-LABEL: hw.module @InlineCombinational(
hw.module @InlineCombinational(in %a: i42, in %b: i42, in %c: i8917, out u: i42, out v: i9001) {
  // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %a, %b
  // CHECK-NEXT: [[TMP2:%.+]] = comb.concat %a, %b, %c
  // CHECK-NEXT: hw.output [[TMP1]], [[TMP2]]
  %0, %1 = llhd.combinational -> i42, i9001 {
    %2 = comb.xor %a, %b : i42
    %3 = comb.concat %a, %b, %c : i42, i42, i8917
    llhd.yield %2, %3 : i42, i9001
  }
  hw.output %0, %1 : i42, i9001
}

// CHECK-LABEL: hw.module @IgnoreMultiBlockHalt
hw.module @IgnoreMultiBlockHalt(in %a : i1, in %b : i1, out v1 : i1, out v2 : i1) {
  // CHECK: llhd.halt %a, %a
  %0:2 = llhd.process -> i1, i1 {
  ^bb0:
    cf.br ^bb1
  ^bb1:
    cf.cond_br %b, ^bb1, ^bb2
  ^bb2:
    %true = hw.constant true
    llhd.halt %a, %a : i1, i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// CHECK-LABEL: hw.module @DeduplicateHaltOperands0
hw.module @DeduplicateHaltOperands0(in %a : i1, in %b : i1,
                                    out v1 : i1, out v2 : i1, out v3 : i1, out v4 : i1) {
  // CHECK:      %0:2 = llhd.process -> i1, i1 {
  // CHECK-NEXT:   llhd.halt %a, %b : i1, i1
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output %0#0, %0#1, %0#0, %0#1
  %false = hw.constant false
  %0:6 = llhd.process -> i1, i1, i1, i1, i1, i1 {
    %true = hw.constant true
    llhd.halt %false, %a, %b, %a, %true, %b : i1, i1, i1, i1, i1, i1
  }
  hw.output %0#1, %0#2, %0#3, %0#5 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @CanonProcessHalt0
hw.module @CanonProcessHalt0(out v1 : i1, out v2 : i1) {
  // CHECK-NOT: llhd.halt
  // CHECK: hw.output %false, %true
  %false = hw.constant false
  %0:2 = llhd.process -> i1, i1 {
    %true = hw.constant true
    llhd.halt %false, %true : i1, i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// CHECK-LABEL: hw.module @CanonProcessHalt1
hw.module @CanonProcessHalt1(in %a : i1, in %b : i1,
                             out v1 : i1, out v2 : i1, out v3 : i1, out v4 : i1) {
  // CHECK:      %0:2 = llhd.process -> i1, i1 {
  // CHECK-NEXT:   llhd.halt %a, %b : i1, i1
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output %0#1, %false, %0#0, %true
  %0:4 = llhd.process -> i1, i1, i1, i1 {
    %false = hw.constant false
    %true = hw.constant true
    llhd.halt %false, %a, %true, %b : i1, i1, i1, i1
  }
  hw.output %0#3, %0#0, %0#1, %0#2 : i1, i1, i1, i1
}
