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
