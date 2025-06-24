// RUN: circt-opt --llhd-inline-calls %s | FileCheck %s
// RUN: circt-opt --llhd-inline-calls --symbol-dce %s | FileCheck %s --check-prefixes=CHECK,CHECK-DCE

// CHECK-LABEL: @Basic
hw.module @Basic(in %a: i42, in %b: i42, out u: i42, out v: i42) {
  // CHECK: llhd.combinational
  %0:2 = llhd.combinational -> i42, i42 {
    // CHECK-NOT: call @foo
    %1:2 = func.call @foo(%a, %b) : (i42, i42) -> (i42, i42)
    // CHECK-NEXT:   cf.br [[BB1:\^.+]]
    // CHECK-NEXT: [[BB1]]:
    // CHECK-NEXT:   [[TMP1:%.+]] = comb.add %a, %b :
    // CHECK-NOT:    call @bar
    // CHECK-NEXT:   [[TMP2:%.+]] = hw.constant 42 :
    // CHECK-NEXT:   [[TMP3:%.+]] = comb.xor %a, [[TMP2]] :
    // CHECK-NEXT:   [[TMP4:%.+]] = comb.mul [[TMP3]], %b :
    // CHECK-NEXT:   cf.br [[BB2:\^.+]]
    // CHECK-NEXT: [[BB2]]:
    // CHECK-NEXT:   cf.br [[BB3:\^.+]]([[TMP1]], [[TMP4]] : i42, i42)
    // CHECK-NEXT: [[BB3]]([[CALLRES0:%.+]]: i42, [[CALLRES1:%.+]]: i42):

    // CHECK-NEXT: scf.execute_region
    %2:2 = scf.execute_region -> (i42, i42) {
      // CHECK-NOT: call @foo
      %3:2 = func.call @foo(%1#0, %1#1) : (i42, i42) -> (i42, i42)
      // CHECK-NEXT:   cf.br [[BB1:\^.+]]
      // CHECK-NEXT: [[BB1]]:
      // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[CALLRES0]], [[CALLRES1]] :
      // CHECK-NOT:    call @bar
      // CHECK-NEXT:   [[TMP2:%.+]] = hw.constant 42 :
      // CHECK-NEXT:   [[TMP3:%.+]] = comb.xor [[CALLRES0]], [[TMP2]] :
      // CHECK-NEXT:   [[TMP4:%.+]] = comb.mul [[TMP3]], [[CALLRES1]] :
      // CHECK-NEXT:   cf.br [[BB2:\^.+]]
      // CHECK-NEXT: [[BB2]]:
      // CHECK-NEXT:   cf.br [[BB3:\^.+]]([[TMP1]], [[TMP4]] : i42, i42)
      // CHECK-NEXT: [[BB3]]([[CALLRES0:%.+]]: i42, [[CALLRES1:%.+]]: i42):

      // CHECK-NEXT: scf.yield [[CALLRES0]], [[CALLRES1]]
      scf.yield %3#0, %3#1 : i42, i42
    }

    llhd.yield %2#0, %2#1 : i42, i42
  }
  hw.output %0#0, %0#1 : i42, i42
}

// CHECK-DCE-NOT: @foo
func.func private @foo(%arg0: i42, %arg1: i42) -> (i42, i42) {
  cf.br ^bb1
^bb1:
  %0 = comb.add %arg0, %arg1 : i42
  %1 = call @bar(%arg0) : (i42) -> i42
  %2 = comb.mul %1, %arg1 : i42
  cf.br ^bb2
^bb2:
  return %0, %2 : i42, i42
}

// CHECK-DCE-NOT: @bar
func.func private @bar(%arg0: i42) -> i42 {
  %0 = hw.constant 42 : i42
  %1 = comb.xor %arg0, %0 : i42
  return %1 : i42
}
