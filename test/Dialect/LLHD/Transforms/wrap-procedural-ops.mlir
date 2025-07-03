// RUN: circt-opt --llhd-wrap-procedural-ops %s | FileCheck %s

func.func private @someFunc(%arg0: i42, %arg1: i9001) -> (i43, i9002)

// CHECK-LABEL: @Calls(
hw.module @Calls(in %a: i42, in %b: i9001, out u: i43, out v: i9002) {
  // CHECK-NEXT: [[TMP1:%.+]]:2 = llhd.combinational -> i43, i9002 {
  // CHECK-NEXT:   [[TMP2:%.+]]:2 = func.call @someFunc
  // CHECK-NEXT:   llhd.yield [[TMP2]]#0, [[TMP2]]#1
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output [[TMP1]]#0, [[TMP1]]#1
  %0:2 = func.call @someFunc(%a, %b) : (i42, i9001) -> (i43, i9002)
  hw.output %0#0, %0#1 : i43, i9002
}

// CHECK-LABEL: @Ifs(
hw.module @Ifs(in %a: i42, in %b: i42, in %c: i1, out z: i42) {
  // CHECK-NEXT: [[TMP1:%.+]] = llhd.combinational -> i42 {
  // CHECK-NEXT:   [[TMP2:%.+]] = scf.if %c -> (i42) {
  // CHECK-NEXT:     scf.yield %a
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     scf.yield %b
  // CHECK-NEXT:   }
  // CHECK-NEXT:   llhd.yield [[TMP2]]
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output [[TMP1]]
  %0 = scf.if %c -> (i42) {
    scf.yield %a : i42
  } else {
    scf.yield %b : i42
  }
  hw.output %0 : i42
}
