// RUN: circt-opt --llhd-lower-processes %s | FileCheck %s

// CHECK-LABEL: @Trivial(
hw.module @Trivial() {
  // CHECK:      scf.execute_region {
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   scf.yield
  // CHECK-NEXT: }
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb1
  }
}

// CHECK-LABEL: @BlockArgs(
hw.module @BlockArgs(in %a: i42, in %b: i42) {
  // CHECK:      scf.execute_region {
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   cf.br ^bb2
  // CHECK-NEXT: ^bb2:
  // CHECK-NEXT:   scf.yield
  // CHECK-NEXT: }
  llhd.process {
    cf.br ^bb1
  ^bb1:
    cf.br ^bb4(%a : i42)
  ^bb2:
    cf.br ^bb3
  ^bb3:
    cf.br ^bb4(%a : i42)
  ^bb4(%0: i42):
    llhd.wait (%a : i42), ^bb2
  }
}

// CHECK-LABEL: @SupportYieldOperands(
hw.module @SupportYieldOperands(in %a: i42) {
  // CHECK:      scf.execute_region -> i42 {
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   scf.yield %a : i42
  // CHECK-NEXT: }
  llhd.process -> i42 {
    cf.br ^bb1
  ^bb1:
    llhd.wait yield (%a : i42), (%a : i42), ^bb1
  }
}

// CHECK-LABEL: @SupportSeparateProbesOfSameValue(
hw.module @SupportSeparateProbesOfSameValue() {
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : i42
  %0 = llhd.prb %a : !hw.inout<i42>
  %1 = llhd.prb %a : !hw.inout<i42>
  // CHECK: scf.execute_region
  llhd.process -> i42 {
    cf.br ^bb1
  ^bb1:
    llhd.wait yield (%0 : i42), (%1 : i42), ^bb1
  }
}

// CHECK-LABEL: @SupportObservedArrays(
hw.module @SupportObservedArrays(in %a: i42, in %b: i42) {
  // CHECK: scf.execute_region
  %0 = hw.array_create %a, %b : i42
  llhd.process {
    cf.br ^bb1
  ^bb1:
    comb.add %a, %b : i42
    llhd.wait (%0 : !hw.array<2xi42>), ^bb1
  }
}

// CHECK-LABEL: @SupportObservedStructs(
hw.module @SupportObservedStructs(in %a: i42, in %b: i42) {
  // CHECK: scf.execute_region
  %0 = hw.struct_create (%a, %b) : !hw.struct<a: i42, b: i42>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    comb.add %a, %b : i42
    llhd.wait (%0 : !hw.struct<a: i42, b: i42>), ^bb1
  }
}

// CHECK-LABEL: @SupportObservedConcats(
hw.module @SupportObservedConcats(in %a: i42, in %b: i42) {
  // CHECK: scf.execute_region
  %0 = comb.concat %a, %b : i42, i42
  llhd.process {
    cf.br ^bb1
  ^bb1:
    comb.add %a, %b : i42
    llhd.wait (%0 : i84), ^bb1
  }
}

// CHECK-LABEL: @SupportObservedBitcasts(
hw.module @SupportObservedBitcasts(in %a: i42, in %b: i42) {
  // CHECK: scf.execute_region
  %0 = hw.bitcast %a : (i42) -> !hw.array<2xi21>
  %1 = hw.bitcast %b : (i42) -> !hw.array<3xi14>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    comb.add %a, %b : i42
    llhd.wait (%0, %1 : !hw.array<2xi21>, !hw.array<3xi14>), ^bb1
  }
}

// CHECK-LABEL: @CommonPattern1(
hw.module @CommonPattern1(in %a: i42, in %b: i42, in %c: i1) {
  // CHECK:      scf.execute_region -> i42 {
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   cf.cond_br %c, ^bb2(%a : i42), ^bb2(%b : i42)
  // CHECK-NEXT: ^bb2([[ARG:%.+]]: i42):
  // CHECK-NEXT:   scf.yield [[ARG]] : i42
  // CHECK-NEXT: }
  %0 = llhd.process -> i42 {
    cf.br ^bb2(%a, %b : i42, i42)
  ^bb1:
    cf.br ^bb2(%a, %b : i42, i42)
  ^bb2(%1: i42, %2: i42):
    cf.cond_br %c, ^bb3(%1 : i42), ^bb3(%2 : i42)
  ^bb3(%3: i42):
    llhd.wait yield (%3 : i42), (%a, %b, %c, %0 : i42, i42, i1, i42), ^bb1
  }
}

// CHECK-LABEL: @SkipIfMultipleWaits(
hw.module @SkipIfMultipleWaits() {
  // CHECK: llhd.process
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb2
  ^bb2:
    llhd.wait ^bb1
  }
}

// CHECK-LABEL: @SkipIfNoWaits(
hw.module @SkipIfNoWaits() {
  // CHECK: llhd.process
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.halt
  }
}

// CHECK-LABEL: @SkipIfWaitHasDestinationOperands(
hw.module @SkipIfWaitHasDestinationOperands(in %a: i42) {
  // CHECK: llhd.process
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb2(%a : i42)
  ^bb2(%0: i42):
    cf.br ^bb1
  }
}

// CHECK-LABEL: @SkipIfEntryAndWaitConvergeInWrongSpot(
hw.module @SkipIfEntryAndWaitConvergeInWrongSpot(in %a: i42) {
  // CHECK: llhd.process
  llhd.process {
    cf.br ^bb2  // skip logic after wait
  ^bb1:
    %0 = comb.add %a, %a : i42
    cf.br ^bb2
  ^bb2:
    llhd.wait ^bb1
  }
}

// CHECK-LABEL: @SkipIfEntryAndWaitConvergeWithDifferentBlockArgs(
hw.module @SkipIfEntryAndWaitConvergeWithDifferentBlockArgs(in %a: i42, in %b: i42) {
  // CHECK: llhd.process
  llhd.process {
    cf.br ^bb2(%a : i42)
  ^bb1:
    cf.br ^bb2(%b : i42)
  ^bb2(%0: i42):
    llhd.wait ^bb1
  }
}

// CHECK-LABEL: @SkipIfValueUnobserved(
hw.module @SkipIfValueUnobserved(in %a: i42) {
  // CHECK: llhd.process
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %0 = comb.add %a, %a : i42
    llhd.wait ^bb1
  }
}
