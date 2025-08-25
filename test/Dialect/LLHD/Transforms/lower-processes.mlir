// RUN: circt-opt --llhd-lower-processes %s | FileCheck %s

func.func private @dummy()

// CHECK-LABEL: @Trivial(
hw.module @Trivial() {
  // CHECK:      llhd.combinational {
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   llhd.yield
  // CHECK-NEXT: }
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb1
  }
}

// CHECK-LABEL: @BlockArgs(
hw.module @BlockArgs(in %a: i42, in %b: i42) {
  // CHECK:      llhd.combinational {
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   cf.br ^bb2
  // CHECK-NEXT: ^bb2:
  // CHECK-NEXT:   llhd.yield
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
  // CHECK:      llhd.combinational -> i42 {
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   llhd.yield %a : i42
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
  // CHECK: llhd.combinational
  llhd.process -> i42 {
    cf.br ^bb1
  ^bb1:
    llhd.wait yield (%0 : i42), (%1 : i42), ^bb1
  }
}

// CHECK-LABEL: @SupportObservedArrays(
hw.module @SupportObservedArrays(in %a: i42, in %b: i42) {
  // CHECK: llhd.combinational
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
  // CHECK: llhd.combinational
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
  // CHECK: llhd.combinational
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
  // CHECK: llhd.combinational
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
  // CHECK:      llhd.combinational -> i42 {
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   cf.cond_br %c, ^bb2(%a : i42), ^bb2(%b : i42)
  // CHECK-NEXT: ^bb2([[ARG:%.+]]: i42):
  // CHECK-NEXT:   llhd.yield [[ARG]] : i42
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
hw.module @SkipIfWaitHasDestinationOperands(in %a: i42, in %b: i42) {
  // CHECK: llhd.process
  // CHECK: llhd.wait yield ({{.*}} : i42), ^bb2({{.*}} : i42)
  %0 = llhd.process -> i42 {
    cf.br ^bb1(%a : i42)
  ^bb1(%1: i42):
    %false = hw.constant false
    %c100 = hw.constant 100 : i42
    cf.cond_br %false, ^bb2(%1 : i42), ^bb3(%c100 : i42)
  ^bb2(%2: i42):
    llhd.wait yield (%2 : i42), ^bb2(%2 : i42)
  ^bb3(%3: i42):
    %4 = comb.add %3, %b : i42
    cf.br ^bb1(%4 : i42)
  }
}

// CHECK-LABEL: @SkipIfEntryAndWaitConvergeInWrongSpot(
hw.module @SkipIfEntryAndWaitConvergeInWrongSpot(in %a: i42) {
  // CHECK: llhd.process
  %c100 = hw.constant 100 : i42
  llhd.process {
    cf.br ^bb2(%c100 : i42)
  ^bb1:
    %0 = comb.add %a, %a : i42
    cf.br ^bb2(%0 : i42)
  ^bb2(%1 : i42):
    llhd.wait (%1 : i42), ^bb1
  }
}

// CHECK-LABEL: @SkipIfEntryAndWaitConvergeWithDifferentBlockArgs(
hw.module @SkipIfEntryAndWaitConvergeWithDifferentBlockArgs(in %a: i42, in %b: i42) {
  // CHECK: llhd.process
  %0 = llhd.process -> i42 {
    cf.br ^bb2(%a : i42)
  ^bb1:
    cf.br ^bb2(%b : i42)
  ^bb2(%0: i42):
    llhd.wait yield (%0 : i42), ^bb1
  }
}

// CHECK-LABEL: @SkipIfValueUnobserved(
hw.module @SkipIfValueUnobserved(in %a: i42) {
  // CHECK: llhd.process
  %0 = llhd.process -> i42 {
    cf.br ^bb1
  ^bb1:
    %0 = comb.add %a, %a : i42
    llhd.wait yield (%0 : i42), ^bb1
  }
}

// CHECK-LABEL: @AllowEntryAndWaitToConvergeWithEquivalentBlockArgs(
hw.module @AllowEntryAndWaitToConvergeWithEquivalentBlockArgs(in %a : i42) {
  // CHECK:      llhd.combinational -> i42 {
  // CHECK-NEXT:   [[ADD:%.+]] = comb.add %a, %a : i42
  // CHECK-NEXT:   cf.br ^bb1
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT:   llhd.yield [[ADD]] : i42
  // CHECK-NEXT: }
  %0 = llhd.process -> i42 {
    %1 = comb.add %a, %a : i42
    cf.br ^bb2(%1 : i42)
  ^bb1:
    %2 = comb.add %a, %a : i42
    cf.br ^bb2(%2 : i42)
  ^bb2(%3: i42):
    llhd.wait yield (%3 : i42), (%a : i42), ^bb1
  }
}

// CHECK-LABEL: @SkipIfEntryAndWaitConvergeWithSideEffectingOps(
hw.module @SkipIfEntryAndWaitConvergeWithSideEffectingOps(in %a : i42) {
  // CHECK: llhd.process
  %0 = llhd.process -> i42 {
    func.call @dummy() : () -> ()
    cf.br ^bb2
  ^bb1:
    func.call @dummy() : () -> ()
    cf.br ^bb2
  ^bb2:
    llhd.wait yield (%a : i42), (%a : i42), ^bb1
  }
}

// CHECK-LABEL: @PruneWaitOperands
hw.module @PruneWaitOperands(in %clock : i1, in %f1 : i2, in %f2 : i3) {
  %false = hw.constant false
  %c10_i8 = hw.constant 10 : i8
  %c20_i8 = hw.constant 20 : i8
  %c100_i10 = hw.constant 100 : i10
  %c110_i10 = hw.constant 110 : i10
  %true = hw.constant true
  // CHECK:      llhd.process -> i1, i8, i2, i10, i3 {
  // CHECK-NEXT:   cf.br ^bb1(%c10_i8, %c100_i10 : i8, i10)
  // CHECK-NEXT: ^bb1(%1: i8, %2: i10):
  // CHECK-NEXT:   llhd.wait yield (%clock, %1, %f1, %2, %f2 : i1, i8, i2, i10, i3), (%clock : i1), ^bb2
  // CHECK-NEXT: ^bb2:
  // CHECK:        cf.cond_br {{.*}}, ^bb3, ^bb1(%c20_i8, %c110_i10 : i8, i10)
  // CHECK-NEXT: ^bb3:
  // CHECK:        cf.cond_br {{.*}}, ^bb1(%c10_i8, %c100_i10 : i8, i10), ^bb1(%c20_i8, %c110_i10 : i8, i10)
  // CHECK-NEXT: }
  %0:5 = llhd.process -> i1, i8, i2, i10, i3 {
    cf.br ^bb1(%clock, %c10_i8, %f1, %c100_i10, %f2 : i1, i8, i2, i10, i3)
  ^bb1(%1: i1, %2: i8, %3: i2, %4: i10, %5: i3):
    llhd.wait yield (%1, %2, %3, %4, %5 : i1, i8, i2, i10, i3), (%clock : i1), ^bb2(%2, %1, %3, %5, %4 : i8, i1, i2, i3, i10)
  ^bb2(%6: i8, %7: i1, %8: i2, %9: i3, %10: i10):
    %15 = comb.xor bin %7, %true : i1
    %16 = comb.and bin %15, %clock : i1
    cf.cond_br %16, ^bb3, ^bb1(%clock, %c20_i8, %f1, %c110_i10, %f2 : i1, i8, i2, i10, i3)
  ^bb3:
    %25 = comb.and %true, %clock : i1
    cf.cond_br %25,
      ^bb1(%clock, %c10_i8, %f1, %c100_i10, %f2 : i1, i8, i2, i10, i3),
      ^bb1(%clock, %c20_i8, %f1, %c110_i10, %f2 : i1, i8, i2, i10, i3)
  }
  hw.output
}
