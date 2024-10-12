// RUN: circt-opt %s -verify-diagnostics -seq-firreg-randomization | FileCheck %s --check-prefixes=COMMON,CHECK
// RUN: circt-opt %s -verify-diagnostics --pass-pipeline='builtin.module(seq-firreg-randomization{emit-sv=true})'  | FileCheck %s --check-prefixes=COMMON,SV
sv.macro.decl @INIT_RANDOM_PROLOG_
hw.module @top(in %clk: !seq.clock, in %rst: i1, in %i: i18, out o: i18, out j: i18) {
  %c0_i18 = hw.constant 0 : i18
  %r0 = seq.compreg %i, %clk reset %rst, %c0_i18 : i18
  %r1 = seq.compreg %i, %clk : i18
  // COMMON:      %r0 = seq.compreg %i, %clk reset %rst, %c0_i18 initial %0#0 : i18
  // COMMON-NEXT: %r1 = seq.compreg %i, %clk initial %0#1 : i18
  // COMMON-NEXT: %0:2 = seq.initial {
  // CHECK-NEXT:   %[[RAND1:.+]] = func.call @random() : () -> i32
  // CHECK-NEXT:   %[[RAND2:.+]] = func.call @random() : () -> i32
  // CHECK-NEXT:   %[[EXTRACT1:.+]] = comb.extract %[[RAND1]] from 0 : (i32) -> i18
  // CHECK-NEXT:   %[[EXTRACT2:.+]] = comb.extract %[[RAND1]] from 18 : (i32) -> i14
  // CHECK-NEXT:   %[[EXTRACT3:.+]] = comb.extract %[[RAND2]] from 0 : (i32) -> i4
  // CHECK-NEXT:   %[[CONCAT:.+]] = comb.concat %[[EXTRACT2]], %[[EXTRACT3]] : i14, i4
  // CHECK-NEXT:   seq.yield %[[EXTRACT1]], %[[CONCAT]] : i18, i18

  // SV-NEXT:      sv.verbatim "`INIT_RANDOM_PROLOG_"
  // SV-NEXT:      %RANDOM = sv.macro.ref.se @RANDOM() : () -> i32
  // SV-NEXT:      %RANDOM_0 = sv.macro.ref.se @RANDOM() : () -> i32
  // SV-NEXT:      %[[EXTRACT1:.+]] = comb.extract %RANDOM from 0 : (i32) -> i18
  // SV-NEXT:      %[[EXTRACT2:.+]] = comb.extract %RANDOM from 18 : (i32) -> i14
  // SV-NEXT:      %[[EXTRACT3:.+]] = comb.extract %RANDOM_0 from 0 : (i32) -> i4
  // SV-NEXT:      %[[CONCAT:.+]] = comb.concat %[[EXTRACT2]], %[[EXTRACT3]] : i14, i4
  // SV-NEXT:      seq.yield %[[EXTRACT1]], %[[CONCAT]] : i18, i18

  // COMMON: } : !seq.immutable<i18>, !seq.immutable<i18>

  hw.output %r0, %r1: i18, i18
}
