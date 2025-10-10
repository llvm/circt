// RUN: circt-opt --llhd-deseq %s | FileCheck %s

// CHECK-LABEL: @ClockPosEdge(
hw.module @ClockPosEdge(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.firreg %d clock [[CLK]] : i42
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock : i1), ^bb2(%clock : i1)
  ^bb2(%5: i1):
    %6 = comb.xor bin %5, %true : i1
    %7 = comb.and bin %6, %clock : i1  // posedge clock
    cf.cond_br %7, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ClockNegEdge(
hw.module @ClockNegEdge(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[CLK_INV:%.+]] = seq.clock_inv [[CLK]]
  // CHECK: [[REG:%.+]] = seq.firreg %d clock [[CLK_INV]] : i42
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock : i1), ^bb2(%clock : i1)
  ^bb2(%5: i1):
    %6 = comb.xor bin %clock, %true : i1
    %7 = comb.and bin %5, %6 : i1  // negedge clock
    cf.cond_br %7, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ClockPosEdgeWithActiveLowReset(
hw.module @ClockPosEdgeWithActiveLowReset(in %clock: i1, in %reset: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[RST_INV:%.+]] = comb.xor %reset, %true
  // CHECK: [[REG:%.+]] = seq.firreg %d clock [[CLK]] reset async [[RST_INV]], %c42_i42 : i42
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    %c42_i42 = hw.constant 42 : i42
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock, %reset : i1, i1), ^bb2(%clock, %reset : i1, i1)
  ^bb2(%5: i1, %6: i1):
    %7 = comb.xor bin %5, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    %9 = comb.xor bin %reset, %true : i1
    %10 = comb.and bin %6, %9 : i1  // negedge reset
    %11 = comb.or bin %8, %10 : i1
    cf.cond_br %11, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset, ^bb1(%d, %true : i42, i1), ^bb1(%c42_i42, %true : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ClockNegEdgeWithActiveHighReset(
hw.module @ClockNegEdgeWithActiveHighReset(in %clock: i1, in %reset: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[CLK_INV:%.+]] = seq.clock_inv [[CLK]]
  // CHECK: [[REG:%.+]] = seq.firreg %d clock [[CLK_INV]] reset async %reset, %c42_i42 : i42
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    %c42_i42 = hw.constant 42 : i42
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock, %reset : i1, i1), ^bb2(%clock, %reset : i1, i1)
  ^bb2(%5: i1, %6: i1):
    %7 = comb.xor bin %clock, %true : i1
    %8 = comb.and bin %5, %7 : i1  // negedge clock
    %9 = comb.xor bin %6, %true : i1
    %10 = comb.and bin %9, %reset : i1  // posedge reset
    %11 = comb.or bin %8, %10 : i1
    cf.cond_br %11, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset, ^bb1(%c42_i42, %true : i42, i1), ^bb1(%d, %true : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ClockWithEnable(
hw.module @ClockWithEnable(in %clock: i1, in %d: i42, in %en: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[ER:%.+]]:2 = llhd.combinational -> i42, i1 {
  // CHECK:   cf.cond_br %en, [[BB:\^.+]](%d, %true : i42, i1), [[BB]](%c0_i42, %false : i42, i1)
  // CHECK: [[BB]]([[RESULT:%.+]]: i42, [[ENABLE:%.+]]: i1):
  // CHECK:   llhd.yield [[RESULT]], [[ENABLE]] : i42, i1
  // CHECK: }
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock : i1), ^bb2(%clock : i1)
  ^bb2(%5: i1):
    %6 = comb.xor bin %5, %true : i1
    %7 = comb.and bin %6, %clock : i1  // posedge clock
    cf.cond_br %7, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %en, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[MUX:%.+]] = comb.mux bin [[ER]]#1, [[ER]]#0, [[REG:%.+]] : i42
  // CHECK: [[REG]] = seq.firreg [[MUX]] clock [[CLK]] : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ClockWithEnableAndReset(
hw.module @ClockWithEnableAndReset(in %clock: i1, in %reset: i1, in %d: i42, in %en: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[ER:%.+]]:2 = llhd.combinational -> i42, i1 {
  // CHECK:   cf.cond_br %en, [[BB:\^.+]](%d, %true : i42, i1), [[BB]](%c0_i42, %false : i42, i1)
  // CHECK: [[BB]]([[RESULT:%.+]]: i42, [[ENABLE:%.+]]: i1):
  // CHECK:   llhd.yield [[RESULT]], [[ENABLE]] : i42, i1
  // CHECK: }
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    %c42_i42 = hw.constant 42 : i42
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock, %reset : i1, i1), ^bb2(%clock, %reset : i1, i1)
  ^bb2(%5: i1, %6: i1):
    %7 = comb.xor bin %5, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    %9 = comb.xor bin %6, %true : i1
    %10 = comb.and bin %9, %reset : i1  // posedge reset
    %11 = comb.or bin %8, %10 : i1
    cf.cond_br %11, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset, ^bb1(%c42_i42, %true : i42, i1), ^bb4
  ^bb4:
    cf.cond_br %en, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[MUX:%.+]] = comb.mux bin [[ER]]#1, [[ER]]#0, [[REG:%.+]] : i42
  // CHECK: [[REG]] = seq.firreg [[MUX]] clock [[CLK]] reset async %reset, %c42_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ChasePastValuesThroughControlFlow(
hw.module @ChasePastValuesThroughControlFlow(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: seq.firreg
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false, %clock : i42, i1, i1)
  ^bb1(%3: i42, %4: i1, %5: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock : i1), ^bb2(%5 : i1)
  ^bb2(%6: i1):
    %7 = comb.xor bin %6, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    cf.cond_br %8, ^bb3, ^bb1(%c0_i42, %false, %clock : i42, i1, i1)
  ^bb3:
    cf.cond_br %clock, ^bb1(%d, %true, %true : i42, i1, i1), ^bb1(%d, %true, %false : i42, i1, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @AbortIfPastValueUnobserved(
hw.module @AbortIfPastValueUnobserved(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.process
  // CHECK-NOT: seq.compreg
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), ^bb2(%clock : i1)
  ^bb2(%5: i1):
    %6 = comb.xor bin %5, %true : i1
    %7 = comb.and bin %6, %clock : i1  // posedge clock
    cf.cond_br %7, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @AbortIfPastValueNotI1(
hw.module @AbortIfPastValueNotI1(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.process
  // CHECK-NOT: seq.compreg
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock : i1), ^bb2(%clock, %d : i1, i42)
  ^bb2(%5: i1, %6: i42):
    %7 = comb.xor bin %5, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    cf.cond_br %8, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @AbortIfPastValueLocal(
hw.module @AbortIfPastValueLocal(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.process
  // CHECK-NOT: seq.compreg
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), ^bb2(%true : i1)
  ^bb2(%5: i1):
    %6 = comb.xor bin %5, %true : i1
    %7 = comb.and bin %6, %clock : i1  // posedge clock
    cf.cond_br %7, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @AbortIfMultipleClocks(
hw.module @AbortIfMultipleClocks(in %clock1: i1, in %clock2: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.process
  // CHECK-NOT: seq.compreg
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock1, %clock2 : i1, i1), ^bb2(%clock1, %clock2 : i1, i1)
  ^bb2(%5: i1, %6: i1):
    %7 = comb.xor bin %5, %true : i1
    %8 = comb.and bin %7, %clock1 : i1  // posedge clock1
    %9 = comb.xor bin %6, %true : i1
    %10 = comb.and bin %9, %clock2 : i1  // posedge clock2
    %11 = comb.or bin %8, %10 : i1
    cf.cond_br %11, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @AbortIfMultipleResets(
hw.module @AbortIfMultipleResets(in %clock: i1, in %reset1: i1, in %reset2: i1, in %d: i42, in %en: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.process
  // CHECK-NOT: seq.compreg
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    %c42_i42 = hw.constant 42 : i42
    %c43_i42 = hw.constant 43 : i42
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock, %reset1, %reset2 : i1, i1, i1), ^bb2(%clock, %reset1, %reset2 : i1, i1, i1)
  ^bb2(%5: i1, %6: i1, %7: i1):
    %8 = comb.xor bin %5, %true : i1
    %9 = comb.and bin %8, %clock : i1  // posedge clock
    %10 = comb.xor bin %6, %true : i1
    %11 = comb.and bin %10, %reset1 : i1  // posedge reset1
    %12 = comb.xor bin %7, %true : i1
    %13 = comb.and bin %12, %reset2 : i1  // posedge reset2
    %14 = comb.or bin %9, %11, %13 : i1
    cf.cond_br %14, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset1, ^bb1(%c42_i42, %true : i42, i1), ^bb4
  ^bb4:
    cf.cond_br %reset2, ^bb1(%c43_i42, %true : i42, i1), ^bb1(%d, %true : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @AbortOnAndOfMultipleEdges(
hw.module @AbortOnAndOfMultipleEdges(in %clock: i1, in %reset: i1, in %d: i42, in %en: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.process
  // CHECK-NOT: seq.compreg
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    %c42_i42 = hw.constant 42 : i42
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock, %reset : i1, i1), ^bb2(%clock, %reset : i1, i1)
  ^bb2(%5: i1, %6: i1):
    %7 = comb.xor bin %5, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    %9 = comb.xor bin %6, %true : i1
    %10 = comb.and bin %9, %reset : i1  // posedge reset
    %11 = comb.and bin %8, %10 : i1  // <- (posedge clock) AND (posedge reset)
    cf.cond_br %11, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset, ^bb1(%c42_i42, %true : i42, i1), ^bb1(%d, %true : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @AcceptMuxForReset(
hw.module @AcceptMuxForReset(in %clock: i1, in %reset: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.firreg %d clock [[CLK]] reset async %reset, %c42_i42 : i42
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    %c42_i42 = hw.constant 42 : i42
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock, %reset : i1, i1), ^bb2(%clock, %reset : i1, i1)
  ^bb2(%5: i1, %6: i1):
    %7 = comb.xor bin %5, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    %9 = comb.xor bin %6, %true : i1
    %10 = comb.and bin %9, %reset : i1  // posedge reset
    %11 = comb.or bin %8, %10 : i1
    cf.cond_br %11, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    %12 = comb.mux %reset, %c42_i42, %d : i42
    cf.br ^bb1(%12, %true : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ComplexControlFlow(
hw.module @ComplexControlFlow(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[ER:%.+]]:2 = llhd.combinational -> i42, i1 {
  // CHECK:   cf.br ^bb1(%c0_i42, %false : i42, i1)
  // CHECK: ^bb1({{%.+}}: i42, {{%.+}}: i1):
  // CHECK:   cf.br ^bb2
  // CHECK: ^bb2:
  // CHECK:   cf.br ^bb3(%c0_i42, %c0_i42 : i42, i42)
  // CHECK: ^bb3({{%.+}}: i42, {{%.+}}: i42):
  // CHECK:   comb.shru
  // CHECK:   comb.extract
  // CHECK:   comb.concat
  // CHECK:   [[RESULT:%.+]] = comb.add
  // CHECK:   comb.add
  // CHECK:   [[TMP:%.+]] = comb.icmp ult
  // CHECK:   cf.cond_br [[TMP]], ^bb3({{.+}}), ^bb4([[RESULT]], %true : i42, i1)
  // CHECK: ^bb4([[RESULT:%.+]]: i42, [[ENABLE:%.+]]: i1):
  // CHECK:   llhd.yield [[RESULT]], [[ENABLE]] : i42, i1
  // CHECK: }
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock : i1), ^bb2(%clock : i1)
  ^bb2(%5: i1):
    %6 = comb.xor bin %5, %true : i1
    %7 = comb.and bin %6, %clock : i1  // posedge clock
    cf.cond_br %7, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.br ^bb4(%c0_i42, %c0_i42 : i42, i42)
  ^bb4(%8: i42, %9: i42):
    %10 = comb.shru %d, %8 : i42
    %11 = comb.extract %10 from 0 : (i42) -> i1
    %c0_i41 = hw.constant 0 : i41
    %12 = comb.concat %c0_i41, %11 : i41, i1
    %13 = comb.add %9, %12 : i42
    %c1_i42 = hw.constant 1 : i42
    %c42_i42 = hw.constant 42 : i42
    %14 = comb.add %8, %c1_i42 : i42
    %15 = comb.icmp ult %14, %c42_i42 : i42
    cf.cond_br %15, ^bb4(%14, %13 : i42, i42), ^bb1(%13, %true : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[MUX:%.+]] = comb.mux bin [[ER]]#1, [[ER]]#0, [[REG:%.+]] : i42
  // CHECK: [[REG]] = seq.firreg [[MUX]] clock [[CLK]] : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ClockAndResetSameConst(
hw.module @ClockAndResetSameConst(in %clock: i1, in %reset: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.firreg %c42_i42 clock [[CLK]] reset async %reset, %c42_i42 : i42
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    %c42_i42 = hw.constant 42 : i42
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock, %reset : i1, i1), ^bb2(%clock, %reset : i1, i1)
  ^bb2(%5: i1, %6: i1):
    %7 = comb.xor bin %5, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    %9 = comb.xor bin %6, %true : i1
    %10 = comb.and bin %9, %reset : i1  // posedge reset
    %11 = comb.or bin %8, %10 : i1
    cf.cond_br %11, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset, ^bb1(%c42_i42, %true : i42, i1), ^bb1(%c42_i42, %true : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @ClockAndResetDifferentConst(
hw.module @ClockAndResetDifferentConst(in %clock: i1, in %reset: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.firreg %c42_i42 clock [[CLK]] reset async %reset, %c0_i42 : i42
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    %c42_i42 = hw.constant 42 : i42
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock, %reset : i1, i1), ^bb2(%clock, %reset : i1, i1)
  ^bb2(%5: i1, %6: i1):
    %7 = comb.xor bin %5, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    %9 = comb.xor bin %6, %true : i1
    %10 = comb.and bin %9, %reset : i1  // posedge reset
    %11 = comb.or bin %8, %10 : i1
    cf.cond_br %11, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset, ^bb1(%c0_i42, %true : i42, i1), ^bb1(%c42_i42, %true : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}

// CHECK-LABEL: @NonConstButStaticReset(
hw.module @NonConstButStaticReset(in %clock: i1, in %reset: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %1 = hw.struct_create (%c0_i42, %c0_i42) : !hw.struct<a: i42, b: i42>
  // CHECK: [[FIELD:%.+]] = hw.struct_extract {{%.+}}["a"]
  %2 = hw.struct_extract %1["a"] : !hw.struct<a: i42, b: i42>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.firreg %d clock [[CLK]] reset async %reset, [[FIELD]] : i42
  %3, %4 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%5: i42, %6: i1):
    llhd.wait yield (%5, %6 : i42, i1), (%clock, %reset : i1, i1), ^bb2(%clock, %reset : i1, i1)
  ^bb2(%7: i1, %8: i1):
    %9 = comb.xor bin %7, %true : i1
    %10 = comb.and bin %9, %clock : i1  // posedge clock
    %11 = comb.xor bin %8, %true : i1
    %12 = comb.and bin %11, %reset : i1  // posedge reset
    %13 = comb.or bin %10, %12 : i1
    cf.cond_br %13, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset, ^bb1(%2, %true : i42, i1), ^bb1(%d, %true : i42, i1)
  }
  %5 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %5, %3 after %0 if %4 : i42
}

// The following example of a dual-port memory caused excessive memory usage and
// runtime in a previous implementation of the deseq pass.
// CHECK-LABEL: @LargeControlFlowRegression(
hw.module @LargeControlFlowRegression(in %clk: i1, in %rstn: i1, in %a: i6, in %d0: i15, in %e0: i1, in %d1: i15, in %e1: i1) {
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c-30_i6 = hw.constant -30 : i6
  %c-32_i6 = hw.constant -32 : i6
  %c31_i6 = hw.constant 31 : i6
  %c30_i6 = hw.constant 30 : i6
  %c29_i6 = hw.constant 29 : i6
  %c28_i6 = hw.constant 28 : i6
  %c27_i6 = hw.constant 27 : i6
  %c26_i6 = hw.constant 26 : i6
  %c25_i6 = hw.constant 25 : i6
  %c24_i6 = hw.constant 24 : i6
  %c23_i6 = hw.constant 23 : i6
  %c22_i6 = hw.constant 22 : i6
  %c21_i6 = hw.constant 21 : i6
  %c20_i6 = hw.constant 20 : i6
  %c19_i6 = hw.constant 19 : i6
  %c18_i6 = hw.constant 18 : i6
  %c17_i6 = hw.constant 17 : i6
  %c16_i6 = hw.constant 16 : i6
  %c15_i6 = hw.constant 15 : i6
  %c14_i6 = hw.constant 14 : i6
  %c13_i6 = hw.constant 13 : i6
  %c12_i6 = hw.constant 12 : i6
  %c11_i6 = hw.constant 11 : i6
  %c10_i6 = hw.constant 10 : i6
  %c9_i6 = hw.constant 9 : i6
  %c8_i6 = hw.constant 8 : i6
  %c7_i6 = hw.constant 7 : i6
  %c6_i6 = hw.constant 6 : i6
  %c5_i6 = hw.constant 5 : i6
  %c4_i6 = hw.constant 4 : i6
  %c3_i6 = hw.constant 3 : i6
  %c2_i6 = hw.constant 2 : i6
  %c1_i6 = hw.constant 1 : i6
  %c0_i15 = hw.constant 0 : i15
  %c0_i6 = hw.constant 0 : i6
  %false = hw.constant false
  %c-31_i6 = hw.constant -31 : i6
  %mem0 = llhd.sig %c0_i15 : i15
  %mem1 = llhd.sig %c0_i15 : i15
  %mem2 = llhd.sig %c0_i15 : i15
  %mem3 = llhd.sig %c0_i15 : i15
  %mem4 = llhd.sig %c0_i15 : i15
  %mem5 = llhd.sig %c0_i15 : i15
  %mem6 = llhd.sig %c0_i15 : i15
  %mem7 = llhd.sig %c0_i15 : i15
  %mem8 = llhd.sig %c0_i15 : i15
  %mem9 = llhd.sig %c0_i15 : i15
  %mem10 = llhd.sig %c0_i15 : i15
  %mem11 = llhd.sig %c0_i15 : i15
  %mem12 = llhd.sig %c0_i15 : i15
  %mem13 = llhd.sig %c0_i15 : i15
  %mem14 = llhd.sig %c0_i15 : i15
  %mem15 = llhd.sig %c0_i15 : i15
  %mem16 = llhd.sig %c0_i15 : i15
  %mem17 = llhd.sig %c0_i15 : i15
  %mem18 = llhd.sig %c0_i15 : i15
  %mem19 = llhd.sig %c0_i15 : i15
  %mem20 = llhd.sig %c0_i15 : i15
  %mem21 = llhd.sig %c0_i15 : i15
  %mem22 = llhd.sig %c0_i15 : i15
  %mem23 = llhd.sig %c0_i15 : i15
  %mem24 = llhd.sig %c0_i15 : i15
  %mem25 = llhd.sig %c0_i15 : i15
  %mem26 = llhd.sig %c0_i15 : i15
  %mem27 = llhd.sig %c0_i15 : i15
  %mem28 = llhd.sig %c0_i15 : i15
  %mem29 = llhd.sig %c0_i15 : i15
  %mem30 = llhd.sig %c0_i15 : i15
  %mem31 = llhd.sig %c0_i15 : i15
  %mem32 = llhd.sig %c0_i15 : i15
  %mem33 = llhd.sig %c0_i15 : i15
  %mem34 = llhd.sig %c0_i15 : i15
  // CHECK-NOT: llhd.process
  // CHECK: [[ER:%.+]]:70 = llhd.combinational ->
  %1:70 = llhd.process -> i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1 {
    cf.br ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb1(%2: i1, %3: i1, %4: i15, %5: i1, %6: i15, %7: i1, %8: i15, %9: i1, %10: i15, %11: i1, %12: i15, %13: i1, %14: i15, %15: i1, %16: i15, %17: i1, %18: i15, %19: i1, %20: i15, %21: i1, %22: i15, %23: i1, %24: i15, %25: i1, %26: i15, %27: i1, %28: i15, %29: i1, %30: i15, %31: i1, %32: i15, %33: i1, %34: i15, %35: i1, %36: i15, %37: i1, %38: i15, %39: i1, %40: i15, %41: i1, %42: i15, %43: i1, %44: i15, %45: i1, %46: i15, %47: i1, %48: i15, %49: i1, %50: i15, %51: i1, %52: i15, %53: i1, %54: i15, %55: i1, %56: i15, %57: i1, %58: i15, %59: i1, %60: i15, %61: i1, %62: i15, %63: i1, %64: i15, %65: i1, %66: i15, %67: i1, %68: i15, %69: i1, %70: i15, %71: i1, %72: i15, %73: i1):
    llhd.wait yield (%4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73 : i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), (%clk, %rstn : i1, i1), ^bb2(%2, %3 : i1, i1)
  ^bb2(%74: i1, %75: i1):
    %76 = comb.xor bin %74, %true : i1
    %77 = comb.and bin %76, %clk : i1
    %78 = comb.xor bin %rstn, %true : i1
    %79 = comb.and bin %75, %78 : i1
    %80 = comb.or bin %77, %79 : i1
    cf.cond_br %80, ^bb3, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb3:
    %81 = comb.xor %rstn, %true : i1
    cf.cond_br %81, ^bb1(%clk, %rstn, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true, %c0_i15, %true : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb4
  ^bb4:
    %82 = comb.icmp ceq %a, %c0_i6 : i6
    cf.cond_br %82, ^bb5, ^bb6
  ^bb5:
    %83 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %83, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %83, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb6:
    %84 = comb.icmp ceq %a, %c1_i6 : i6
    cf.cond_br %84, ^bb7, ^bb8
  ^bb7:
    %85 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %85, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %85, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb8:
    %86 = comb.icmp ceq %a, %c2_i6 : i6
    cf.cond_br %86, ^bb9, ^bb10
  ^bb9:
    %87 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %87, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %87, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb10:
    %88 = comb.icmp ceq %a, %c3_i6 : i6
    cf.cond_br %88, ^bb11, ^bb12
  ^bb11:
    %89 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %89, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %89, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb12:
    %90 = comb.icmp ceq %a, %c4_i6 : i6
    cf.cond_br %90, ^bb13, ^bb14
  ^bb13:
    %91 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %91, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %91, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb14:
    %92 = comb.icmp ceq %a, %c5_i6 : i6
    cf.cond_br %92, ^bb15, ^bb16
  ^bb15:
    %93 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %93, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %93, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb16:
    %94 = comb.icmp ceq %a, %c6_i6 : i6
    cf.cond_br %94, ^bb17, ^bb18
  ^bb17:
    %95 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %95, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %95, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb18:
    %96 = comb.icmp ceq %a, %c7_i6 : i6
    cf.cond_br %96, ^bb19, ^bb20
  ^bb19:
    %97 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %97, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %97, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb20:
    %98 = comb.icmp ceq %a, %c8_i6 : i6
    cf.cond_br %98, ^bb21, ^bb22
  ^bb21:
    %99 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %99, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %99, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb22:
    %100 = comb.icmp ceq %a, %c9_i6 : i6
    cf.cond_br %100, ^bb23, ^bb24
  ^bb23:
    %101 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %101, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %101, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb24:
    %102 = comb.icmp ceq %a, %c10_i6 : i6
    cf.cond_br %102, ^bb25, ^bb26
  ^bb25:
    %103 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %103, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %103, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb26:
    %104 = comb.icmp ceq %a, %c11_i6 : i6
    cf.cond_br %104, ^bb27, ^bb28
  ^bb27:
    %105 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %105, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %105, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb28:
    %106 = comb.icmp ceq %a, %c12_i6 : i6
    cf.cond_br %106, ^bb29, ^bb30
  ^bb29:
    %107 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %107, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %107, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb30:
    %108 = comb.icmp ceq %a, %c13_i6 : i6
    cf.cond_br %108, ^bb31, ^bb32
  ^bb31:
    %109 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %109, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %109, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb32:
    %110 = comb.icmp ceq %a, %c14_i6 : i6
    cf.cond_br %110, ^bb33, ^bb34
  ^bb33:
    %111 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %111, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %111, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb34:
    %112 = comb.icmp ceq %a, %c15_i6 : i6
    cf.cond_br %112, ^bb35, ^bb36
  ^bb35:
    %113 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %113, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %113, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb36:
    %114 = comb.icmp ceq %a, %c16_i6 : i6
    cf.cond_br %114, ^bb37, ^bb38
  ^bb37:
    %115 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %115, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %115, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb38:
    %116 = comb.icmp ceq %a, %c17_i6 : i6
    cf.cond_br %116, ^bb39, ^bb40
  ^bb39:
    %117 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %117, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %117, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb40:
    %118 = comb.icmp ceq %a, %c18_i6 : i6
    cf.cond_br %118, ^bb41, ^bb42
  ^bb41:
    %119 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %119, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %119, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb42:
    %120 = comb.icmp ceq %a, %c19_i6 : i6
    cf.cond_br %120, ^bb43, ^bb44
  ^bb43:
    %121 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %121, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %121, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb44:
    %122 = comb.icmp ceq %a, %c20_i6 : i6
    cf.cond_br %122, ^bb45, ^bb46
  ^bb45:
    %123 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %123, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %123, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb46:
    %124 = comb.icmp ceq %a, %c21_i6 : i6
    cf.cond_br %124, ^bb47, ^bb48
  ^bb47:
    %125 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %125, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %125, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb48:
    %126 = comb.icmp ceq %a, %c22_i6 : i6
    cf.cond_br %126, ^bb49, ^bb50
  ^bb49:
    %127 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %127, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %127, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb50:
    %128 = comb.icmp ceq %a, %c23_i6 : i6
    cf.cond_br %128, ^bb51, ^bb52
  ^bb51:
    %129 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %129, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %129, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb52:
    %130 = comb.icmp ceq %a, %c24_i6 : i6
    cf.cond_br %130, ^bb53, ^bb54
  ^bb53:
    %131 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %131, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %131, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb54:
    %132 = comb.icmp ceq %a, %c25_i6 : i6
    cf.cond_br %132, ^bb55, ^bb56
  ^bb55:
    %133 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %133, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %133, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb56:
    %134 = comb.icmp ceq %a, %c26_i6 : i6
    cf.cond_br %134, ^bb57, ^bb58
  ^bb57:
    %135 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %135, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %135, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb58:
    %136 = comb.icmp ceq %a, %c27_i6 : i6
    cf.cond_br %136, ^bb59, ^bb60
  ^bb59:
    %137 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %137, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %137, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb60:
    %138 = comb.icmp ceq %a, %c28_i6 : i6
    cf.cond_br %138, ^bb61, ^bb62
  ^bb61:
    %139 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %139, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %139, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb62:
    %140 = comb.icmp ceq %a, %c29_i6 : i6
    cf.cond_br %140, ^bb63, ^bb64
  ^bb63:
    %141 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %141, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %141, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb64:
    %142 = comb.icmp ceq %a, %c30_i6 : i6
    cf.cond_br %142, ^bb65, ^bb66
  ^bb65:
    %143 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %143, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %143, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb66:
    %144 = comb.icmp ceq %a, %c31_i6 : i6
    cf.cond_br %144, ^bb67, ^bb68
  ^bb67:
    %145 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %145, %e0, %d1, %true, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %145, %e0, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb68:
    %146 = comb.icmp ceq %a, %c-32_i6 : i6
    cf.cond_br %146, ^bb69, ^bb70
  ^bb69:
    %147 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %147, %e0, %d1, %true, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %147, %e0, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb70:
    %148 = comb.icmp ceq %a, %c-31_i6 : i6
    cf.cond_br %148, ^bb71, ^bb72
  ^bb71:
    %149 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %149, %e0, %d1, %true : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %149, %e0, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb72:
    %150 = comb.icmp ceq %a, %c-30_i6 : i6
    cf.cond_br %150, ^bb73, ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  ^bb73:
    %151 = arith.select %e0, %d0, %c0_i15 : i15
    cf.cond_br %e1, ^bb1(%clk, %rstn, %d1, %true, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %151, %e0 : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1), ^bb1(%clk, %rstn, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %c0_i15, %false, %151, %e0 : i1, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1, i15, i1)
  }
  // CHECK: [[CLK:%.+]] = seq.to_clock %clk
  // CHECK: [[RST:%.+]] = comb.xor %rstn, %true
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#1, [[ER]]#0, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#3, [[ER]]#2, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#5, [[ER]]#4, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#7, [[ER]]#6, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#9, [[ER]]#8, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#11, [[ER]]#10, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#13, [[ER]]#12, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#15, [[ER]]#14, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#17, [[ER]]#16, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#19, [[ER]]#18, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#21, [[ER]]#20, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#23, [[ER]]#22, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#25, [[ER]]#24, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#27, [[ER]]#26, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#29, [[ER]]#28, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#31, [[ER]]#30, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#33, [[ER]]#32, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#35, [[ER]]#34, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#37, [[ER]]#36, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#39, [[ER]]#38, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#41, [[ER]]#40, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#43, [[ER]]#42, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#45, [[ER]]#44, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#47, [[ER]]#46, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#49, [[ER]]#48, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#51, [[ER]]#50, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#53, [[ER]]#52, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#55, [[ER]]#54, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#57, [[ER]]#56, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#59, [[ER]]#58, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#61, [[ER]]#60, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#63, [[ER]]#62, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#65, [[ER]]#64, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#67, [[ER]]#66, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  // CHECK: [[TMP:%.+]] = comb.mux bin [[ER]]#69, [[ER]]#68, {{%.+}}
  // CHECK: seq.firreg [[TMP]] clock [[CLK]] reset async [[RST]], %c0_i15
  llhd.drv %mem0, %1#0 after %0 if %1#1 : i15
  llhd.drv %mem1, %1#2 after %0 if %1#3 : i15
  llhd.drv %mem2, %1#4 after %0 if %1#5 : i15
  llhd.drv %mem3, %1#6 after %0 if %1#7 : i15
  llhd.drv %mem4, %1#8 after %0 if %1#9 : i15
  llhd.drv %mem5, %1#10 after %0 if %1#11 : i15
  llhd.drv %mem6, %1#12 after %0 if %1#13 : i15
  llhd.drv %mem7, %1#14 after %0 if %1#15 : i15
  llhd.drv %mem8, %1#16 after %0 if %1#17 : i15
  llhd.drv %mem9, %1#18 after %0 if %1#19 : i15
  llhd.drv %mem10, %1#20 after %0 if %1#21 : i15
  llhd.drv %mem11, %1#22 after %0 if %1#23 : i15
  llhd.drv %mem12, %1#24 after %0 if %1#25 : i15
  llhd.drv %mem13, %1#26 after %0 if %1#27 : i15
  llhd.drv %mem14, %1#28 after %0 if %1#29 : i15
  llhd.drv %mem15, %1#30 after %0 if %1#31 : i15
  llhd.drv %mem16, %1#32 after %0 if %1#33 : i15
  llhd.drv %mem17, %1#34 after %0 if %1#35 : i15
  llhd.drv %mem18, %1#36 after %0 if %1#37 : i15
  llhd.drv %mem19, %1#38 after %0 if %1#39 : i15
  llhd.drv %mem20, %1#40 after %0 if %1#41 : i15
  llhd.drv %mem21, %1#42 after %0 if %1#43 : i15
  llhd.drv %mem22, %1#44 after %0 if %1#45 : i15
  llhd.drv %mem23, %1#46 after %0 if %1#47 : i15
  llhd.drv %mem24, %1#48 after %0 if %1#49 : i15
  llhd.drv %mem25, %1#50 after %0 if %1#51 : i15
  llhd.drv %mem26, %1#52 after %0 if %1#53 : i15
  llhd.drv %mem27, %1#54 after %0 if %1#55 : i15
  llhd.drv %mem28, %1#56 after %0 if %1#57 : i15
  llhd.drv %mem29, %1#58 after %0 if %1#59 : i15
  llhd.drv %mem30, %1#60 after %0 if %1#61 : i15
  llhd.drv %mem31, %1#62 after %0 if %1#63 : i15
  llhd.drv %mem32, %1#64 after %0 if %1#65 : i15
  llhd.drv %mem33, %1#66 after %0 if %1#67 : i15
  llhd.drv %mem34, %1#68 after %0 if %1#69 : i15
  hw.output
}

// See https://github.com/llvm/circt/issues/8512
// CHECK-LABEL: @OpOnConstantInputsMistakenlyPoison(
hw.module @OpOnConstantInputsMistakenlyPoison(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.firreg {{%.+}} clock [[CLK]] : i42
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%3: i42, %4: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock : i1), ^bb2(%clock : i1)
  ^bb2(%5: i1):
    %6 = comb.xor bin %5, %true : i1
    %7 = comb.and bin %6, %clock : i1
    cf.cond_br %7, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    %8 = comb.icmp ne %false, %false : i1
    %9 = comb.icmp eq %true, %true : i1
    %10 = comb.xor %8, %9 : i1
    cf.cond_br %10, ^bb1(%d, %true : i42, i1), ^bb1(%c0_i42, %false : i42, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : i42
}
