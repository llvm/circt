// RUN: circt-opt --llhd-deseq %s | FileCheck %s

// CHECK-LABEL: @ClockPosEdge(
hw.module @ClockPosEdge(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK]] : i42
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ClockNegEdge(
hw.module @ClockNegEdge(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[CLK_INV:%.+]] = seq.clock_inv [[CLK]]
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK_INV]] : i42
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ClockPosEdgeWithActiveLowReset(
hw.module @ClockPosEdgeWithActiveLowReset(in %clock: i1, in %reset: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[RST_INV:%.+]] = comb.xor %reset, %true
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK]] reset [[RST_INV]], %c42_i42 : i42
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ClockNegEdgeWithActiveHighReset(
hw.module @ClockNegEdgeWithActiveHighReset(in %clock: i1, in %reset: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[CLK_INV:%.+]] = seq.clock_inv [[CLK]]
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK_INV]] reset %reset, %c42_i42 : i42
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ClockWithEnable(
hw.module @ClockWithEnable(in %clock: i1, in %d: i42, in %en: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg.ce %d, [[CLK]], %en : i42
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
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ClockWithEnableAndReset(
hw.module @ClockWithEnableAndReset(in %clock: i1, in %reset: i1, in %d: i42, in %en: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg.ce %d, [[CLK]], %en reset %reset, %c42_i42 : i42
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
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ChasePastValuesThroughControlFlow(
hw.module @ChasePastValuesThroughControlFlow(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: seq.compreg
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @AcceptMuxForReset(
hw.module @AcceptMuxForReset(in %clock: i1, in %reset: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK]] reset %reset, %c42_i42 : i42
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ComplexControlFlow(
hw.module @ComplexControlFlow(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[ER:%.+]]:2 = scf.execute_region -> (i42, i1) {
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
  // CHECK:   scf.yield [[RESULT]], [[ENABLE]] : i42, i1
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
  // CHECK: [[REG:%.+]] = seq.compreg.ce [[ER]]#0, [[CLK]], [[ER]]#1 : i42
  // CHECK: llhd.drv {{%.+}}, [[REG]] after {{%.+}} :
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ClockAndResetSameConst(
hw.module @ClockAndResetSameConst(in %clock: i1, in %reset: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg %c42_i42, [[CLK]] reset %reset, %c42_i42 : i42
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}

// CHECK-LABEL: @ClockAndResetDifferentConst(
hw.module @ClockAndResetDifferentConst(in %clock: i1, in %reset: i1) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg %c42_i42, [[CLK]] reset %reset, %c0_i42 : i42
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
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
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
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK]] reset %reset, [[FIELD]] : i42
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
  llhd.drv %5, %3 after %0 if %4 : !hw.inout<i42>
}
