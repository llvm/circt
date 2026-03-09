// RUN: circt-opt -canonicalize %s | FileCheck %s

hw.module.extern @Observe(in %x : i32)

// CHECK-LABEL: @FirReg
hw.module @FirReg(in %clk : !seq.clock, in %in : i32) {
  %false = hw.constant false
  %true = hw.constant true

  // Registers that update to themselves should be replaced with a constant 0.
  %reg0 = seq.firreg %reg0 clock %clk : i32
  hw.instance "reg0" @Observe(x: %reg0: i32) -> ()
  // CHECK: hw.instance "reg0" @Observe(x: %c0_i32: i32) -> ()

  %clk_true = seq.to_clock %true
  %clk_false = seq.to_clock %false

  // Registers that are never clocked should be replaced with a constant 0.
  %reg1a = seq.firreg %in clock %clk_false : i32
  %reg1b = seq.firreg %in clock %clk_true : i32
  hw.instance "reg1a" @Observe(x: %reg1a: i32) -> ()
  hw.instance "reg1b" @Observe(x: %reg1b: i32) -> ()
  // CHECK: hw.instance "reg1a" @Observe(x: %c0_i32: i32) -> ()
  // CHECK: hw.instance "reg1b" @Observe(x: %c0_i32: i32) -> ()
}

// Should not optimize away the register if it has a symbol.
// CHECK-LABEL: @FirRegSymbol
hw.module @FirRegSymbol(in %clk : !seq.clock, out out : i32) {
  // CHECK: %reg = seq.firreg %reg clock %clk sym @reg : i32
  // CHECK: hw.output %reg : i32
  %reg = seq.firreg %reg clock %clk sym @reg : i32
  hw.output %reg : i32
}

// CHECK-LABEL: @FirRegReset
hw.module @FirRegReset(in %clk : !seq.clock, in %in : i32, in %r : i1, in %v : i32) {
  %c3_i32 = hw.constant 3 : i32
  %false = hw.constant false
  %true = hw.constant true

  // Registers that update to themselves should be replaced with their reset
  // value only when a reset value is constant.
  %reg0a = seq.firreg %reg0a clock %clk reset sync %r, %v : i32
  %reg0b = seq.firreg %reg0b clock %clk reset sync %r, %c0_i32 : i32
  hw.instance "reg0a" @Observe(x: %reg0a: i32) -> ()
  hw.instance "reg0b" @Observe(x: %reg0b: i32) -> ()
  // CHECK: hw.instance "reg0a" @Observe(x: %reg0a: i32) -> ()
  // CHECK-NEXT: hw.instance "reg0b" @Observe(x: %c0_i32: i32) -> ()

  // Registers that never reset should drop their reset value.
  %reg1 = seq.firreg %in clock %clk reset sync %false, %v : i32
  hw.instance "reg1" @Observe(x: %reg1: i32) -> ()
  // CHECK: %reg1 = seq.firreg %in clock %clk : i32
  // CHECK: hw.instance "reg1" @Observe(x: %reg1: i32) -> ()

  // Registers that are permanently reset should be replaced with their reset
  // value.
  %reg2a = seq.firreg %in clock %clk reset sync %true, %v : i32
  %reg2b = seq.firreg %in clock %clk reset async %true, %v : i32
  hw.instance "reg2a" @Observe(x: %reg2a: i32) -> ()
  hw.instance "reg2b" @Observe(x: %reg2b: i32) -> ()
  // CHECK: hw.instance "reg2a" @Observe(x: %v: i32) -> ()
  // CHECK: hw.instance "reg2b" @Observe(x: %v: i32) -> ()

  // Registers that are never clocked should be replaced with their reset value.
  %clk_true = seq.to_clock %true
  %clk_false = seq.to_clock %false
  %c0_i32 = hw.constant 0 : i32
  %reg3a = seq.firreg %in clock %clk_false reset sync %r, %v : i32
  %reg3b = seq.firreg %in clock %clk_true reset sync %r, %v : i32
  %reg3c = seq.firreg %in clock %clk_true reset sync %r, %c0_i32 : i32
  hw.instance "reg3a" @Observe(x: %reg3a: i32) -> ()
  hw.instance "reg3b" @Observe(x: %reg3b: i32) -> ()
  hw.instance "reg3c" @Observe(x: %reg3c: i32) -> ()
  // CHECK: hw.instance "reg3a" @Observe(x: %reg3a: i32) -> ()
  // CHECK: hw.instance "reg3b" @Observe(x: %reg3b: i32) -> ()
  // CHECK: hw.instance "reg3c" @Observe(x: %c0_i32: i32) -> ()

  // A register with preset value is not folded right now
  // CHECK: %reg_preset = seq.firreg
  %reg_preset = seq.firreg %reg_preset clock %clk reset sync %r, %c0_i32 preset 3: i32

  // A register with 0 preset value is folded.
  %reg_preset_0 = seq.firreg %reg_preset_0 clock %clk preset 0: i32
  hw.instance "reg_preset_0" @Observe(x: %reg_preset_0: i32) -> ()
  // CHECK-NEXT: hw.instance "reg_preset_0" @Observe(x: %c0_i32: i32) -> ()

  // A register with const false reset and 0 preset value is folded.
  %reg_preset_1 = seq.firreg %reg_preset_1 clock %clk reset sync %false, %c0_i32 preset 0: i32
  // CHECK-NEXT: hw.instance "reg_preset_1" @Observe(x: %c0_i32: i32) -> ()
  hw.instance "reg_preset_1" @Observe(x: %reg_preset_1: i32) -> ()

  // A register with const false reset and 0 preset value is folded.
  %reg_preset_2 = seq.firreg %reg_preset_2 clock %clk reset sync %false, %c0_i32 preset 3: i32
  // CHECK-NEXT: hw.instance "reg_preset_2" @Observe(x: %c3_i32: i32) -> ()
  hw.instance "reg_preset_2" @Observe(x: %reg_preset_2: i32) -> ()

  %reg_preset_3 = seq.firreg %reg_preset_3 clock %clk reset sync %r, %c3_i32 preset 3: i32
  // CHECK-NEXT: hw.instance "reg_preset_3" @Observe(x: %c3_i32: i32) -> ()
  hw.instance "reg_preset_3" @Observe(x: %reg_preset_3: i32) -> ()
}

// CHECK-LABEL: @FirRegAggregate
hw.module @FirRegAggregate(in %clk : !seq.clock, out out : !hw.struct<foo: i32>) {
  // TODO: Use constant aggregate attribute once supported.
  // CHECK:      %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT: %0 = hw.bitcast %c0_i32 : (i32) -> !hw.struct<foo: i32>
  // CHECK-NEXT: hw.output %0
  %reg = seq.firreg %reg clock %clk : !hw.struct<foo: i32>
  hw.output %reg : !hw.struct<foo: i32>
}

// CHECK-LABEL: @UninitializedArrayElement
hw.module @UninitializedArrayElement(in %a : i1, in %clock : !seq.clock, out b: !hw.array<2xi1>) {
  // CHECK:      %false = hw.constant false
  // CHECK-NEXT: %0 = hw.array_create %false, %a : i1
  // CHECK-NEXT: %r = seq.firreg %0 clock %clock : !hw.array<2xi1>
  // CHECK-NEXT: hw.output %r : !hw.array<2xi1>
  %true = hw.constant true
  %r = seq.firreg %1 clock %clock : !hw.array<2xi1>
  %0 = hw.array_get %r[%true] : !hw.array<2xi1>, i1
  %1 = hw.array_create %0, %a : i1
  hw.output %r : !hw.array<2xi1>
}

// CHECK-LABEL: hw.module @ClockGate
hw.module @ClockGate(in %clock : !seq.clock, in %enable : i1, in %enable2 : i1, in %testEnable : i1) {
  %false = hw.constant false
  %true = hw.constant true
  // CHECK-NEXT: [[CLOCK_LOW:%.+]] = seq.const_clock low
  %falseClock = seq.const_clock low

  // CHECK-NEXT: %zeroClock = hw.wire [[CLOCK_LOW]] sym @zeroClock
  %0 = seq.clock_gate %falseClock, %enable
  %zeroClock = hw.wire %0 sym @zeroClock : !seq.clock

  // CHECK-NEXT: %alwaysOff1 = hw.wire [[CLOCK_LOW]] sym @alwaysOff1
  // CHECK-NEXT: %alwaysOff2 = hw.wire [[CLOCK_LOW]] sym @alwaysOff2
  %1 = seq.clock_gate %clock, %false
  %2 = seq.clock_gate %clock, %false, %false
  %alwaysOff1 = hw.wire %1 sym @alwaysOff1 : !seq.clock
  %alwaysOff2 = hw.wire %2 sym @alwaysOff2 : !seq.clock

  // CHECK-NEXT: %alwaysOn1 = hw.wire %clock sym @alwaysOn1
  // CHECK-NEXT: %alwaysOn2 = hw.wire %clock sym @alwaysOn2
  // CHECK-NEXT: %alwaysOn3 = hw.wire %clock sym @alwaysOn3
  %3 = seq.clock_gate %clock, %true
  %4 = seq.clock_gate %clock, %true, %testEnable
  %5 = seq.clock_gate %clock, %enable, %true
  %alwaysOn1 = hw.wire %3 sym @alwaysOn1 : !seq.clock
  %alwaysOn2 = hw.wire %4 sym @alwaysOn2 : !seq.clock
  %alwaysOn3 = hw.wire %5 sym @alwaysOn3 : !seq.clock

  // CHECK-NEXT: [[TMP:%.+]] = seq.clock_gate %clock, %enable
  // CHECK-NEXT: %dropTestEnable = hw.wire [[TMP]] sym @dropTestEnable
  %6 = seq.clock_gate %clock, %enable, %false
  %dropTestEnable = hw.wire %6 sym @dropTestEnable : !seq.clock

  // CHECK-NEXT: [[TCG1:%.+]] = seq.clock_gate %clock, %enable
  // CHECK-NEXT: %transitiveClock1 = hw.wire [[TCG1]] sym @transitiveClock1  : !seq.clock
  %7 = seq.clock_gate %clock, %enable
  %8 = seq.clock_gate %clock, %enable
  %transitiveClock1 = hw.wire %7 sym @transitiveClock1 : !seq.clock

  // CHECK-NEXT: [[TCG2:%.+]] = seq.clock_gate %clock, %enable, %testEnable
  // CHECK-NEXT: [[TCG3:%.+]] = seq.clock_gate [[TCG2]], %enable
  // CHECK-NEXT: %transitiveClock2 = hw.wire [[TCG3]] sym @transitiveClock2  : !seq.clock
  %9 = seq.clock_gate %clock, %enable, %testEnable
  %10 = seq.clock_gate %9, %enable2
  %11 = seq.clock_gate %10, %enable, %testEnable
  %transitiveClock2 = hw.wire %11 sym @transitiveClock2 : !seq.clock
}

// CHECK-LABEL: hw.module @ClockMux
hw.module @ClockMux(in %cond : i1, in %trueClock : !seq.clock, in %falseClock : !seq.clock, out clk0 : !seq.clock, out clk1 : !seq.clock){
  %false = hw.constant false
  %true = hw.constant true
  %clock_true = seq.clock_mux %true, %trueClock, %falseClock
  %clock_false = seq.clock_mux %false, %trueClock, %falseClock
  // CHECK: hw.output %trueClock, %falseClock : !seq.clock, !seq.clock
  hw.output %clock_true, %clock_false : !seq.clock, !seq.clock
}

// CHECK-LABEL: @FirMem
hw.module @FirMem(in %addr : i4, in %clock : !seq.clock, in %data : i42, out out: i42) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i3 = hw.constant 0 : i3
  %c-1_i3 = hw.constant -1 : i3

  // CHECK: [[CLK_FALSE:%.+]] = seq.const_clock low
  // CHECK: [[CLK_TRUE:%.+]] = seq.const_clock high
  %clk_false = seq.to_clock %false
  %clk_true = seq.to_clock %true

  // CHECK: [[MEM:%.+]] = seq.firmem
  %0 = seq.firmem 0, 1, undefined, undefined : <12 x 42, mask 3>

  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %clock :
  %1 = seq.firmem.read_port %0[%addr], clock %clock enable %true : <12 x 42, mask 3>

  // CHECK-NEXT: seq.firmem.write_port [[MEM]][%addr] = %data, clock %clock {w0}
  seq.firmem.write_port %0[%addr] = %data, clock %clock enable %true {w0} : <12 x 42, mask 3>
  // CHECK-NOT: {w1}
  seq.firmem.write_port %0[%addr] = %data, clock %clock enable %false {w1} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.write_port [[MEM]][%addr] = %data, clock %clock {w2}
  seq.firmem.write_port %0[%addr] = %data, clock %clock mask %c-1_i3 {w2} : <12 x 42, mask 3>, i3
  // CHECK-NOT: {w3}
  seq.firmem.write_port %0[%addr] = %data, clock %clock mask %c0_i3 {w3} : <12 x 42, mask 3>, i3
  // CHECK-NOT: {w4}
  seq.firmem.write_port %0[%addr] = %data, clock %clk_true {w4} : <12 x 42, mask 3>
  // CHECK-NOT: {w5}
  seq.firmem.write_port %0[%addr] = %data, clock %clk_false {w5} : <12 x 42, mask 3>

  // CHECK-NEXT: seq.firmem.read_write_port [[MEM]][%addr] = %data if %true, clock %clock {rw0}
  %2 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clock enable %true {rw0} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %clock enable %false {rw1}
  %3 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clock enable %false {rw1} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.read_write_port [[MEM]][%addr] = %data if %true, clock %clock {rw2}
  %4 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clock mask %c-1_i3 {rw2} : <12 x 42, mask 3>, i3
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %clock {rw3}
  %5 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clock mask %c0_i3 {rw3} : <12 x 42, mask 3>, i3
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %clock {rw4}
  %6 = seq.firmem.read_write_port %0[%addr] = %data if %false, clock %clock {rw4} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock [[CLK_TRUE]] {rw5}
  %7 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clk_true {rw5} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock [[CLK_FALSE]] {rw6}
  %8 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clk_false {rw6} : <12 x 42, mask 3>

  // CHECK: %has_symbol = seq.firmem
  %has_symbol = seq.firmem sym @someMem 0, 1, undefined, undefined : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.write_port %has_symbol
  seq.firmem.write_port %has_symbol[%addr] = %data, clock %clock enable %false {w0} : <12 x 42, mask 3>


  // CHECK-NOT: seq.firmem
  %no_readers = seq.firmem 0, 1, undefined, undefined : <12 x 42, mask 3>
  seq.firmem.write_port %no_readers[%addr] = %data, clock %clk_false {w5} : <12 x 42, mask 3>

  %9 = comb.xor %1, %2, %3, %4, %5, %6, %7, %8 : i42
  hw.output %9 : i42
}

// CHECK-LABEL: @through_clock
hw.module @through_clock(in %clock : !seq.clock, out out : !seq.clock) {
  // CHECK: hw.output %clock : !seq.clock
  %tmp = seq.from_clock %clock
  %out = seq.to_clock %tmp
  hw.output %out : !seq.clock
}

// CHECK-LABEL: @through_wire
hw.module @through_wire(in %clock : i1, out out: i1) {
  // CHECK: hw.output %clock : i1
  %tmp = seq.to_clock %clock
  %out = seq.from_clock %tmp
  hw.output %out : i1
}

// CHECK-LABEL: @const_clock
hw.module @const_clock(out clock_true : !seq.clock, out clock_false : !seq.clock) {
  // CHECK: [[CLOCK_TRUE:%.+]] = seq.const_clock high
  // CHECK: [[CLOCK_FALSE:%.+]] = seq.const_clock low

  %true = hw.constant 1 : i1
  %clock_true = seq.to_clock %true

  %false = hw.constant 0 : i1
  %clock_false = seq.to_clock %false

  // CHECK: hw.output [[CLOCK_FALSE]], [[CLOCK_TRUE]]
  hw.output %clock_false, %clock_true : !seq.clock, !seq.clock
}

// CHECK-LABEL: @const_clock_reg
hw.module @const_clock_reg(in %clock : !seq.clock, out r_data : !seq.clock) {
  // CHECK: seq.const_clock low
  %0 = seq.const_clock  low
  %1 = seq.firreg %1 clock %0 : !seq.clock
  hw.output %1 : !seq.clock
}

// CHECK-LABEL: @clock_inv
hw.module @clock_inv(in %clock : !seq.clock, out clock_true : !seq.clock, out clock_false : !seq.clock, out same_clock: !seq.clock) {
  %clk_low = seq.const_clock low
  %clk_high = seq.const_clock high

  %clk_inv_low = seq.clock_inv %clk_low
  %clk_inv_high = seq.clock_inv %clk_high

  %clk_inv = seq.clock_inv %clock
  %clk_orig = seq.clock_inv %clk_inv


  // CHECK: [[CLK_HIGH:%.+]] = seq.const_clock high
  // CHECK: [[CLK_LOW:%.+]] = seq.const_clock low
  // CHECK: hw.output [[CLK_HIGH]], [[CLK_LOW]], %clock
  hw.output %clk_inv_low, %clk_inv_high, %clk_orig : !seq.clock, !seq.clock, !seq.clock

}

// CHECK-LABEL: @CompRegClockEnabled(
hw.module @CompRegClockEnabled(in %clock: !seq.clock, in %d: i42, in %x: i42, in %en: i1, out q0: i42, out q1: i42, out q2 : i42) {
  // CHECK-NEXT: seq.compreg.ce %d, %clock, %en
  // CHECK-NEXT: seq.compreg.ce %d, %clock, %en
  // CHECK-NEXT: seq.compreg %x, %clock
  %0 = comb.mux %en, %d, %x : i42
  %1 = seq.compreg.ce %0, %clock, %en : i42
  %2 = arith.select %en, %d, %x : i42
  %3 = seq.compreg.ce %2, %clock, %en : i42
  %true = hw.constant true
  %4 = seq.compreg.ce %x, %clock, %true : i42
  hw.output %1, %3, %4 : i42, i42, i42
}

// CHECK-LABEL: @FirMemCanonicalization
hw.module @FirMemCanonicalization(in %addr : i4, in %clock : !seq.clock, in %data : i32, out read_data1: i32, out read_data2: i32, out mixed_read: i32) {
  %true = hw.constant true
  %false = hw.constant false

  // Write-only memory should be completely erased
  // CHECK-NOT: %write_only_mem
  // CHECK-NOT: seq.firmem.write_port {{.*}} %write_only_mem
  %write_only_mem = seq.firmem 0, 1, undefined, undefined : <16 x 32>
  seq.firmem.write_port %write_only_mem[%addr] = %data, clock %clock enable %true : <16 x 32>

  // Read-only memory with multiple read ports should be replaced with constant 0
  // CHECK-NOT: %read_only_mem
  // CHECK-NOT: seq.firmem.read_port {{.*}} %read_only_mem
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i32
  %read_only_mem = seq.firmem 0, 1, undefined, undefined : <16 x 32>
  %read_data1 = seq.firmem.read_port %read_only_mem[%addr], clock %clock enable %true : <16 x 32>
  %read_data2 = seq.firmem.read_port %read_only_mem[%addr], clock %clock enable %true : <16 x 32>

  // Memory with symbol should be preserved
  // CHECK: %mem_with_symbol = seq.firmem sym @memSym
  // CHECK: seq.firmem.write_port %mem_with_symbol
  %mem_with_symbol = seq.firmem sym @memSym 0, 1, undefined, undefined : <16 x 32>
  seq.firmem.write_port %mem_with_symbol[%addr] = %data, clock %clock enable %true : <16 x 32>

  // Mixed read/write memory should be preserved
  // CHECK: %mixed_mem = seq.firmem
  // CHECK-NEXT: %[[MIXED_READ:.*]] = seq.firmem.read_port %mixed_mem
  // CHECK-NEXT: seq.firmem.write_port %mixed_mem
  %mixed_mem = seq.firmem 0, 1, undefined, undefined : <16 x 32>
  %mixed_read = seq.firmem.read_port %mixed_mem[%addr], clock %clock enable %true : <16 x 32>
  seq.firmem.write_port %mixed_mem[%addr] = %data, clock %clock enable %true : <16 x 32>

  // Memory with read-write port (read-only case)
  // CHECK-NOT: %rw_read_only_mem
  // CHECK-NOT: seq.firmem.read_write_port {{.*}} %rw_read_only_mem
  %rw_read_only_mem = seq.firmem 0, 1, undefined, undefined : <16 x 32>
  %rw_read_data = seq.firmem.read_write_port %rw_read_only_mem[%addr] = %data if %false, clock %clock enable %true : <16 x 32>

  // CHECK: hw.output [[ZERO]], [[ZERO]], %[[MIXED_READ]] : i32, i32, i32
  hw.output %read_data1, %read_data2, %mixed_read : i32, i32, i32
}

// CHECK-LABEL: @FirRegMuxConstant
hw.module @FirRegMuxConstant(in %clk : !seq.clock, in %cond : i1, in %in : i32) {
  %c42_i32 = hw.constant 42 : i32
  %c0_i32 = hw.constant 0 : i32
  %c5_i32 = hw.constant 5 : i32

  // Register with mux where true branch is register itself and false branch is constant
  // Should be replaced with the constant
  %mux1 = comb.mux %cond, %reg1, %c42_i32 : i32
  %reg1 = seq.firreg %mux1 clock %clk : i32
  hw.instance "reg1" @Observe(x: %reg1: i32) -> ()
  // CHECK: hw.instance "reg1" @Observe(x: %c42_i32: i32) -> ()

  // Register with mux where false branch is register itself and true branch is constant
  // Should be replaced with the constant
  %mux2 = comb.mux %cond, %c0_i32, %reg2 : i32
  %reg2 = seq.firreg %mux2 clock %clk : i32
  hw.instance "reg2" @Observe(x: %reg2: i32) -> ()
  // CHECK: hw.instance "reg2" @Observe(x: %c0_i32: i32) -> ()

  // Register with mux where neither branch is the register itself
  // Should not be optimized
  %mux3 = comb.mux %cond, %in, %c5_i32 : i32
  %reg3 = seq.firreg %mux3 clock %clk : i32
  hw.instance "reg3" @Observe(x: %reg3: i32) -> ()
  // CHECK: %reg3 = seq.firreg
  // CHECK: hw.instance "reg3" @Observe(x: %reg3: i32) -> ()
}

// CHECK-LABEL: @FirRegMuxConstantWithPreset
hw.module @FirRegMuxConstantWithPreset(in %clk : !seq.clock, in %cond : i1) {
  %c42_i32 = hw.constant 42 : i32
  %c0_i32 = hw.constant 0 : i32

  // Register with preset should not be optimized
  %mux1 = comb.mux %cond, %reg1, %c42_i32 : i32
  %reg1 = seq.firreg %mux1 clock %clk preset 42: i32
  // CHECK: hw.instance "reg1" @Observe(x: %reg1: i32) -> ()
  hw.instance "reg1" @Observe(x: %reg1: i32) -> ()
}

// CHECK-LABEL: @FirRegMuxConstantWithReset
hw.module @FirRegMuxConstantWithReset(in %clk : !seq.clock, in %cond : i1, in %rst : i1) {
  %c42_i32 = hw.constant 42 : i32
  %c0_i32 = hw.constant 0 : i32
  %c5_i32 = hw.constant 5 : i32

  // Register with constant reset value that matches the mux constant - should be optimized
  %mux1 = comb.mux %cond, %reg1, %c42_i32 : i32
  %reg1 = seq.firreg %mux1 clock %clk reset sync %rst, %c42_i32 : i32
  hw.instance "reg1" @Observe(x: %reg1: i32) -> ()
  // CHECK: hw.instance "reg1" @Observe(x: %c42_i32: i32) -> ()

  // Register with constant reset value that doesn't match the mux constant - should not be optimized
  %mux2 = comb.mux %cond, %reg2, %c42_i32 : i32
  %reg2 = seq.firreg %mux2 clock %clk reset sync %rst, %c0_i32 : i32
  hw.instance "reg2" @Observe(x: %reg2: i32) -> ()
  // CHECK: %reg2 = seq.firreg
  // CHECK: hw.instance "reg2" @Observe(x: %reg2: i32) -> ()

  // Register with non-constant reset value - should not be optimized
  %mux3 = comb.mux %cond, %reg3, %c42_i32 : i32
  %reg3 = seq.firreg %mux3 clock %clk reset sync %rst, %c5_i32 : i32
  hw.instance "reg3" @Observe(x: %reg3: i32) -> ()
  // CHECK: %reg3 = seq.firreg
  // CHECK: hw.instance "reg3" @Observe(x: %reg3: i32) -> ()

  // Register with async reset - constant reset value that matches - should be optimized
  %mux4 = comb.mux %cond, %reg4, %c0_i32 : i32
  %reg4 = seq.firreg %mux4 clock %clk reset async %rst, %c0_i32 : i32
  hw.instance "reg4" @Observe(x: %reg4: i32) -> ()
  // CHECK: hw.instance "reg4" @Observe(x: %c0_i32: i32) -> ()
}

// CHECK-LABEL: @clock_div
hw.module @clock_div(in %clock : !seq.clock, out div_combined: !seq.clock) {
  // Nested clock dividers should be combined: clock_div(clock_div(clock, 2), 1) -> clock_div(clock, 3)
  %div1 = seq.clock_div %clock by 2
  %div2 = seq.clock_div %div1 by 1
  // CHECK: %[[DIV3:.+]] = seq.clock_div %clock by 3
  // CHECK: hw.output %[[DIV3]]
  hw.output %div2 : !seq.clock
}
