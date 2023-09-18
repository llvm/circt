// RUN: circt-opt %s | circt-opt | FileCheck %s

hw.module @d1(%clk : i1, %rst : i1) -> () {
// CHECK: %myMemory = seq.hlmem @myMemory %clk, %rst : <4xi32>
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4xi32>

  %c0_i2 = hw.constant 0 : i2
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  // CHECK: %myMemory_rdata = seq.read %myMemory[%c0_i2] rden %true {latency = 0 : i64} : !seq.hlmem<4xi32>
  %myMemory_rdata = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 0} : !seq.hlmem<4xi32>

  // CHECK: seq.write %myMemory[%c0_i2] %c42_i32 wren %true {latency = 1 : i64} : !seq.hlmem<4xi32>
  seq.write %myMemory[%c0_i2] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4xi32>
  hw.output
}

hw.module @d2(%clk : i1, %rst : i1) -> () {
// CHECK: %myMemory = seq.hlmem @myMemory %clk, %rst : <4x8xi32>
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4x8xi32>

  %c0_i2 = hw.constant 0 : i2
  %c0_i3 = hw.constant 0 : i3
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  // CHECK: %myMemory_rdata = seq.read %myMemory[%c0_i2, %c0_i3] {latency = 0 : i64} : !seq.hlmem<4x8xi32>
  %myMemory_rdata = seq.read %myMemory[%c0_i2, %c0_i3] { latency = 0} : !seq.hlmem<4x8xi32>

  // CHECK: seq.write %myMemory[%c0_i2, %c0_i3] %c42_i32 wren %true {latency = 1 : i64} : !seq.hlmem<4x8xi32>
  seq.write %myMemory[%c0_i2, %c0_i3] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4x8xi32>
  hw.output
}

hw.module @d0(%clk : i1, %rst : i1) -> () {
// CHECK: %myMemory = seq.hlmem @myMemory %clk, %rst : <1xi32>
  %myMemory = seq.hlmem @myMemory %clk, %rst : <1xi32>

  %c0_i0 = hw.constant 0 : i0
  %c0_i3 = hw.constant 0 : i3
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  // CHECK: %myMemory_rdata = seq.read %myMemory[%c0_i0] rden %true {latency = 0 : i64} : !seq.hlmem<1xi32>
  %myMemory_rdata = seq.read %myMemory[%c0_i0] rden %c1_i1 { latency = 0} : !seq.hlmem<1xi32>
  hw.output
}

// CHECK-LABEL: hw.module @ClockGate
hw.module @ClockGate(%clock: !seq.clock, %enable: i1, %test_enable: i1) {
  // CHECK-NEXT: seq.clock_gate %clock, %enable
  // CHECK-NEXT: seq.clock_gate %clock, %enable, %test_enable
  // CHECK-NEXT: seq.clock_gate %clock, %enable, %test_enable sym @gate_sym
  %cg0 = seq.clock_gate %clock, %enable
  %cg1 = seq.clock_gate %clock, %enable, %test_enable
  %cg2 = seq.clock_gate %clock, %enable, %test_enable sym @gate_sym
}

// CHECK-LABEL: hw.module @ClockMux
hw.module @ClockMux(%cond: i1, %trueClock: !seq.clock, %falseClock: !seq.clock) -> (clock: !seq.clock) {
  %clock = seq.clock_mux %cond, %trueClock, %falseClock
  hw.output %clock : !seq.clock
}

hw.module @fifo1(%clk : i1, %rst : i1, %in : i32, %rdEn : i1, %wrEn : i1) -> () {
  // CHECK: %out, %full, %empty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  %out, %full, %empty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}

hw.module @fifo2(%clk : i1, %rst : i1, %in : i32, %rdEn : i1, %wrEn : i1) -> () {
  // CHECK: %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 3 almost_full 2 almost_empty 1 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 3 almost_full 2 almost_empty 1 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}


hw.module @preset(%clock : !seq.clock, %reset : i1, %next : i32) -> () {
  // CHECK: %reg = seq.firreg %next clock %clock preset 0 : i32
  %reg = seq.firreg %next clock %clock preset 0 : i32
}

hw.module @clock_dividers(%clock: !seq.clock) -> () {
  // CHECK: seq.clock_div %clock by 1
  %by_2 = seq.clock_div %clock by 1
}

hw.module @clock_const() -> () {
  // CHECK: seq.const_clock high
  %high = seq.const_clock high
  // CHECK: seq.const_clock low
  %low = seq.const_clock low
}