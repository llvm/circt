// RUN: circt-opt --sink-clock-gates %s | circt-opt | FileCheck %s

// CHECK-LABEL: hw.module @ClockGate
hw.module @ClockGate(in %clock: !seq.clock, in %enable: i1, in %enable2: i1, in %test_enable: i1) {
  // CHECK-NOT: seq.clock_gate %clock, %enable
  // CHECK-NOT: seq.clock_gate %clock, %enable, %test_enable
  // CHECK-NOT: seq.clock_gate %clock, %enable, %test_enable sym @gate_sym
  %cg0 = seq.clock_gate %clock, %enable
  %cg1 = seq.clock_gate %clock, %enable, %test_enable
  %cg2 = seq.clock_gate %clock, %enable2, %test_enable sym @gate_sym
  // CHECK:  %[[V0:.+]] = comb.or %enable, %test_enable : i1
  // CHECK:  %[[V1:.+]] = comb.or %enable2, %test_enable : i1
  // CHECK:  %[[true:.+]] = hw.constant true
  // CHECK:  %.clock = hw.instance "" @ClockMux(cond: %enable: i1, trueClock: %clock: !seq.clock, falseClock: %clock: !seq.clock, falseClock2: %clock: !seq.clock, trueClock_enable_1: %enable: i1, falseClock_enable_2: %[[V0]]: i1, falseClock2_enable_3: %[[true]]: i1)
  %c = hw.instance "" @ClockMux(cond: %enable: i1, trueClock: %cg0: !seq.clock, falseClock: %cg1: !seq.clock, falseClock2: %clock: !seq.clock ) -> (clock: !seq.clock)
  // CHECK:  hw.instance "a" @d1(clk: %clock: !seq.clock, rst: %enable: i1, clk_enable_0: %enable: i1)
  hw.instance "a" @d1(clk: %cg0 : !seq.clock, rst: %enable : i1) -> ()
  // CHECK:  hw.instance "b" @d1(clk: %clock: !seq.clock, rst: %enable: i1, clk_enable_0: %[[V1]]: i1) -> ()
  hw.instance "b" @d1(clk: %cg2 : !seq.clock, rst: %enable : i1) -> ()
  // CHECK:  %[[true_0:.+]] = hw.constant true
  // CHECK:  hw.instance "c" @d2(clk: %clock: !seq.clock, rst: %enable: i1, clk_enable_0: %[[true_0]]: i1) -> ()
  hw.instance "c" @d2(clk: %clock : !seq.clock, rst: %enable : i1) -> ()
}


// CHECK-LABEL: hw.module private @d1
// CHECK-SAME: (in %clk : !seq.clock, in %rst : i1, in %[[clk_enable_0:.+]] : i1)
hw.module private @d1(in %clk : !seq.clock, in %rst : i1) {
  // CHECK:  seq.clock_gate %clk, %[[clk_enable_0]]
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4xi32>

  %c0_i2 = hw.constant 0 : i2
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  %myMemory_rdata = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 0} : !seq.hlmem<4xi32>

  seq.write %myMemory[%c0_i2] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4xi32>
  // CHECK:  hw.instance "" @d2(clk: %clk: !seq.clock, rst: %rst: i1, clk_enable_0: %[[clk_enable_0]]: i1)
  hw.instance "" @d2(clk: %clk : !seq.clock, rst: %rst : i1) -> ()

  // CHECK:  %.clock = hw.instance "" @ClockMux(cond: %true: i1, trueClock: %clk: !seq.clock, falseClock: %clk: !seq.clock, falseClock2: %clk: !seq.clock, trueClock_enable_1: %[[clk_enable_0]]: i1, falseClock_enable_2: %[[clk_enable_0]]: i1, falseClock2_enable_3: %[[clk_enable_0]]: i1)
  %c = hw.instance "" @ClockMux(cond: %c1_i1: i1, trueClock: %clk: !seq.clock, falseClock: %clk: !seq.clock, falseClock2: %clk: !seq.clock ) -> (clock: !seq.clock)
  hw.output
}

// CHECK-LABEL: hw.module private @d2
// CHECK-SAME: (in %clk : !seq.clock, in %rst : i1, in %[[clk_enable_0:.+]] : i1)
hw.module private @d2(in %clk : !seq.clock, in %rst : i1) {
  // CHECK:  seq.clock_gate %clk, %[[clk_enable_0]]
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4x8xi32>

  %c0_i2 = hw.constant 0 : i2
  %c0_i3 = hw.constant 0 : i3
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32
  hw.instance "" @d0(clk: %clk : !seq.clock, rst: %rst : i1) -> ()

  // CHECK: %myMemory_rdata = seq.read %myMemory[%c0_i2, %c0_i3] {latency = 0 : i64} : !seq.hlmem<4x8xi32>
  %myMemory_rdata = seq.read %myMemory[%c0_i2, %c0_i3] { latency = 0} : !seq.hlmem<4x8xi32>

  // CHECK: seq.write %myMemory[%c0_i2, %c0_i3] %c42_i32 wren %true {latency = 1 : i64} : !seq.hlmem<4x8xi32>
  seq.write %myMemory[%c0_i2, %c0_i3] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4x8xi32>
  %c = hw.instance "" @ClockMux(cond: %rst: i1, trueClock: %clk: !seq.clock, falseClock: %clk: !seq.clock, falseClock2: %clk: !seq.clock ) -> (clock: !seq.clock)
  hw.output
}

// CHECK-LABEL: hw.module private @d0
// CHECK-SAME: (in %clk : !seq.clock, in %rst : i1, in %[[clk_enable_0:.+]] : i1)
hw.module private @d0(in %clk : !seq.clock, in %rst : i1) {
  // CHECK:  seq.clock_gate %clk, %[[clk_enable_0]]
  %myMemory = seq.hlmem @myMemory %clk, %rst : <1xi32>

  %c0_i0 = hw.constant 0 : i0
  %c0_i3 = hw.constant 0 : i3
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  // CHECK:  %reg = seq.compreg.ce %true, %clk, %[[clk_enable_0]] : i1  
  %reg = seq.compreg %c1_i1, %clk : i1
  %myMemory_rdata = seq.read %myMemory[%c0_i0] rden %reg { latency = 0} : !seq.hlmem<1xi32>
  hw.output
}

// CHECK-LABEL: hw.module private @ClockMux
// CHECK-SAME: (in %cond : i1, in %trueClock : !seq.clock, in %falseClock : !seq.clock, in %falseClock2 : !seq.clock, out clock : !seq.clock, in %[[trueClock_enable_1:.+]] : i1, in %[[falseClock_enable_2:.+]] : i1, in %[[falseClock2_enable_3:.+]] : i1)
hw.module private @ClockMux(in %cond: i1, in %trueClock: !seq.clock, in %falseClock: !seq.clock,in %falseClock2: !seq.clock, out clock: !seq.clock) {
  // CHECK-DAG: seq.clock_gate %trueClock, %[[trueClock_enable_1]]
  // CHECK-DAG: seq.clock_gate %falseClock, %[[falseClock_enable_2]]
  // CHECK-DAG: seq.clock_gate %falseClock2, %[[falseClock2_enable_3]]
  %clock = seq.clock_mux %cond, %trueClock, %falseClock
  %clock2 = seq.clock_mux %cond, %trueClock, %falseClock2
  %c42_i32 = hw.constant 42 : i32
  hw.instance "ckg" @fifo1(clk: %trueClock : !seq.clock, rst: %cond : i1, in: %c42_i32 : i32, rdEn: %cond : i1, wrEn : %cond: i1) -> ()
  hw.output %clock : !seq.clock
}

hw.module private @fifo1(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1) {
  // CHECK:  seq.clock_gate %clk, %[[clk_enable_0]]
  %out, %full, %empty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}
