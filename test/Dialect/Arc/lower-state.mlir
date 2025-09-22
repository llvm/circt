// RUN: circt-opt %s --arc-lower-state | FileCheck %s

func.func private @VoidFunc()
func.func private @RandomI42() -> i42
func.func private @ConsumeI42(i42)
func.func private @IdI42(i42) -> i42

arc.define @Not(%arg0: !seq.clock) -> !seq.clock {
  %0 = seq.clock_inv %arg0
  arc.output %0 : !seq.clock
}

arc.define @IdI42Arc(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

arc.define @IdI2AndI42Arc(%arg0: i2, %arg1: i42) -> (i2, i42) {
  arc.output %arg0, %arg1 : i2, i42
}

arc.define @IdI2AndI42AndI1Arc(%arg0: i2, %arg1: i42, %arg2: i1) -> (i2, i42, i1) {
  arc.output %arg0, %arg1, %arg2 : i2, i42, i1
}

arc.define @IdI2AndI42AndI1AndI42Arc(%arg0: i2, %arg1: i42, %arg2: i1, %arg3: i42) -> (i2, i42, i1, i42) {
  arc.output %arg0, %arg1, %arg2, %arg3 : i2, i42, i1, i42
}

arc.define @RandomI42Arc() -> i42 {
  %0 = hw.constant 42 : i42
  arc.output %0 : i42
}

arc.define @RandomI42AndI19Arc() -> (i42, i19) {
  %0 = hw.constant 42 : i42
  %1 = hw.constant 1337 : i19
  arc.output %0, %1 : i42, i19
}

// CHECK-LABEL: arc.model @Empty
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage):
// CHECK-NEXT:  }
hw.module @Empty() {}

// CHECK-LABEL: arc.model @InputToOutput
hw.module @InputToOutput(in %a: i42, out b: i42) {
  // CHECK:      [[TMP1:%.+]] = arc.state_read %in_a
  // CHECK-NEXT: [[TMP2:%.+]] = comb.xor [[TMP1]]
  // CHECK-NEXT: [[TMP3:%.+]] = func.call @IdI42([[TMP2]])
  %2 = comb.xor %1 : i42
  %1 = func.call @IdI42(%0) : (i42) -> i42
  %0 = comb.xor %a : i42
  // CHECK-NEXT: func.call @ConsumeI42([[TMP2]])
  func.call @ConsumeI42(%0) : (i42) -> ()
  // CHECK-NEXT: [[TMP4:%.+]] = comb.xor [[TMP3]]
  // CHECK-NEXT: arc.state_write %out_b = [[TMP4]]
  hw.output %2 : i42
}

// CHECK-LABEL: arc.model @ReadsBeforeUpdate
hw.module @ReadsBeforeUpdate(in %clock: !seq.clock, in %a: i42, out b: i42) {
  // CHECK: [[Q1:%.+]] = arc.alloc_state %arg0 {name = "q1"}
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK:      scf.if {{%.+}} {
  // CHECK-NEXT:   [[TMP1:%.+]] = arc.state_read [[Q0]]
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.call @IdI42Arc([[TMP1]])
  // CHECK-NEXT:   arc.state_write [[Q1]] = [[TMP2]]
  // CHECK-NEXT:   [[TMP1:%.+]] = arc.state_read %in_a
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.call @IdI42Arc([[TMP1]])
  // CHECK-NEXT:   arc.state_write [[Q0]] = [[TMP2]]
  // CHECK-NEXT: }
  %q1 = arc.state @IdI42Arc(%q0) clock %clock latency 1 {names = ["q1"]} : (i42) -> i42
  %q0 = arc.state @IdI42Arc(%a) clock %clock latency 1 {names = ["q0"]} : (i42) -> i42
  // CHECK-NEXT: [[TMP:%.+]] = arc.state_read [[Q1]]
  // CHECK-NEXT: arc.state_write %out_b = [[TMP]]
  hw.output %q1 : i42
}

// CHECK-LABEL: arc.model @ReadsAfterUpdate
hw.module @ReadsAfterUpdate(in %clock: !seq.clock, in %a: i42, out b: i42) {
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK: [[Q1:%.+]] = arc.alloc_state %arg0 {name = "q1"}
  // CHECK: [[Q0_OLD:%.+]] = arc.state_read [[Q0]]
  // CHECK:      scf.if {{%.+}} {
  // CHECK-NEXT:   [[TMP1:%.+]] = arc.state_read %in_a
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.call @IdI42Arc([[TMP1]])
  // CHECK-NEXT:   arc.state_write [[Q0]] = [[TMP2]]
  // CHECK-NEXT:   [[TMP:%.+]] = arc.call @IdI42Arc([[Q0_OLD]])
  // CHECK-NEXT:   arc.state_write [[Q1]] = [[TMP]]
  // CHECK-NEXT: }
  %q0 = arc.state @IdI42Arc(%a) clock %clock latency 1 {names = ["q0"]} : (i42) -> i42
  %q1 = arc.state @IdI42Arc(%q0) clock %clock latency 1 {names = ["q1"]} : (i42) -> i42
  // CHECK-NEXT: [[TMP:%.+]] = arc.state_read [[Q1]]
  // CHECK-NEXT: arc.state_write %out_b = [[TMP]]
  hw.output %q1 : i42
}

// CHECK-LABEL: arc.model @ClockDivBy4
hw.module @ClockDivBy4(in %clock: !seq.clock, out z: !seq.clock) {
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK: [[Q1:%.+]] = arc.alloc_state %arg0 {name = "q1"}
  // CHECK: [[TMP1:%.+]] = arc.state_read %in_clock
  // CHECK: [[TMP2:%.+]] = seq.from_clock [[TMP1]]
  // CHECK: [[CLOCK_EDGE:%.+]] = comb.and {{%.+}}, [[TMP2]]
  // CHECK: scf.if [[CLOCK_EDGE]] {
  // CHECK:   arc.state_write [[Q0]]
  // CHECK: }
  %q0 = arc.state @Not(%q0) clock %clock latency 1 {names = ["q0"]} : (!seq.clock) -> !seq.clock
  // CHECK: [[TMP1:%.+]] = arc.state_read [[Q0]]
  // CHECK: [[TMP2:%.+]] = seq.from_clock [[TMP1]]
  // CHECK: [[Q0_EDGE:%.+]] = comb.and {{%.+}}, [[TMP2]]
  // CHECK: scf.if [[Q0_EDGE]] {
  // CHECK:   arc.state_write [[Q1]]
  // CHECK: }
  %q1 = arc.state @Not(%q1) clock %q0 latency 1 {names = ["q1"]} : (!seq.clock) -> !seq.clock
  // CHECK: [[Q1_NEW:%.+]] = arc.state_read [[Q1]]
  // CHECK: arc.state_write %out_z = [[Q1_NEW]]
  hw.output %q1 : !seq.clock
}

// CHECK-LABEL: arc.model @EnablePort
hw.module @EnablePort(in %clock: !seq.clock, in %a: i42, in %en: i1) {
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[EN:%.+]] = arc.state_read %in_en
  // CHECK:   scf.if [[EN]] {
  // CHECK:     arc.state_write
  // CHECK:     arc.state_write
  // CHECK:   }
  // CHECK: }
  %q0 = arc.state @IdI42Arc(%a) clock %clock enable %en latency 1 : (i42) -> i42
  %q1 = arc.state @IdI42Arc(%q0) clock %clock enable %en latency 1 : (i42) -> i42
}

// CHECK-LABEL: arc.model @EnableLocal
hw.module @EnableLocal(in %clock: !seq.clock, in %a: i42) {
  // CHECK: [[TMP1:%.+]] = hw.constant 9001
  // CHECK: [[TMP2:%.+]] = arc.state_read %in_a
  // CHECK: [[TMP3:%.+]] = comb.icmp ne [[TMP2]], [[TMP1]]
  // CHECK: scf.if {{%.+}} {
  // CHECK:   scf.if [[TMP3]] {
  // CHECK:     arc.state_write
  // CHECK:     arc.state_write
  // CHECK:   }
  // CHECK: }
  %0 = hw.constant 9001 : i42
  %1 = comb.icmp ne %a, %0 : i42
  %q0 = arc.state @IdI42Arc(%a) clock %clock enable %1 latency 1 : (i42) -> i42
  %q1 = arc.state @IdI42Arc(%q0) clock %clock enable %1 latency 1 : (i42) -> i42
}

// CHECK-LABEL: arc.model @Reset
hw.module @Reset(in %clock: !seq.clock, in %a: i42, in %reset: i1) {
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[RESET:%.+]] = arc.state_read %in_reset
  // CHECK:   scf.if [[RESET]] {
  // CHECK:     arc.state_write
  // CHECK:     arc.state_write
  // CHECK:   }
  // CHECK: }
  %q0 = arc.state @IdI42Arc(%a) clock %clock reset %reset latency 1 : (i42) -> i42
  %q1 = arc.state @IdI42Arc(%q0) clock %clock reset %reset latency 1 : (i42) -> i42
}

// CHECK-LABEL: arc.model @ResetLocal
hw.module @ResetLocal(in %clock: !seq.clock, in %a: i42) {
  // CHECK: [[TMP1:%.+]] = hw.constant 9001
  // CHECK: [[TMP2:%.+]] = arc.state_read %in_a
  // CHECK: [[TMP3:%.+]] = comb.icmp ne [[TMP2]], [[TMP1]]
  // CHECK: scf.if {{%.+}} {
  // CHECK:   scf.if [[TMP3]] {
  // CHECK:     arc.state_write
  // CHECK:     arc.state_write
  // CHECK:   }
  // CHECK: }
  %0 = hw.constant 9001 : i42
  %1 = comb.icmp ne %a, %0 : i42
  %q0 = arc.state @IdI42Arc(%a) clock %clock reset %1 latency 1 : (i42) -> i42
  %q1 = arc.state @IdI42Arc(%q0) clock %clock reset %1 latency 1 : (i42) -> i42
}

// CHECK-LABEL: arc.model @ResetAndEnable
hw.module @ResetAndEnable(in %clock: !seq.clock, in %a: i42, in %reset: i1, in %en: i1) {
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK: [[Q1:%.+]] = arc.alloc_state %arg0 {name = "q1"}
  // CHECK: [[Q2:%.+]] = arc.alloc_state %arg0 {name = "q2"}
  // CHECK: [[Q3:%.+]] = arc.alloc_state %arg0 {name = "q3"}
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[EN:%.+]] = arc.state_read %in_en
  // CHECK:   scf.if [[EN]] {
  // CHECK:     arc.state_write [[Q0]]
  // CHECK:   }
  // CHECK:   [[RESET:%.+]] = arc.state_read %in_reset
  // CHECK:   scf.if [[RESET]] {
  // CHECK:     [[TMP:%.+]] = hw.constant 0
  // CHECK:     arc.state_write [[Q1]] = [[TMP]]
  // CHECK:     [[TMP:%.+]] = hw.constant 0
  // CHECK:     arc.state_write [[Q2]] = [[TMP]]
  // CHECK:     [[TMP:%.+]] = hw.constant 0
  // CHECK:     arc.state_write [[Q3]] = [[TMP]]
  // CHECK:   } else {
  // CHECK:     [[EN:%.+]] = arc.state_read %in_en
  // CHECK:     scf.if [[EN]] {
  // CHECK:       arc.state_write [[Q1]]
  // CHECK:       arc.state_write [[Q2]]
  // CHECK:     }
  // CHECK:     arc.state_write [[Q3]]
  // CHECK:   }
  // CHECK: }
  %q0 = arc.state @IdI42Arc(%a) clock %clock enable %en latency 1 {names = ["q0"]} : (i42) -> i42
  %q1 = arc.state @IdI42Arc(%q0) clock %clock enable %en reset %reset latency 1 {names = ["q1"]} : (i42) -> i42
  %q2 = arc.state @IdI42Arc(%q1) clock %clock enable %en reset %reset latency 1 {names = ["q2"]} : (i42) -> i42
  %q3 = arc.state @IdI42Arc(%q2) clock %clock reset %reset latency 1 {names = ["q3"]} : (i42) -> i42
}

// CHECK-LABEL: arc.model @BlackBox
hw.module @BlackBox(in %clock: !seq.clock, in %a: i42, out b: i42) {
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK: [[S:%.+]] = arc.alloc_state %arg0 {name = "ext/s"}
  // CHECK: [[P:%.+]] = arc.alloc_state %arg0 {name = "ext/p"}
  // CHECK: [[Q:%.+]] = arc.alloc_state %arg0 {name = "ext/q"}
  // CHECK: [[R:%.+]] = arc.alloc_state %arg0 {name = "ext/r"}
  // CHECK: [[Q1:%.+]] = arc.alloc_state %arg0 {name = "q1"}
  // CHECK: scf.if {{%.+}} {
  // CHECK:   arc.state_write [[Q0]]
  // CHECK: }
  %0 = arc.state @IdI42Arc(%a) clock %clock latency 1 {names = ["q0"]} : (i42) -> i42
  // CHECK: [[Q0_NEW:%.+]] = arc.state_read [[Q0]]
  // CHECK: [[S_NEW:%.+]] = arc.state_read [[S]]
  // CHECK: [[TMP:%.+]] = comb.and [[Q0_NEW]], [[S_NEW]]
  // CHECK: [[Q0_NEW:%.+]] = arc.state_read [[Q0]]
  // CHECK: arc.state_write [[P]] = [[Q0_NEW]]
  // CHECK: arc.state_write [[Q]] = [[TMP]]
  %1 = comb.and %0, %3 : i42
  %2, %3 = hw.instance "ext" @BlackBoxExt(p: %0: i42, q: %1: i42) -> (r: i42, s: i42)
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[S_NEW:%.+]] = arc.state_read [[S]]
  // CHECK:   [[TMP:%.+]] = arc.call @IdI42Arc([[S_NEW]])
  // CHECK:   arc.state_write [[Q1]] = [[TMP]]
  // CHECK: }
  %4 = arc.state @IdI42Arc(%3) clock %clock latency 1 {names = ["q1"]} : (i42) -> i42
  // CHECK: [[R_NEW:%.+]] = arc.state_read [[R]]
  // CHECK: [[Q1_NEW:%.+]] = arc.state_read [[Q1]]
  // CHECK: [[TMP:%.+]] = comb.or [[R_NEW]], [[Q1_NEW]]
  // CHECK: arc.state_write %out_b = [[TMP]]
  %5 = comb.or %2, %4 : i42
  hw.output %5 : i42
}

hw.module.extern private @BlackBoxExt(in %p: i42, in %q: i42, out r: i42, out s: i42)

// CHECK-LABEL: arc.model @MemoryInputToOutput
hw.module @MemoryInputToOutput(in %clock: !seq.clock, in %a: i2, in %b: i42, out c: i42) {
  // CHECK: [[MEM:%.+]] = arc.alloc_memory
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[A:%.+]] = arc.state_read %in_a
  // CHECK:   [[B:%.+]] = arc.state_read %in_b
  // CHECK:   [[TMP:%.+]]:2 = arc.call @IdI2AndI42Arc([[A]], [[B]])
  // CHECK:   arc.memory_write [[MEM]][[[TMP]]#0], [[TMP]]#1
  // CHECK: }
  %mem = arc.memory <4 x i42, i2>
  %0 = arc.memory_read_port %mem[%a] : <4 x i42, i2>
  arc.memory_write_port %mem, @IdI2AndI42Arc(%a, %b) clock %clock latency 1 : <4 x i42, i2>, i2, i42
  // CHECK: [[A:%.+]] = arc.state_read %in_a
  // CHECK: [[TMP:%.+]] = arc.memory_read [[MEM]][[[A]]]
  // CHECK: arc.state_write %out_c = [[TMP]]
  hw.output %0 : i42
}

// CHECK-LABEL: arc.model @MemoryReadBeforeUpdate
hw.module @MemoryReadBeforeUpdate(in %clock: !seq.clock, in %a: i2, in %b: i42, in %en: i1, out c: i42) {
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK: [[MEM:%.+]] = arc.alloc_memory
  // Port ops are ignored. Writes are lowered with `arc.memory`, reads when they
  // are used by an op.
  %0 = arc.memory_read_port %mem[%a] : <4 x i42, i2>
  arc.memory_write_port %mem, @IdI2AndI42Arc(%a, %b) clock %clock latency 1 : <4 x i42, i2>, i2, i42
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[A:%.+]] = arc.state_read %in_a
  // CHECK:   [[TMP1:%.+]] = arc.memory_read [[MEM]][[[A]]]
  // CHECK:   [[TMP2:%.+]] = arc.call @IdI42Arc([[TMP1]])
  // CHECK:   arc.state_write [[Q0]] = [[TMP2]]
  %q0 = arc.state @IdI42Arc(%0) clock %clock latency 1 {names = ["q0"]} : (i42) -> i42
  // CHECK:   [[A:%.+]] = arc.state_read %in_a
  // CHECK:   [[B:%.+]] = arc.state_read %in_b
  // CHECK:   [[TMP:%.+]]:2 = arc.call @IdI2AndI42Arc([[A]], [[B]])
  // CHECK:   arc.memory_write [[MEM]][[[TMP]]#0], [[TMP]]#1
  %mem = arc.memory <4 x i42, i2>
  // CHECK: }
  // CHECK: [[Q0_NEW:%.+]] = arc.state_read [[Q0]]
  // CHECK: arc.state_write %out_c = [[Q0_NEW]]
  hw.output %q0 : i42
}

// CHECK-LABEL: arc.model @MemoryReadAfterUpdate
hw.module @MemoryReadAfterUpdate(in %clock: !seq.clock, in %a: i2, in %b: i42, in %en: i1, out c: i42) {
  // CHECK: [[MEM:%.+]] = arc.alloc_memory
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // Port ops are ignored. Writes are lowered with `arc.memory`, reads when they
  // are used by an op.
  %0 = arc.memory_read_port %mem[%a] : <4 x i42, i2>
  arc.memory_write_port %mem, @IdI2AndI42Arc(%a, %b) clock %clock latency 1 : <4 x i42, i2>, i2, i42
  // CHECK: [[A:%.+]] = arc.state_read %in_a
  // CHECK: [[READ_OLD:%.+]] = arc.memory_read [[MEM]][[[A]]]
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[A:%.+]] = arc.state_read %in_a
  // CHECK:   [[B:%.+]] = arc.state_read %in_b
  // CHECK:   [[TMP:%.+]]:2 = arc.call @IdI2AndI42Arc([[A]], [[B]])
  // CHECK:   arc.memory_write [[MEM]][[[TMP]]#0], [[TMP]]#1
  %mem = arc.memory <4 x i42, i2>
  // CHECK:   [[TMP:%.+]] = arc.call @IdI42Arc([[READ_OLD]])
  // CHECK:   arc.state_write [[Q0]] = [[TMP]]
  %q0 = arc.state @IdI42Arc(%0) clock %clock latency 1 {names = ["q0"]} : (i42) -> i42
  // CHECK: }
  // CHECK: [[Q0_NEW:%.+]] = arc.state_read [[Q0]]
  // CHECK: arc.state_write %out_c = [[Q0_NEW]]
  hw.output %q0 : i42
}

// CHECK-LABEL: arc.model @MemoryEnable
hw.module @MemoryEnable(in %clock: !seq.clock, in %a: i2, in %b: i42, in %en: i1) {
  // CHECK: [[MEM:%.+]] = arc.alloc_memory
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[A:%.+]] = arc.state_read %in_a
  // CHECK:   [[B:%.+]] = arc.state_read %in_b
  // CHECK:   [[EN:%.+]] = arc.state_read %in_en
  // CHECK:   [[TMP:%.+]]:3 = arc.call @IdI2AndI42AndI1Arc([[A]], [[B]], [[EN]])
  // CHECK:   scf.if [[TMP]]#2 {
  // CHECK:     arc.memory_write [[MEM]][[[TMP]]#0], [[TMP]]#1
  // CHECK:   }
  // CHECK: }
  %mem = arc.memory <4 x i42, i2>
  arc.memory_write_port %mem, @IdI2AndI42AndI1Arc(%a, %b, %en) clock %clock enable latency 1 : <4 x i42, i2>, i2, i42, i1
}

// CHECK-LABEL: arc.model @MemoryEnableAndMask
hw.module @MemoryEnableAndMask(in %clock: !seq.clock, in %a: i2, in %b: i42, in %en: i1, in %mask: i42) {
  // CHECK: [[MEM:%.+]] = arc.alloc_memory
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[A:%.+]] = arc.state_read %in_a
  // CHECK:   [[B:%.+]] = arc.state_read %in_b
  // CHECK:   [[EN:%.+]] = arc.state_read %in_en
  // CHECK:   [[MASK:%.+]] = arc.state_read %in_mask
  // CHECK:   [[TMP:%.+]]:4 = arc.call @IdI2AndI42AndI1AndI42Arc([[A]], [[B]], [[EN]], [[MASK]])
  // CHECK:   scf.if [[TMP]]#2 {
  // CHECK:     [[ALL_ONES:%.+]] = hw.constant -1
  // CHECK:     [[MASK_INV:%.+]] = comb.xor bin [[TMP]]#3, [[ALL_ONES]]
  // CHECK:     [[DATA_OLD:%.+]] = arc.memory_read [[MEM]][[[TMP]]#0]
  // CHECK:     [[MASKED_OLD:%.+]] = comb.and bin [[MASK_INV]], [[DATA_OLD]]
  // CHECK:     [[MASKED_NEW:%.+]] = comb.and bin [[TMP]]#3, [[TMP]]#1
  // CHECK:     [[DATA_NEW:%.+]] = comb.or bin [[MASKED_OLD]], [[MASKED_NEW]]
  // CHECK:     arc.memory_write [[MEM]][[[TMP]]#0], [[DATA_NEW]]
  // CHECK:   }
  // CHECK: }
  %mem = arc.memory <4 x i42, i2>
  arc.memory_write_port %mem, @IdI2AndI42AndI1AndI42Arc(%a, %b, %en, %mask) clock %clock enable mask latency 1 : <4 x i42, i2>, i2, i42, i1, i42
}

// CHECK-LABEL: arc.model @SimpleInitial
hw.module @SimpleInitial() {
  // CHECK: arc.initial {
  // CHECK:   func.call @VoidFunc() {initA}
  // CHECK:   func.call @VoidFunc() {initB}
  // CHECK: }
  // CHECK: func.call @VoidFunc() {body}
  seq.initial() {
    func.call @VoidFunc() {initA} : () -> ()
  } : () -> ()
  func.call @VoidFunc() {body} : () -> ()
  seq.initial() {
    func.call @VoidFunc() {initB} : () -> ()
  } : () -> ()
}

// CHECK-LABEL: arc.model @InitialWithDependencies
hw.module @InitialWithDependencies() {
  // CHECK:      arc.initial {
  // CHECK-NEXT:   func.call @VoidFunc() {before}
  // CHECK-NEXT:   [[A:%.+]] = func.call @RandomI42() {initA}
  // CHECK-NEXT:   [[B:%.+]] = func.call @RandomI42() {initB}
  // CHECK-NEXT:   [[TMP:%.+]] = comb.add [[A]], [[B]]
  // CHECK-NEXT:   func.call @ConsumeI42([[TMP]])
  // CHECK-NEXT:   func.call @VoidFunc() {after}
  // CHECK-NEXT: }

  seq.initial() {
    func.call @VoidFunc() {before} : () -> ()
  } : () -> ()

  // This pulls up %initA, %initB, %initC since it depends on their SSA values.
  seq.initial(%initC) {
  ^bb0(%arg0: i42):
    func.call @ConsumeI42(%arg0) : (i42) -> ()
  } : (!seq.immutable<i42>) -> ()

  seq.initial() {
    func.call @VoidFunc() {after} : () -> ()
  } : () -> ()

  // The following is pulled up.
  %initA = seq.initial() {
    %1 = func.call @RandomI42() {initA} : () -> i42
    seq.yield %1 : i42
  } : () -> (!seq.immutable<i42>)

  %initB = seq.initial() {
    %2 = func.call @RandomI42() {initB} : () -> i42
    seq.yield %2 : i42
  } : () -> (!seq.immutable<i42>)

  %initC = seq.initial(%initA, %initB) {
  ^bb0(%arg0: i42, %arg1: i42):
    %3 = comb.add %arg0, %arg1 : i42
    seq.yield %3 : i42
  } : (!seq.immutable<i42>, !seq.immutable<i42>) -> (!seq.immutable<i42>)
}

// CHECK-LABEL: arc.model @FromImmutableCast
hw.module @FromImmutableCast() {
  // CHECK: [[STORAGE:%.+]] = arc.alloc_state
  // CHECK: arc.initial {
  // CHECK:   [[TMP:%.+]] = func.call @RandomI42()
  // CHECK:   arc.state_write [[STORAGE]] = [[TMP]]
  // CHECK: }
  // CHECK: [[TMP:%.+]] = arc.state_read [[STORAGE]]
  // CHECK: func.call @ConsumeI42([[TMP]])
  func.call @ConsumeI42(%1) : (i42) -> ()
  %1 = seq.from_immutable %0 : (!seq.immutable<i42>) -> i42
  %0 = seq.initial() {
    %2 = func.call @RandomI42() : () -> (i42)
    seq.yield %2 : i42
  } : () -> (!seq.immutable<i42>)
}

// CHECK-LABEL: arc.model @Taps
hw.module @Taps() {
  // CHECK: [[STORAGE:%.+]] = arc.alloc_state
  // CHECK: [[TMP:%.+]] = func.call @RandomI42()
  // CHECK: arc.state_write [[STORAGE]] = [[TMP]]
  %0 = func.call @RandomI42() : () -> i42
  arc.tap %0 {names = ["myTap"]} : i42
}

// CHECK-LABEL: arc.model @StateInitializerUsesOtherState
hw.module @StateInitializerUsesOtherState(in %clock: !seq.clock, in %a: i42, in %b: i19) {
  // CHECK: [[Q2:%.+]] = arc.alloc_state %arg0 {name = "q2"}
  // CHECK: [[Q3:%.+]] = arc.alloc_state %arg0 {name = "q3"}
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK: [[Q1:%.+]] = arc.alloc_state %arg0 {name = "q1"}
  // CHECK:      arc.initial {
  // CHECK-NEXT:   [[A:%.+]] = arc.state_read %in_a
  // CHECK-NEXT:   arc.state_write [[Q0]] = [[A]]
  // CHECK-NEXT:   [[TMP1:%.+]] = arc.state_read [[Q0]]
  // CHECK-NEXT:   [[TMP2:%.+]] = comb.xor [[TMP1]]
  // CHECK-NEXT:   arc.state_write [[Q1]] = [[TMP2]]
  // CHECK-NEXT:   [[TMP:%.+]] = arc.state_read [[Q1]]
  // CHECK-NEXT:   arc.state_write [[Q2]] = [[TMP]]
  // CHECK-NEXT:   [[TMP:%.+]] = arc.state_read %in_b
  // CHECK-NEXT:   arc.state_write [[Q3]] = [[TMP]]
  // CHECK-NEXT: }
  arc.state @RandomI42AndI19Arc() clock %clock initial (%2, %b : i42, i19) latency 1 {names = ["q2", "q3"]} : () -> (i42, i19)
  %2 = arc.state @RandomI42Arc() clock %clock initial (%1 : i42) latency 1 {names = ["q1"]} : () -> i42
  %1 = comb.xor %0 : i42
  %0 = arc.state @RandomI42Arc() clock %clock initial (%a : i42) latency 1 {names = ["q0"]} : () -> i42
}

// CHECK-LABEL: arc.model @StateInitializerUsesInitial
hw.module @StateInitializerUsesInitial(in %clock: !seq.clock) {
  // CHECK:      [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK:      arc.initial {
  // CHECK-NEXT:   [[TMP1:%.+]] = hw.constant 9001
  // CHECK-NEXT:   [[TMP2:%.+]] = comb.xor [[TMP1]]
  // CHECK-NEXT:   arc.state_write [[Q0]] = [[TMP2]]
  // CHECK-NEXT: }
  %0 = seq.initial() {
    %4 = hw.constant 9001 : i42
    seq.yield %4 : i42
  } : () -> !seq.immutable<i42>
  %1 = seq.initial(%0) {
  ^bb0(%5: i42):
    %6 = comb.xor %5 : i42
    seq.yield %6 : i42
  } : (!seq.immutable<i42>) -> !seq.immutable<i42>
  %2 = seq.from_immutable %1 : (!seq.immutable<i42>) -> i42
  %3 = arc.state @RandomI42Arc() clock %clock initial (%2 : i42) latency 1 {names = ["q0"]} : () -> i42
}

// CHECK-LABEL: arc.model @SimpleFinal
hw.module @SimpleFinal() {
  // CHECK:      arc.final {
  // CHECK-NEXT:   func.call @VoidFunc() {finalA}
  // CHECK-NEXT:   func.call @VoidFunc() {finalB}
  // CHECK-NEXT: }
  // CHECK-NEXT: func.call @VoidFunc() {body}
  llhd.final {
    func.call @VoidFunc() {finalA} : () -> ()
    llhd.halt
  }
  func.call @VoidFunc() {body} : () -> ()
  llhd.final {
    func.call @VoidFunc() {finalB} : () -> ()
    llhd.halt
  }
}

// CHECK-LABEL: arc.model @FinalWithDependencies
hw.module @FinalWithDependencies() {
  // CHECK:      arc.final {
  // CHECK-NEXT:   [[TMP:%.+]] = hw.constant 9001
  // CHECK-NEXT:   func.call @ConsumeI42([[TMP]]) {finalA}
  // CHECK-NEXT:   [[TMP:%.+]] = func.call @RandomI42() {sideEffect}
  // CHECK-NEXT:   func.call @ConsumeI42([[TMP]]) {finalB}
  // CHECK-NEXT: }
  // CHECK-NEXT: func.call @VoidFunc() {body}
  // CHECK-NEXT: func.call @RandomI42() {sideEffect}
  llhd.final {
    func.call @ConsumeI42(%0) {finalA} : (i42) -> ()
    llhd.halt
  }
  func.call @VoidFunc() {body} : () -> ()
  llhd.final {
    func.call @ConsumeI42(%1) {finalB} : (i42) -> ()
    llhd.halt
  }
  %0 = hw.constant 9001 : i42
  %1 = func.call @RandomI42() {sideEffect} : () -> i42
}

// CHECK-LABEL: arc.model @FinalWithControlFlow
hw.module @FinalWithControlFlow() {
  // CHECK:      arc.final {
  // CHECK-NEXT:   scf.execute_region {
  // CHECK-NEXT:     [[TMP:%.+]] = func.call @RandomI42()
  // CHECK-NEXT:     cf.br ^[[BB:.+]]([[TMP]] : i42)
  // CHECK-NEXT:   ^[[BB]]([[TMP:%.+]]: i42)
  // CHECK-NEXT:     func.call @ConsumeI42([[TMP]])
  // CHECK-NEXT:     scf.yield
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  llhd.final {
    %0 = func.call @RandomI42() : () -> i42
    cf.br ^bb0(%0 : i42)
  ^bb0(%1: i42):
    func.call @ConsumeI42(%1) : (i42) -> ()
    llhd.halt
  }
}

// CHECK-LABEL: arc.model @UnclockedDpiCall
hw.module @UnclockedDpiCall(in %a: i42, out b: i42) {
  // CHECK: [[TMP1:%.+]] = arc.state_read %in_a
  // CHECK: [[TMP2:%.+]] = func.call @IdI42([[TMP1]])
  %0 = sim.func.dpi.call @IdI42(%a) : (i42) -> i42
  // CHECK: arc.state_write %out_b = [[TMP2]]
  hw.output %0 : i42
}

// CHECK-LABEL: arc.model @ClockedDpiCall
hw.module @ClockedDpiCall(in %clock: !seq.clock, in %a: i42, out b: i42) {
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  // CHECK: [[Q1:%.+]] = arc.alloc_state %arg0 {name = "q1"}
  // CHECK: [[Q0_OLD:%.+]] = arc.state_read [[Q0]]
  // CHECK:      scf.if {{%.+}} {
  // CHECK-NEXT:   [[TMP1:%.+]] = arc.state_read %in_a
  // CHECK-NEXT:   [[TMP2:%.+]] = func.call @IdI42([[TMP1]])
  // CHECK-NEXT:   arc.state_write [[Q0]] = [[TMP2]]
  // CHECK-NEXT:   [[TMP3:%.+]] = func.call @IdI42([[Q0_OLD]])
  // CHECK-NEXT:   arc.state_write [[Q1]] = [[TMP3]]
  // CHECK-NEXT: }
  %0 = sim.func.dpi.call @IdI42(%a) clock %clock {names = ["q0"]} : (i42) -> i42
  %1 = sim.func.dpi.call @IdI42(%0) clock %clock {names = ["q1"]} : (i42) -> i42
  // CHECK-NEXT: [[TMP:%.+]] = arc.state_read [[Q1]]
  // CHECK-NEXT: arc.state_write %out_b = [[TMP]]
  hw.output %1 : i42
}

// CHECK-LABEL: arc.model @OpsWithRegions
hw.module @OpsWithRegions(in %clock: !seq.clock, in %a: i42, in %b: i1, out c: i42) {
  // CHECK: [[Q0:%.+]] = arc.alloc_state %arg0 {name = "q0"}
  %0 = scf.if %b -> (i42) {
    scf.yield %a : i42
  } else {
    scf.yield %1 : i42
  }
  // CHECK: [[A:%.+]] = arc.state_read %in_a
  // CHECK: [[Q0_OLD:%.+]] = arc.state_read [[Q0]]
  // CHECK: [[B:%.+]] = arc.state_read %in_b
  // CHECK: [[IF_OLD:%.+]] = scf.if [[B]] -> (i42) {
  // CHECK:   scf.yield [[A]]
  // CHECK: } else {
  // CHECK:   scf.yield [[Q0_OLD]]
  // CHECK: }
  // CHECK: scf.if {{%.+}} {
  // CHECK:   [[TMP:%.+]] = arc.call @IdI42Arc([[IF_OLD]])
  // CHECK:   arc.state_write [[Q0]] = [[TMP]]
  // CHECK: }
  %1 = arc.state @IdI42Arc(%0) clock %clock latency 1 {names = ["q0"]} : (i42) -> i42
  // CHECK: [[A:%.+]] = arc.state_read %in_a
  // CHECK: [[Q0_NEW:%.+]] = arc.state_read [[Q0]]
  // CHECK: [[B:%.+]] = arc.state_read %in_b
  // CHECK: [[IF_NEW:%.+]] = scf.if [[B]] -> (i42) {
  // CHECK:   scf.yield [[A]]
  // CHECK: } else {
  // CHECK:   scf.yield [[Q0_NEW]]
  // CHECK: }
  // CHECK: arc.state_write %out_c = [[IF_NEW]]
  hw.output %0 : i42
}

// Regression check on worklist producing false positive comb loop errors.
// CHECK-LABEL: @CombLoopRegression
hw.module @CombLoopRegression(in %clk: !seq.clock) {
  %0 = arc.state @CombLoopRegressionArc1(%3, %3) clock %clk latency 1 : (i1, i1) -> i1
  %1, %2 = arc.call @CombLoopRegressionArc2(%0) : (i1) -> (i1, i1)
  %3 = arc.call @CombLoopRegressionArc1(%1, %2) : (i1, i1) -> i1
}
arc.define @CombLoopRegressionArc1(%arg0: i1, %arg1: i1) -> i1 {
  arc.output %arg0 : i1
}
arc.define @CombLoopRegressionArc2(%arg0: i1) -> (i1, i1) {
  arc.output %arg0, %arg0 : i1, i1
}

// Regression check for invalid memory port lowering errors.
// CHECK-LABEL: arc.model @MemoryPortRegression
hw.module private @MemoryPortRegression(in %clock: !seq.clock, in %reset: i1, in %in: i3, out x: i3) {
  %0 = arc.memory <2 x i3, i1>
  %1 = arc.memory_read_port %0[%3] : <2 x i3, i1>
  arc.memory_write_port %0, @MemoryPortRegressionArc1(%3, %in) clock %clock latency 1 : <2 x i3, i1>, i1, i3
  %3 = arc.state @MemoryPortRegressionArc2(%reset) clock %clock latency 1 : (i1) -> i1
  %4 = arc.call @MemoryPortRegressionArc3(%1) : (i3) -> i3
  hw.output %4 : i3
}
arc.define @MemoryPortRegressionArc1(%arg0: i1, %arg1: i3) -> (i1, i3) {
  arc.output %arg0, %arg1 : i1, i3
}
arc.define @MemoryPortRegressionArc2(%arg0: i1) -> i1 {
  arc.output %arg0 : i1
}
arc.define @MemoryPortRegressionArc3(%arg0: i3) -> i3 {
  arc.output %arg0 : i3
}
