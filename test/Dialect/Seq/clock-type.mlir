// RUN: circt-opt --lower-seq-to-sv %s | FileCheck %s

hw.generator.schema @Some_schema, "Some_schema", ["dummy"]

// CHECK-LABEL: hw.module.generated @generated, @Some_schema(in %clock : i1, in %other : i1, out out : i32)
hw.module.generated @generated, @Some_schema(in %clock: !seq.clock, in %other: i1, out out: i32) attributes { dummy = 1 : i32 }

// CHECK-LABEL: hw.module.extern @extern(in %clock : i1, in %other : i1, out out : i32)
hw.module.extern @extern(in %clock: !seq.clock, in %other: i1, out out: i32)

// CHECK-LABEL: @const_clock
hw.module @const_clock() {
  // CHECK: hw.constant false
  seq.const_clock low

  // CHECK: hw.constant true
  seq.const_clock high
}

// CHECK-LABEL: hw.module @top(in %clock : i1, in %other : i1, in %wire : i1, out out : i1)
hw.module @top(in %clock: !seq.clock, in %other: i1, in %wire: i1, out out: !seq.clock) {


  // CHECK: %tap_generated = hw.wire %clock  : i1
  %tap_generated = hw.wire %clock : !seq.clock

  // CHECK: %generated.out = hw.instance "generated" @generated(clock: %tap_generated: i1, other: %other: i1) -> (out: i32)
  %generated.out = hw.instance "generated" @generated(clock: %tap_generated: !seq.clock, other: %other: i1) -> (out: i32)

  // CHECK: %extern.out = hw.instance "extern" @generated(clock: %clock: i1, other: %other: i1) -> (out: i32)
  %extern.out = hw.instance "extern" @generated(clock: %clock: !seq.clock, other: %other: i1) -> (out: i32)

  // CHECK: %inner.out = hw.instance "inner" @inner(clock: %clock: i1, other: %other: i1) -> (out: i32)
  %out_inner = hw.instance "inner" @inner(clock: %clock: !seq.clock, other: %other: i1) -> (out: i32)

  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK: [[NEGATED:%.+]] = comb.xor %clock, [[TRUE]] : i1
  %clock_wire = seq.from_clock %clock
  %one = hw.constant 1 : i1
  %tmp = comb.xor %clock_wire, %one : i1
  %negated = seq.to_clock %tmp

  // CHECK: hw.output [[NEGATED]] : i1
  hw.output %negated : !seq.clock
}

// CHECK-LABEL: hw.module private @inner(in %clock : i1, in %other : i1, out out : i32)
hw.module private @inner(in %clock: !seq.clock, in %other: i1, out out: i32) {
  %cst = hw.constant 0 : i32
  hw.output %cst : i32
}

// CHECK-LABEL: hw.module private @SinkSource(in %clock : i1, out out : i1)
hw.module private @SinkSource(in %clock: !seq.clock, out out: !seq.clock) {
  // CHECK: hw.output %false : i1
  %out = seq.const_clock low
  hw.output %out : !seq.clock
}

// CHECK-LABEL: hw.module public @CrossReferences()
hw.module public @CrossReferences() {
  // CHECK: %a.out = hw.instance "a" @SinkSource(clock: %b.out: i1) -> (out: i1)
  %a.out = hw.instance "a" @SinkSource(clock: %b.out: !seq.clock) -> (out: !seq.clock)
  // CHECK: %b.out = hw.instance "b" @SinkSource(clock: %a.out: i1) -> (out: i1)
  %b.out = hw.instance "b" @SinkSource(clock: %a.out: !seq.clock) -> (out: !seq.clock)
}

// CHECK-LABEL: hw.module @ClockAgg(in %c : !hw.struct<clock: i1>, out oc : !hw.struct<clock: i1>)
// CHECK: [[CLOCK:%.+]] = hw.struct_extract %c["clock"] : !hw.struct<clock: i1>
// CHECK: [[STRUCT:%.+]] = hw.struct_create ([[CLOCK]]) : !hw.struct<clock: i1>
// CHECL: hw.output [[STRUCT]] : !hw.struct<clock: i1>
hw.module @ClockAgg(in %c: !hw.struct<clock: !seq.clock>, out oc: !hw.struct<clock: !seq.clock>) {
  %clock = hw.struct_extract %c["clock"] : !hw.struct<clock: !seq.clock>
  %0 = hw.struct_create (%clock) : !hw.struct<clock: !seq.clock>
  hw.output %0 : !hw.struct<clock: !seq.clock>
}

// CHECK-LABEL: hw.module @ClockArray(in %c : !hw.array<1xi1>, out oc : !hw.array<1xi1>) {
// CHECK: [[CLOCK:%.+]] = hw.array_get %c[%false] : !hw.array<1xi1>, i1
// CHECK: [[ARRAY:%.+]] = hw.array_create [[CLOCK]] : i1
// CHECK: hw.output [[ARRAY]] : !hw.array<1xi1>
hw.module @ClockArray(in %c: !hw.array<1x!seq.clock>, out oc: !hw.array<1x!seq.clock>) {
  %idx = hw.constant 0 : i1
  %clock = hw.array_get %c[%idx] : !hw.array<1x!seq.clock>, i1
  %0 = hw.array_create %clock : !seq.clock
  hw.output %0 : !hw.array<1x!seq.clock>
}
