// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=SV
// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv='lower-to-always-ff=false' | FileCheck %s --check-prefix=ALWAYS
hw.module @top(in %clk: !seq.clock, in %rst: i1, in %i: i32, in %s: !hw.struct<foo: i32>) {
  %rv = hw.constant 0 : i32

  %r0 = seq.compreg %i, %clk reset %rst, %rv : i32
  seq.compreg %i, %clk : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk reset %rst, %c0_i32  : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk : i32
  // SV: [[REG0:%.+]] = sv.reg  : !hw.inout<i32>
  // SV: [[REG5:%.+]] = sv.read_inout [[REG0]] : !hw.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG0]], %i : i32
  // SV: }(syncreset : posedge %rst)  {
  // SV:   sv.passign [[REG0]], %c0_i32 : i32
  // SV: }
  // SV: [[REG1:%.+]] = sv.reg  : !hw.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG1]], %i : i32
  // SV: }
  // ALWAYS: [[R0:%.+]] = sv.reg : !hw.inout<i32>
  // ALWAYS: [[R0_VAL:%.+]] = sv.read_inout [[R0]] : !hw.inout<i32>
  // ALWAYS: sv.always posedge %clk {
  // ALWAYS:   sv.if %rst {
  // ALWAYS:     sv.passign [[R0]], %c0_i32 : i32
  // ALWAYS:   } else {
  // ALWAYS:     sv.passign [[R0]], %i : i32
  // ALWAYS:   }
  // ALWAYS: }
  // ALWAYS: [[R1:%.+]] = sv.reg : !hw.inout<i32>
  // ALWAYS: sv.always posedge %clk {
  // ALWAYS:   sv.passign [[R1]], %i : i32
  // ALWAYS: }

  %sv = hw.struct_create (%r0) : !hw.struct<foo: i32>

  %foo = seq.compreg %s, %clk reset %rst, %sv {sv.attributes=[#sv.attribute<"dont_merge">]} : !hw.struct<foo: i32>
  seq.compreg %s, %clk : !hw.struct<foo: i32>
  // CHECK: %foo = seq.compreg %s, %clk reset %rst, %{{.+}} : !hw.struct<foo: i32>
  // CHECK: %{{.+}} = seq.compreg %s, %clk : !hw.struct<foo: i32>

  // SV: [[REGST:%.+]] = hw.struct_create ([[REG5]]) : !hw.struct<foo: i32>
  // SV: %foo = sv.reg {sv.attributes = [#sv.attribute<"dont_merge">]} : !hw.inout<struct<foo: i32>>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign %foo, %s : !hw.struct<foo: i32>
  // SV: }(syncreset : posedge %rst)  {
  // SV:   sv.passign %foo, [[REGST]] : !hw.struct<foo: i32>
  // SV: }
  // SV: [[REG4:%.+]] = sv.reg : !hw.inout<struct<foo: i32>>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG4]], %s : !hw.struct<foo: i32>
  // SV: }
  // ALWAYS: [[FOO_NEXT:%.+]] = hw.struct_create ([[R0_VAL]]) : !hw.struct<foo: i32>
  // ALWAYS: %foo = sv.reg {sv.attributes = [#sv.attribute<"dont_merge">]} : !hw.inout<struct<foo: i32>>
  // ALWAYS: sv.always posedge %clk {
  // ALWAYS:   sv.if %rst {
  // ALWAYS:     sv.passign %foo, [[FOO_NEXT]] : !hw.struct<foo: i32>
  // ALWAYS:   } else {
  // ALWAYS:     sv.passign %foo, %s : !hw.struct<foo: i32>
  // ALWAYS:   }
  // ALWAYS: }
  // ALWAYS: [[REG4:%.+]] = sv.reg : !hw.inout<struct<foo: i32>>
  // ALWAYS: sv.always posedge %clk {
  // ALWAYS:   sv.passign [[REG4]], %s : !hw.struct<foo: i32>
  // ALWAYS: }

  %bar = seq.compreg sym @reg1 %i, %clk : i32
  seq.compreg sym @reg2 %i, %clk : i32
  // CHECK: %bar = seq.compreg sym @reg1
  // CHECK: seq.compreg sym @reg2

  // SV: %bar = sv.reg sym @reg1
  // SV: sv.reg sym @reg2

  %withPowerOn = seq.compreg sym @withPowerOn %i, %clk reset %rst, %rv powerOn %rv : i32
  // SV: %withPowerOn = sv.reg init %c0_i32 sym @withPowerOn : !hw.inout<i32>
}

hw.module @top_ce(in %clk: !seq.clock, in %rst: i1, in %ce: i1, in %i: i32) {
  %rv = hw.constant 0 : i32

  %r0 = seq.compreg.ce %i, %clk, %ce reset %rst, %rv : i32
  // CHECK: %r0 = seq.compreg.ce %i, %clk, %ce reset %rst, %c0_i32  : i32
  // SV: [[REG_CE0:%.+]] = sv.reg  : !hw.inout<i32>
  // SV: [[REG_CE5:%.+]] = sv.read_inout [[REG0]] : !hw.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.if %ce {
  // SV:     sv.passign [[REG_CE0]], %i : i32
  // SV:   }
  // SV: }(syncreset : posedge %rst)  {
  // SV:   sv.passign [[REG_CE0]], %c0_i32 : i32
  // SV: }
  // ALWAYS: [[R0:%.+]] = sv.reg : !hw.inout<i32>
  // ALWAYS: [[R0_VAL:%.+]] = sv.read_inout [[R0]] : !hw.inout<i32>
  // ALWAYS: sv.always posedge %clk {
  // ALWAYS:   sv.if %rst {
  // ALWAYS:     sv.passign [[R0]], %c0_i32 : i32
  // ALWAYS:   } else {
  // ALWAYS:     sv.if %ce {
  // ALWAYS:       sv.passign [[R0]], %i : i32
  // ALWAYS:     }
  // ALWAYS:   }
  // ALWAYS: }

  %withPowerOn = seq.compreg.ce sym @withPowerOn %i, %clk, %ce reset %rst, %rv powerOn %rv : i32
  // SV: %withPowerOn = sv.reg init %c0_i32 sym @withPowerOn : !hw.inout<i32>
}

// SV-LABEL: @reg_of_clock_type
hw.module @reg_of_clock_type(in %clk: !seq.clock, in %rst: i1, in %i: !seq.clock, out out: !seq.clock) {
  // SV: [[REG0:%.+]] = sv.reg : !hw.inout<i1>
  // SV: [[REG0_VAL:%.+]] = sv.read_inout [[REG0]] : !hw.inout<i1>
  // SV: sv.alwaysff(posedge %clk) {
  // SV:   sv.passign [[REG0]], %i : i1
  // SV: }
  %r0 = seq.compreg %i, %clk : !seq.clock

  // SV: [[REG1:%.+]] = sv.reg : !hw.inout<i1>
  // SV: [[REG1_VAL:%.+]] = sv.read_inout [[REG1]] : !hw.inout<i1>
  // SV: sv.alwaysff(posedge %clk) {
  // SV:   sv.passign [[REG1]], [[REG0_VAL]] : i1
  // SV: }
  %r1 = seq.compreg %r0, %clk : !seq.clock

  // SV: hw.output [[REG1_VAL]] : i1
  hw.output %r1 : !seq.clock
}
