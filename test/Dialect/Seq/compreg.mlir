// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=SV
// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv='lower-to-always-ff=false' | FileCheck %s --check-prefix=ALWAYS
hw.module @top(in %clk: !seq.clock, in %rst: i1, in %i: i32, in %s: !hw.struct<foo: i32>) {

  %r0 = seq.compreg %i, %clk reset %rst, %c0_i32 : i32
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

  %rv = seq.initial () {
    %c0_i32_0 = hw.constant 0 : i32
    seq.yield %c0_i32_0 : i32
  } : () -> !seq.immutable<i32>

  %c0_i32 = hw.constant 0 : i32

  %withinitial = seq.compreg sym @withinitial %i, %clk reset %rst, %c0_i32 initial %rv : i32
  // SV: %withinitial = sv.reg init %{{c0_i32.*}} sym @withinitial : !hw.inout<i32>
}

hw.module @top_ce(in %clk: !seq.clock, in %rst: i1, in %ce: i1, in %i: i32) {
  %rv = hw.constant 0 : i32
  %init = seq.initial () {
    %c0_i32 = hw.constant 0 : i32
    seq.yield %c0_i32 : i32
  } : () -> !seq.immutable<i32>

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

  %withinitial = seq.compreg.ce sym @withinitial %i, %clk, %ce reset %rst, %rv initial %init : i32
  // SV: %withinitial = sv.reg init %{{c0_i32.*}} sym @withinitial : !hw.inout<i32>
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

hw.module @init_with_call(in %clk: !seq.clock, in %rst: i1, in %i: i32, in %s: !hw.struct<foo: i32>, out o: i32) {
  // SV:     sv.initial {
  // SV-NEXT:   [[V0:%.+]] = sv.system "random"() : () -> i32
  // SV-NEXT:   [[V1:%.+]] = comb.add [[V0]], [[V0]] : i32
  // SV-NEXT:   sv.bpassign %reg, [[V0]] : i32
  // SV-NEXT:   sv.bpassign %reg2, [[V1]] : i32
  // SV-NEXT:   sv.bpassign [[REG:%.+]], [[V1]] : i32
  // SV-NEXT: }
  %init = seq.initial () {
    %rand = sv.system "random"() : () -> i32
    seq.yield %rand : i32
  } : () -> !seq.immutable<i32>

  %add = seq.initial (%init) {
    ^bb0(%arg0 : i32):
    %0 = comb.add %arg0, %arg0: i32
    seq.yield %0 : i32
  } : (!seq.immutable<i32>) -> !seq.immutable<i32>

  // SV: %reg = sv.reg : !hw.inout<i32>
  %c0_i32 = hw.constant 0 : i32

  %reg = seq.compreg %i, %clk initial %init : i32
  %reg2 = seq.compreg %i, %clk initial %add : i32

  %add_from_immut = seq.from_immutable %add: (!seq.immutable<i32>) -> i32
  // SV:  [[REG]] = sv.reg
  // SV-NEXT: [[RESULT:%.+]] = sv.read_inout [[REG]]
  // SV-NEXT: hw.output [[RESULT]]

  hw.output %add_from_immut: i32
}

// Test that empty name attributes don't get SSA name hints
hw.module @empty_name_test(in %clk: !seq.clock, in %i: i32, in %ce: i1) {
  // Registers with empty name attributes should get auto-numbered (%0, %1, etc)
  // not named results. The name attribute is elided in the output when empty.
  %0 = seq.compreg %i, %clk {name = ""} : i32
  %1 = seq.compreg.ce %i, %clk, %ce {name = ""} : i32
  
  // Registers with non-empty name attributes should get named SSA values
  %named = seq.compreg %i, %clk {name = "named"} : i32
  %named_ce = seq.compreg.ce %i, %clk, %ce {name = "named_ce"} : i32
  
  // CHECK: @empty_name_test
  // CHECK-NEXT: %0 = seq.compreg %i, %clk : i32
  // CHECK-NEXT: %1 = seq.compreg.ce %i, %clk, %ce : i32
  // CHECK-NEXT: %named = seq.compreg %i, %clk : i32
  // CHECK-NEXT: %named_ce = seq.compreg.ce %i, %clk, %ce : i32
}
