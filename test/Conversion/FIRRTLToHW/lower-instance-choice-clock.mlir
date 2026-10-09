// RUN: circt-opt -lower-firrtl-to-hw %s | FileCheck %s

// Test that instance_choice with clock-typed output ports is lowered correctly.
// Clock types cannot be used as inout element types, so we use i1 wires and
// convert to/from clock when reading/writing.

module {
  firrtl.circuit "Top" {
    sv.macro.decl @__option_Opt_A {sym_visibility = "private"}
    sv.macro.decl @__option_Opt_B {sym_visibility = "private"}
    sv.macro.decl @__target_Opt_inst {sym_visibility = "private"}
    
    firrtl.option @Opt attributes {sym_visibility = "private"} {
      firrtl.option_case @A {case_macro = @__option_Opt_A}
      firrtl.option_case @B {case_macro = @__option_Opt_B}
    }
    
    firrtl.extmodule private @ModA(in clk_in: !firrtl.clock, out clk_out: !firrtl.clock) attributes {convention = #firrtl<convention scalarized>, defname = "ModA"}
    firrtl.extmodule private @ModB(in clk_in: !firrtl.clock, out clk_out: !firrtl.clock) attributes {convention = #firrtl<convention scalarized>, defname = "ModB"}
    
    // CHECK-LABEL: hw.module @Top
    firrtl.module @Top(in %clk: !firrtl.clock, out %out_clk: !firrtl.clock) {
      // CHECK: %inst.clk_out = sv.wire : !hw.inout<i1>
      // CHECK-NEXT: %[[READ:.+]] = sv.read_inout %inst.clk_out : !hw.inout<i1>
      // CHECK-NEXT: %[[TO_CLOCK:.+]] = seq.to_clock %[[READ]]
      %inst_clk_in, %inst_clk_out = firrtl.instance_choice inst {instance_macro = @__target_Opt_inst} @ModA alternatives @Opt { @A -> @ModA, @B -> @ModB } (in clk_in: !firrtl.clock, out clk_out: !firrtl.clock)
      firrtl.matchingconnect %inst_clk_in, %clk : !firrtl.clock
      firrtl.matchingconnect %out_clk, %inst_clk_out : !firrtl.clock
      
      // CHECK: sv.ifdef @__option_Opt_A {
      // CHECK:   %inst_A.clk_out = hw.instance "inst_A"
      // CHECK:   %[[FROM_CLOCK_A:.+]] = seq.from_clock %inst_A.clk_out
      // CHECK:   sv.assign %inst.clk_out, %[[FROM_CLOCK_A]] : i1
      // CHECK: } else {
      // CHECK:   sv.ifdef @__option_Opt_B {
      // CHECK:     %inst_B.clk_out = hw.instance "inst_B"
      // CHECK:     %[[FROM_CLOCK_B:.+]] = seq.from_clock %inst_B.clk_out
      // CHECK:     sv.assign %inst.clk_out, %[[FROM_CLOCK_B]] : i1
      // CHECK:   } else {
      // CHECK:     %inst_default.clk_out = hw.instance "inst_default"
      // CHECK:     %[[FROM_CLOCK_DEFAULT:.+]] = seq.from_clock %inst_default.clk_out
      // CHECK:     sv.assign %inst.clk_out, %[[FROM_CLOCK_DEFAULT]] : i1
      
      // CHECK: hw.output %[[TO_CLOCK]] : !seq.clock
    }
  }
}

