// RUN: circt-opt --msft-create-generic-pd-hierarchy %s | FileCheck %s
// RUN: circt-opt --msft-create-generic-pd-hierarchy --msft-lower-instances --lower-seq-to-sv --msft-export-tcl --export-verilog %s -o %t.mlir | FileCheck %s --check-prefix=TCL

// CHECK-LABEL:   msft.instance.hierarchy @top_generic_pd_config(@top) {
// CHECK:           msft.instance.hierarchy.call @middle_generic_pd_config ref @top.myMiddle
// CHECK:         }

// CHECK:         hw.hierpath @top.myMiddle [@top::@myMiddle]

// CHECK-LABEL:   msft.instance.hierarchy @middle_generic_pd_config(@middle) {
// CHECK:           msft.instance.hierarchy.call @reg_generic_pd_config ref @middle.regA
// CHECK:           msft.instance.hierarchy.call @reg_generic_pd_config ref @middle.regB
// CHECK:         }

// CHECK:         hw.hierpath @middle.regA [@middle::@regA]
// CHECK:         hw.hierpath @middle.regB [@middle::@regB]
// CHECK:         hw.hierpath @reg.reg1 [@reg::@reg1]
// CHECK:         hw.hierpath @reg.reg2 [@reg::@reg2]

// CHECK-LABEL:   msft.instance.hierarchy @reg_generic_pd_config(@reg) {
// CHECK:           msft.pd.multicycle 2 @reg.reg1 -> @reg.reg2
// CHECK:         }

msft.instance.hierarchy @top_pd(@top) {
  msft.instance.dynamic @top::@myMiddle {
    msft.instance.dynamic @middle::@regA {
      msft.instance.dynamic @reg::@reg1 {
        msft.pd.reg_location i4 [*, <1,2,3>, <1,2,4>, <1,2,5>]
      }
    }
  }
}

hw.module @top(in %input : i4, in %clk : !seq.clock) {
  hw.instance "myMiddle" sym @myMiddle @middle(input : %input : i4, clk : %clk : !seq.clock) -> ()
}

hw.module @middle(in %input : i4, in %clk : !seq.clock) {
  hw.instance "regA" sym @regA @reg(input : %input : i4, clk : %clk : !seq.clock) -> ()
  hw.instance "regB" sym @regB @reg(input : %input : i4, clk : %clk : !seq.clock) -> ()
}

hw.module @reg (in %input : i4, in %clk : !seq.clock) {
  %reg1 = seq.compreg sym @reg1 %input, %clk  : i4
  %reg2 = seq.compreg sym @reg2 %reg1, %clk  : i4
  msft.pd.multicycle 2 @reg1 -> @reg2
}


// TCL: proc reg_0_config { parent } {
// TCL:   set_multicycle_path -hold 1 -setup 2 -from [get_registers {$parent|reg1}] -to [get_registers {$parent|reg2}]
// TCL: }
// TCL: proc top_config { parent } {
// TCL:   set_location_assignment FF_X1_Y2_N3 -to $parent|myMiddle|regA|reg1[1]
// TCL:   set_location_assignment FF_X1_Y2_N4 -to $parent|myMiddle|regA|reg1[2]
// TCL:   set_location_assignment FF_X1_Y2_N5 -to $parent|myMiddle|regA|reg1[3]
// TCL:   middle_config $parent|myMiddle
// TCL: }
// TCL: proc middle_config { parent } {
// TCL:   reg_0_config $parent|regA
// TCL:   reg_0_config $parent|regB
// TCL: }
