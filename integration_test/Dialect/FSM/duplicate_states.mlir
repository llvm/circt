// REQUIRES: verilator

// RUN: circt-opt --convert-fsm-to-sv /home/mortbopet/Downloads/fsm.mlir --lower-seq-to-sv --export-split-verilog -o %t
// RUN: verilator --lint-only --top-module M1 fsm_enum_typedefs.sv M1.sv M2.sv

module {
  fsm.machine @M1() attributes {initialState = "A"} {
    fsm.state @A
    fsm.state @B
  }

  fsm.machine @M2() attributes {initialState = "A"} {
    fsm.state @A
    fsm.state @B
  }
}
