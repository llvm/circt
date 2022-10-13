// REQUIRES: verilator

// RUN: circt-opt --export-verilog %s -o %t > %t.sv
// RUN: verilator --lint-only %t.sv

hw.type_scope @fsm_enum_typedecls {
    hw.typedecl @M2_state_t : !hw<enum<M2_state: [A, B]>>
    hw.typedecl @M1_state_t : !hw<enum<M1_state: [A, B]>>
}
