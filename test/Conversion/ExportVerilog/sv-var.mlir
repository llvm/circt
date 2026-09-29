// RUN: circt-opt %s --export-verilog | FileCheck %s --check-prefix=VARLOGIC
// RUN: circt-opt %s --test-apply-lowering-options='options=svVarDeclStyle=logic' --export-verilog | FileCheck %s --check-prefix=LOGIC
// RUN: circt-opt %s --test-apply-lowering-options='options=svVarDeclStyle=reg' --export-verilog | FileCheck %s --check-prefix=REG


hw.type_scope @var_types {
  hw.typedecl @byte_t : i8
}

// VARLOGIC-LABEL: module var_declarations(
// VARLOGIC: var logic{{ *}}scalar;
// VARLOGIC: var logic{{ *}}[7:0]{{ *}}with_init = 8'h0;
// VARLOGIC: var logic{{ *}}[1:0][7:0]{{ *}}packed_0;
// VARLOGIC: var logic{{ *}}[7:0]{{ *}}unpacked[0:1];
// VARLOGIC: var logic{{ *}}[2:0][3:0]{{ *}}packed_unpacked[0:1];
// VARLOGIC: var{{ *}}struct packed {logic a; logic [2:0] b; }{{ *}}struct_value
// VARLOGIC: var{{ *}}union packed {{.*}}union_value;
// VARLOGIC: var{{ *}}enum {{.*}}enum_value;
// VARLOGIC: var{{ *}}byte_t{{ *}}alias_value;
// VARLOGIC: var{{ *}}byte_t{{ *}}alias_unpacked[0:1];

// LOGIC-LABEL: module var_declarations(
// LOGIC: logic{{ *}}scalar;
// LOGIC: logic{{ *}}[7:0]{{ *}}with_init = 8'h0;
// LOGIC: logic{{ *}}[1:0][7:0]{{ *}}packed_0;
// LOGIC: logic{{ *}}[7:0]{{ *}}unpacked[0:1];
// LOGIC: logic{{ *}}[2:0][3:0]{{ *}}packed_unpacked[0:1];
// LOGIC: struct packed {logic a; logic [2:0] b; }{{ *}}struct_value
// LOGIC: union packed {{.*}}union_value;
// LOGIC: enum {{.*}}enum_value;
// LOGIC: byte_t{{ *}}alias_value;
// LOGIC: byte_t{{ *}}alias_unpacked[0:1];

// REG-LABEL: module var_declarations(
// REG: reg{{ *}}scalar;
// REG: reg{{ *}}[7:0]{{ *}}with_init = 8'h0;
// REG: reg{{ *}}[1:0][7:0]{{ *}}packed_0;
// REG: reg{{ *}}[7:0]{{ *}}unpacked[0:1];
// REG: reg{{ *}}[2:0][3:0]{{ *}}packed_unpacked[0:1];
// REG: struct packed {logic a; logic [2:0] b; }{{ *}}struct_value
// REG: union packed {{.*}}union_value;
// REG: enum {{.*}}enum_value;
// REG: byte_t{{ *}}alias_value;
// REG: byte_t{{ *}}alias_unpacked[0:1];


hw.module @var_declarations() {
    %zero_i1 = hw.constant 0 : i1
    %zero_i8 = hw.constant 0 : i8

    %scalar = sv.var : !sv.var<i1>
    %with_init = sv.var init %zero_i8 : !sv.var<i8>

    %packed = sv.var : !sv.var<!hw.array<2xi8>>
    %unpacked = sv.var : !sv.var<!hw.uarray<2xi8>>
    %packed_unpacked = sv.var : !sv.var<!hw.uarray<2x!hw.array<3xi4>>>

    %struct_value = sv.var : !sv.var<!hw.struct<a: i1, b: i3>>
    %union_value = sv.var : !sv.var<!hw.union<raw: i8, bytes: i8>>
    %enum_value = sv.var : !sv.var<!hw.enum<A, B, C>>

    %alias_value = sv.var : !sv.var<!hw.typealias<@var_types::@byte_t, i8>>
    %alias_unpacked = sv.var :  !sv.var<!hw.uarray<2x!hw.typealias<@var_types::@byte_t, i8>>>


}