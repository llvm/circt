// RUN: circt-opt -export-verilog --split-input-file %s | FileCheck %s --strict-whitespace --match-full-lines

sv.interface @IValidReady_Struct  {
  sv.interface.signal @data : !hw.struct<foo: !hw.array<72xi1>, bar: !hw.array<128xi1>, baz: !hw.array<224xi1>>
}

// CHECK-LABEL:module structs({{.*}}
// CHECK:  assign _GEN =
// CHECK-NEXT:    '{
// CHECK-NEXT:      foo: ({_GEN_1, _GEN_0}),
// CHECK-NEXT:      bar: ({_GEN_0, _GEN_0}),
// CHECK-NEXT:      baz:
// CHECK-NEXT:        ({
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_1,
// CHECK-NEXT:           _GEN_0,
// CHECK-NEXT:           _GEN_0
// CHECK-NEXT:         })
// CHECK-NEXT:    };{{.*}}
hw.module @structs(%clk: i1, %rstn: i1) {
  %0 = sv.interface.instance {name = "iface"} : !sv.interface<@IValidReady_Struct>
  sv.interface.signal.assign %0(@IValidReady_Struct::@data) = %s : !hw.struct<foo: !hw.array<72xi1>, bar: !hw.array<128xi1>, baz: !hw.array<224xi1>>
  %c0 = hw.constant 0 : i8
  %c64 = hw.constant 100000 : i64
  %16 = hw.bitcast %c64 : (i64) -> !hw.array<64xi1>
  %58 = hw.bitcast %c0 : (i8) -> !hw.array<8xi1>
  %90 = hw.array_concat %58, %16 : !hw.array<8xi1>, !hw.array<64xi1>
  %91 = hw.array_concat %16, %16 : !hw.array<64xi1>, !hw.array<64xi1>
  %92 = hw.array_concat %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %16, %16 : !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<64xi1>, !hw.array<64xi1>
  %s = hw.struct_create (%90, %91, %92) : !hw.struct<foo: !hw.array<72xi1>, bar: !hw.array<128xi1>, baz: !hw.array<224xi1>>
}

// -----

// CHECK-LABEL:module CoverAssert({{.*}}
hw.module @CoverAssert(
  %clock: i1, %reset: i1,
  %eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0: i4) {
    %c0_i4 = hw.constant 0 : i4
    %true = hw.constant true

    %0 = comb.icmp eq %eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0, %c0_i4 : i4
    %1 = comb.icmp eq %eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0, %c0_i4 : i4

    %2 = comb.xor bin %reset, %true : i1
    %3 = comb.xor bin %reset, %true : i1
    %4 = comb.and bin %0, %2 : i1
    %5 = comb.and bin %1, %3 : i1

// CHECK:  cover__information_label:
// CHECK-NEXT:    cover property (@(posedge clock)
// CHECK-NEXT:                    eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0 == 4'h0
// CHECK-NEXT:                    & ~reset);{{.*}}
    sv.cover.concurrent posedge %clock, %4 label "cover__information_label"

// CHECK:  assert__label:
// CHECK-NEXT:    assert property (@(posedge clock)
// CHECK-NEXT:                     eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0 == 4'h0
// CHECK-NEXT:                     & ~reset)
// CHECK-NEXT:    else $error("assert failed");	{{.*}}
    sv.assert.concurrent posedge %clock, %5 label "assert__label" message "assert failed"
}

