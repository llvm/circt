// RUN: circt-opt %s --lower-sim-to-sv --export-verilog -o %t.mlir | FileCheck %s

// Signless integers retain native signed atoms at standard widths and
// unsigned bit vectors elsewhere. Explicit signedness is preserved.
module attributes {circt.loweringOptions = "disallowPortDeclSharing"} {
  // CHECK-LABEL: import "DPI-C" context function void matrix(
  // CHECK: input bit signed s1
  // CHECK: output bit u1
  // CHECK: inout bit l1
  // CHECK: input bit signed [6:0] s7
  // CHECK: output bit [6:0] u7
  // CHECK: inout bit [6:0] l7
  // CHECK: input byte s8
  // CHECK: output byte unsigned u8
  // CHECK: inout byte l8
  // CHECK: input shortint s16
  // CHECK: output shortint unsigned u16
  // CHECK: inout shortint l16
  // CHECK: input int s32
  // CHECK: output int unsigned u32
  // CHECK: inout int l32
  // CHECK: input longint s64
  // CHECK: output longint unsigned u64
  // CHECK: inout longint l64
  // CHECK: input bit signed [64:0] s65
  // CHECK: output bit [64:0] u65
  // CHECK: inout bit [64:0] l65
  sim.func.dpi @matrix(in %s1: si1,
      out u1: ui1,
      inout %l1: i1,
      in %s7: si7,
      out u7: ui7,
      inout %l7: i7,
      in %s8: si8,
      out u8: ui8,
      inout %l8: i8,
      in %s16: si16,
      out u16: ui16,
      inout %l16: i16,
      in %s32: si32,
      out u32: ui32,
      inout %l32: i32,
      in %s64: si64,
      out u64: ui64,
      inout %l64: i64,
      in %s65: si65,
      out u65: ui65,
      inout %l65: i65)
  hw.module @Top() attributes {emit.fragments = [@matrix_dpi_import_fragument]} {
    hw.output
  }

  // Native SV imports use the same type printer.
  // CHECK-LABEL: import "DPI-C" context function void native(
  // CHECK: output byte unsigned writeData
  sv.func private @native(out writeData: ui8)
  sv.func.dpi.import @native

  // CHECK-LABEL: import "DPI-C" context function bit return_i1();
  sv.func private @return_i1(out result: i1 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_i1
  // CHECK-LABEL: import "DPI-C" context function bit signed return_si1();
  sv.func private @return_si1(out result: si1 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_si1
  // CHECK-LABEL: import "DPI-C" context function bit return_ui1();
  sv.func private @return_ui1(out result: ui1 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_ui1
  // CHECK-LABEL: import "DPI-C" context function byte return_i8();
  sv.func private @return_i8(out result: i8 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_i8
  // CHECK-LABEL: import "DPI-C" context function byte return_si8();
  sv.func private @return_si8(out result: si8 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_si8
  // CHECK-LABEL: import "DPI-C" context function byte unsigned return_ui8();
  sv.func private @return_ui8(out result: ui8 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_ui8
  // CHECK-LABEL: import "DPI-C" context function shortint return_i16();
  sv.func private @return_i16(out result: i16 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_i16
  // CHECK-LABEL: import "DPI-C" context function shortint return_si16();
  sv.func private @return_si16(out result: si16 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_si16
  // CHECK-LABEL: import "DPI-C" context function shortint unsigned return_ui16();
  sv.func private @return_ui16(out result: ui16 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_ui16
  // CHECK-LABEL: import "DPI-C" context function int return_i32();
  sv.func private @return_i32(out result: i32 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_i32
  // CHECK-LABEL: import "DPI-C" context function int return_si32();
  sv.func private @return_si32(out result: si32 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_si32
  // CHECK-LABEL: import "DPI-C" context function int unsigned return_ui32();
  sv.func private @return_ui32(out result: ui32 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_ui32
  // CHECK-LABEL: import "DPI-C" context function longint return_i64();
  sv.func private @return_i64(out result: i64 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_i64
  // CHECK-LABEL: import "DPI-C" context function longint return_si64();
  sv.func private @return_si64(out result: si64 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_si64
  // CHECK-LABEL: import "DPI-C" context function longint unsigned return_ui64();
  sv.func private @return_ui64(out result: ui64 {sv.func.explicitly_returned})
  sv.func.dpi.import @return_ui64

  // Packed dimensions suppress native atoms; typedefs remain four-state.
  // CHECK: typedef logic signed [7:0] SignedByte;
  // CHECK: typedef logic [7:0] UnsignedByte;
  hw.type_scope @Types {
    hw.typedecl @SignedByte : si8
    hw.typedecl @UnsignedByte : ui8
  }
  // CHECK-LABEL: import "DPI-C" context function void aggregates(
  // CHECK: input bit signed [1:0][7:0] packed_s
  // CHECK: input bit [1:0][7:0] packed_u
  // CHECK: input bit [1:0][7:0] packed_l
  // CHECK: input byte unsigned unpacked_u[0:1]
  // CHECK: input byte unsigned open_u[]
  // CHECK: input SignedByte signed_alias
  // CHECK: input UnsignedByte unsigned_alias
  sv.func private @aggregates(in %packed_s: !hw.array<2xsi8>,
      in %packed_u: !hw.array<2xui8>, in %packed_l: !hw.array<2xi8>,
      in %unpacked_u: !hw.uarray<2xui8>, in %open_u: !sv.open_uarray<ui8>,
      in %signed_alias: !hw.typealias<@Types::@SignedByte, si8>,
      in %unsigned_alias: !hw.typealias<@Types::@UnsignedByte, ui8>)
  sv.func.dpi.import @aggregates

}
