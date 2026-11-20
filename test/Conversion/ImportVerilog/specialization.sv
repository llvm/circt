// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-verilog --ir-moore --record-specialization %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --ir-hw --record-specialization %s | FileCheck %s --check-prefix=HW
// RUN: circt-verilog --ir-hw --record-specialization %s | circt-opt --emit-bytecode | circt-opt | FileCheck %s --check-prefix=HW
// REQUIRES: slang
// UNSUPPORTED: valgrind

// Without `--record-specialization`, no `hw.specialization` attribute is
// emitted and modules with 4-state / aggregate / overridden type parameters
// elaborate normally without errors.
// DEFAULT-NOT: hw.specialization

// DEFAULT-LABEL: moore.module @Plain()
// MOORE-LABEL: moore.module @Plain() attributes {hw.specialization = #hw.specialization<"Plain">}
// HW-LABEL: hw.module @Plain() attributes {hw.specialization = #hw.specialization<"Plain">}
module Plain;
endmodule

// Both the first specialization (@DedupB) and uniquified copies (@DedupB_0)
// record their original pre-uniquified name `"DedupB"`. Non-overridden type
// parameters (`T = bit [W-1:0]`) are omitted.
// MOORE-LABEL: moore.module private @DedupB(in %a : !moore.i32) attributes {hw.specialization = #hw.specialization<"DedupB", [#hw.param.decl<"W": si32 = 32> : si32]>}
// MOORE-LABEL: moore.module private @DedupB_{{[0-9]+}}(in %a : !moore.i16) attributes {hw.specialization = #hw.specialization<"DedupB", [#hw.param.decl<"W": si32 = 16> : si32]>}
// HW-LABEL: hw.module private @DedupB(in %a : i32) attributes {hw.specialization = #hw.specialization<"DedupB", [#hw.param.decl<"W": si32 = 32> : si32]>}
// HW-LABEL: hw.module private @DedupB_{{[0-9]+}}(in %a : i16) attributes {hw.specialization = #hw.specialization<"DedupB", [#hw.param.decl<"W": si32 = 16> : si32]>}
module DedupB #(parameter int W, parameter type T = bit [W-1:0])(input T a);
endmodule

// Signed/negative, unsigned/wide, real/shortreal, string, and implicit string
// literal parameters. Localparams are omitted.
// MOORE-LABEL: moore.module private @ScalarParams() attributes {hw.specialization = #hw.specialization<"ScalarParams", [#hw.param.decl<"S": si32 = -4> : si32, #hw.param.decl<"U": ui8 = 255> : ui8, #hw.param.decl<"WIDE": ui64 = 9223372036854775808> : ui64, #hw.param.decl<"R": f64 = 1.500000e+00> : f64, #hw.param.decl<"SR": f32 = 2.500000e+00> : f32, #hw.param.decl<"STR": none = "hello">, #hw.param.decl<"IMP_STR": none = "world">]>}
// HW-LABEL: hw.module private @ScalarParams() attributes {hw.specialization = #hw.specialization<"ScalarParams", [#hw.param.decl<"S": si32 = -4> : si32, #hw.param.decl<"U": ui8 = 255> : ui8, #hw.param.decl<"WIDE": ui64 = 9223372036854775808> : ui64, #hw.param.decl<"R": f64 = 1.500000e+00> : f64, #hw.param.decl<"SR": f32 = 2.500000e+00> : f32, #hw.param.decl<"STR": none = "hello">, #hw.param.decl<"IMP_STR": none = "world">]>}
module ScalarParams #(
  parameter int S = -4,
  parameter bit [7:0] U = 8'hff,
  parameter logic [63:0] WIDE = 64'h8000_0000_0000_0000,
  parameter real R = 1.5,
  parameter shortreal SR = 2.5,
  parameter string STR = "hello",
  parameter IMP_STR = "world"
);
  localparam int LOCAL_SKIP = 99;
  parameter int BODY_LOCAL = 77;
endmodule

// Non-ANSI body parameter without a header parameter list can be overridden.
// MOORE-LABEL: moore.module private @BodyParam() attributes {hw.specialization = #hw.specialization<"BodyParam", [#hw.param.decl<"P": si32 = 42> : si32]>}
// HW-LABEL: hw.module private @BodyParam() attributes {hw.specialization = #hw.specialization<"BodyParam", [#hw.param.decl<"P": si32 = 42> : si32]>}
module BodyParam;
  parameter int P = 1;
  localparam int LP = 2;
endmodule

// 4-state integer with X/Z (including wide vectors without abbreviation),
// unpacked arrays, unpacked structs, and empty strings.
typedef struct {
  int a;
  logic [3:0] b;
} ustruct_t;

// MOORE-LABEL: moore.module private @VerbatimParams() attributes {hw.specialization = #hw.specialization<"VerbatimParams", [#hw.param.decl<"FOUR": none = #hw.param.verbatim<"4'b10x1">>, #hw.param.decl<"WIDE128": ui128 = 295990755076957304698161171062762229231> : ui128, #hw.param.decl<"WIDE_X": none = #hw.param.verbatim<"72'b11111111xxxxxxxxxxxxxxxxzzzzzzzzzzzzzzzz00000000000000000001001000110100">>, #hw.param.decl<"ARR": none = #hw.param.verbatim<"'{4,8}">>, #hw.param.decl<"US": none = #hw.param.verbatim<"'{5,4'b10x1}">>, #hw.param.decl<"EMPTY_IMP": none = "">, #hw.param.decl<"EMPTY_STR": none = "">]>}
// HW-LABEL: hw.module private @VerbatimParams() attributes {hw.specialization = #hw.specialization<"VerbatimParams", [#hw.param.decl<"FOUR": none = #hw.param.verbatim<"4'b10x1">>, #hw.param.decl<"WIDE128": ui128 = 295990755076957304698161171062762229231> : ui128, #hw.param.decl<"WIDE_X": none = #hw.param.verbatim<"72'b11111111xxxxxxxxxxxxxxxxzzzzzzzzzzzzzzzz00000000000000000001001000110100">>, #hw.param.decl<"ARR": none = #hw.param.verbatim<"'{4,8}">>, #hw.param.decl<"US": none = #hw.param.verbatim<"'{5,4'b10x1}">>, #hw.param.decl<"EMPTY_IMP": none = "">, #hw.param.decl<"EMPTY_STR": none = "">]>}
module VerbatimParams #(
  parameter logic [3:0] FOUR = 4'b10x1,
  parameter logic [127:0] WIDE128 = 128'hdead_beef_cafe_babe_0123_4567_89ab_cdef,
  parameter logic [71:0] WIDE_X = 72'hff_xxxx_zzzz_0000_1234,
  parameter int ARR[2] = '{4, 8},
  parameter ustruct_t US = '{a: 5, b: 4'b10x1},
  parameter EMPTY_IMP = "",
  parameter string EMPTY_STR = ""
);
endmodule

typedef logic [7:0] my_byte_t;

package spec_pkg;
  typedef struct packed { logic a; logic [2:0] b; } pkt_t;
endpackage

// Overridden type parameters fall back to verbatim syntax for built-in,
// unit-scope, and package-scope types. Module-local typedefs emit an unvalued
// `ParamDeclAttr` because hierarchical paths (`Top.local_t`) are not legal
// SystemVerilog type overrides.
// MOORE-LABEL: moore.module private @TypeParamChild() attributes {hw.specialization = #hw.specialization<"TypeParamChild", [#hw.param.decl<"T": none = #hw.param.verbatim<"byte">>]>}
// MOORE-LABEL: moore.module private @TypeParamChild_{{[0-9]+}}() attributes {hw.specialization = #hw.specialization<"TypeParamChild", [#hw.param.decl<"T": none = #hw.param.verbatim<"my_byte_t">>]>}
// MOORE-LABEL: moore.module private @TypeParamChild_{{[0-9]+}}() attributes {hw.specialization = #hw.specialization<"TypeParamChild", [#hw.param.decl<"T": none = #hw.param.verbatim<"spec_pkg::pkt_t">>]>}
// MOORE-LABEL: moore.module private @TypeParamChild_{{[0-9]+}}() attributes {hw.specialization = #hw.specialization<"TypeParamChild", [#hw.param.decl<"T": none>]>}
// HW-LABEL: hw.module private @TypeParamChild() attributes {hw.specialization = #hw.specialization<"TypeParamChild", [#hw.param.decl<"T": none = #hw.param.verbatim<"byte">>]>}
// HW-LABEL: hw.module private @TypeParamChild_{{[0-9]+}}() attributes {hw.specialization = #hw.specialization<"TypeParamChild", [#hw.param.decl<"T": none = #hw.param.verbatim<"my_byte_t">>]>}
// HW-LABEL: hw.module private @TypeParamChild_{{[0-9]+}}() attributes {hw.specialization = #hw.specialization<"TypeParamChild", [#hw.param.decl<"T": none = #hw.param.verbatim<"spec_pkg::pkt_t">>]>}
// HW-LABEL: hw.module private @TypeParamChild_{{[0-9]+}}() attributes {hw.specialization = #hw.specialization<"TypeParamChild", [#hw.param.decl<"T": none>]>}
module TypeParamChild #(
  parameter type T = int
);
endmodule

// Split multi-ports mark the boundary as altered.
// MOORE-LABEL: moore.module private @MultiPort(in %a : !moore.l1, in %b : !moore.l1) attributes {hw.specialization = #hw.specialization<"MultiPort", boundaryAltered>}
// HW-LABEL: hw.module private @MultiPort(in %a : i1, in %b : i1) attributes {hw.specialization = #hw.specialization<"MultiPort", boundaryAltered>}
module MultiPort(.p({a, b}));
  input logic a;
  input logic b;
endmodule

// Flattened interface ports mark the boundary as altered.
interface SimpleIf;
  logic sig;
  modport consumer(input sig);
endinterface

// MOORE-LABEL: moore.module private @IfaceConsumer(in %bus_sig : !moore.l1) attributes {hw.specialization = #hw.specialization<"IfaceConsumer", [#hw.param.decl<"W": si32 = 1> : si32], boundaryAltered>}
// HW-LABEL: hw.module private @IfaceConsumer(in %bus_sig : i1) attributes {hw.specialization = #hw.specialization<"IfaceConsumer", [#hw.param.decl<"W": si32 = 1> : si32], boundaryAltered>}
module IfaceConsumer #(parameter int W = 1)(SimpleIf.consumer bus);
endmodule

// Lifted hierarchical references (XMRs) mark the boundary as altered.
// MOORE-LABEL: moore.module private @XmrChild({{.*}}) attributes {hw.specialization = #hw.specialization<"XmrChild", boundaryAltered>}
// HW-LABEL: hw.module private @XmrChild({{.*}}) attributes {hw.specialization = #hw.specialization<"XmrChild", boundaryAltered>}
module XmrChild;
  logic read_x;
  assign read_x = Top.shared_sig;
endmodule

// DEFAULT-NOT: hw.specialization
// DEFAULT-LABEL: moore.module @Top()
// DEFAULT-NOT: hw.specialization
// MOORE-LABEL: moore.module @Top() attributes {hw.specialization = #hw.specialization<"Top">}
// HW-LABEL: hw.module @Top() attributes {hw.specialization = #hw.specialization<"Top">}
module Top;
  typedef struct packed { logic x; } local_t;

  int a;
  shortint b;
  logic shared_sig;
  SimpleIf bus();

  DedupB #(32) b0(a);
  DedupB #(16) b1(b);
  ScalarParams sp();
  BodyParam #(42) bp();
  VerbatimParams vp();
  TypeParamChild #(.T(byte)) tc();
  TypeParamChild #(.T(my_byte_t)) tc_alias();
  TypeParamChild #(.T(spec_pkg::pkt_t)) tc_pkg();
  TypeParamChild #(.T(local_t)) tc_local();
  MultiPort mp(.p({shared_sig, shared_sig}));
  IfaceConsumer ic(bus);
  XmrChild xc();
endmodule
