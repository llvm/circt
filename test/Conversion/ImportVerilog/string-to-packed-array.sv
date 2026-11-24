// RUN: circt-verilog --import-only --top=top %s | FileCheck %s
// RUN: circt-verilog --import-only --top=top %s | circt-opt --convert-moore-to-core --canonicalize | FileCheck %s --check-prefix=CORE
// REQUIRES: slang
// UNSUPPORTED: valgrind

// An uncalled function must not prevent an otherwise empty module from importing.
module top;
  typedef bit [255:0][7:0] bytes_t;
  // CHECK-LABEL: func.func private @convert(
  // CHECK-SAME: [[VALUE:%.*]]: !moore.string
  // CHECK: [[INT:%.*]] = moore.string_to_int [[VALUE]] : i2048
  // CHECK-NEXT: [[PACKED:%.*]] = moore.sbv_to_packed [[INT]] : array<256 x i8>
  // CHECK-NEXT: return [[PACKED]] : !moore.array<256 x i8>
  // CORE-LABEL: func.func private @convert(
  // CORE: [[INT:%.*]] = sim.string.string_to_int %arg0 : i2048
  // CORE-NEXT: [[PACKED:%.*]] = hw.bitcast [[INT]] : (i2048) -> !hw.array<256xi8>
  // CORE-NEXT: return [[PACKED]] : !hw.array<256xi8>
  function automatic bytes_t convert(string value);
    return bytes_t'(value);
  endfunction
endmodule

typedef bit [3:0][7:0] bytes4_t;
typedef logic [3:0][7:0] logic4_t;

// CHECK-LABEL: func.func private @cast_bits(
// CHECK: [[INT:%.*]] = moore.string_to_int %arg0 : i32
// CHECK-NEXT: [[PACKED:%.*]] = moore.sbv_to_packed [[INT]] : array<4 x i8>
// CHECK-NEXT: return [[PACKED]] : !moore.array<4 x i8>
// CORE-LABEL: func.func private @cast_bits(
// CORE: [[INT:%.*]] = sim.string.string_to_int %arg0 : i32
// CORE-NEXT: [[PACKED:%.*]] = hw.bitcast [[INT]] : (i32) -> !hw.array<4xi8>
// CORE-NEXT: return [[PACKED]] : !hw.array<4xi8>
function automatic bytes4_t cast_bits(string value);
  return bytes4_t'(value);
endfunction

// CHECK-LABEL: func.func private @cast_logic(
// CHECK: [[INT:%.*]] = moore.string_to_int %arg0 : i32
// CHECK-NEXT: [[LOGIC:%.*]] = moore.int_to_logic [[INT]] : i32
// CHECK-NEXT: [[PACKED:%.*]] = moore.sbv_to_packed [[LOGIC]] : array<4 x l8>
// CHECK-NEXT: return [[PACKED]] : !moore.array<4 x l8>
// CORE-LABEL: func.func private @cast_logic(
// CORE: [[INT:%.*]] = sim.string.string_to_int %arg0 : i32
// CORE-NEXT: [[PACKED:%.*]] = hw.bitcast [[INT]] : (i32) -> !hw.array<4xi8>
// CORE-NEXT: return [[PACKED]] : !hw.array<4xi8>
function automatic logic4_t cast_logic(string value);
  return logic4_t'(value);
endfunction

// All string lengths use the destination width. The string conversion supplies
// leading zero padding and discards excess leading bytes; the bitcast preserves
// that ordering. Explicit string casts keep these on the string conversion path.

// CHECK-LABEL: func.func private @empty_string(
// CHECK: moore.string_to_int %{{.*}} : i32
// CHECK-NEXT: moore.sbv_to_packed %{{.*}} : array<4 x i8>
// CORE-LABEL: func.func private @empty_string(
// CORE: [[STR:%.*]] = sim.string.literal ""
// CORE-NEXT: [[INT:%.*]] = sim.string.string_to_int [[STR]] : i32
// CORE-NEXT: [[PACKED:%.*]] = hw.bitcast [[INT]] : (i32) -> !hw.array<4xi8>
// CORE-NEXT: return [[PACKED]] : !hw.array<4xi8>
function automatic bytes4_t empty_string();
  return bytes4_t'(string'(""));
endfunction

// CHECK-LABEL: func.func private @short_string(
// CHECK: moore.string_to_int %{{.*}} : i32
// CHECK-NEXT: moore.sbv_to_packed %{{.*}} : array<4 x i8>
// CORE-LABEL: func.func private @short_string(
// CORE: [[STR:%.*]] = sim.string.literal "AB"
// CORE-NEXT: [[INT:%.*]] = sim.string.string_to_int [[STR]] : i32
// CORE-NEXT: [[PACKED:%.*]] = hw.bitcast [[INT]] : (i32) -> !hw.array<4xi8>
// CORE-NEXT: return [[PACKED]] : !hw.array<4xi8>
function automatic bytes4_t short_string();
  return bytes4_t'(string'("AB"));
endfunction

// CHECK-LABEL: func.func private @exact_string(
// CHECK: moore.string_to_int %{{.*}} : i32
// CHECK-NEXT: moore.sbv_to_packed %{{.*}} : array<4 x i8>
// CORE-LABEL: func.func private @exact_string(
// CORE: [[STR:%.*]] = sim.string.literal "ABCD"
// CORE-NEXT: [[INT:%.*]] = sim.string.string_to_int [[STR]] : i32
// CORE-NEXT: [[PACKED:%.*]] = hw.bitcast [[INT]] : (i32) -> !hw.array<4xi8>
// CORE-NEXT: return [[PACKED]] : !hw.array<4xi8>
function automatic bytes4_t exact_string();
  return bytes4_t'(string'("ABCD"));
endfunction

// CHECK-LABEL: func.func private @long_string(
// CHECK: moore.string_to_int %{{.*}} : i32
// CHECK-NEXT: moore.sbv_to_packed %{{.*}} : array<4 x i8>
// CORE-LABEL: func.func private @long_string(
// CORE: [[STR:%.*]] = sim.string.literal "ABCDE"
// CORE-NEXT: [[INT:%.*]] = sim.string.string_to_int [[STR]] : i32
// CORE-NEXT: [[PACKED:%.*]] = hw.bitcast [[INT]] : (i32) -> !hw.array<4xi8>
// CORE-NEXT: return [[PACKED]] : !hw.array<4xi8>
function automatic bytes4_t long_string();
  return bytes4_t'(string'("ABCDE"));
endfunction

// CHECK-LABEL: func.func private @format_bytes(
// CHECK: [[STR:%.*]] = moore.fstring_to_string %{{.*}}
// CHECK-NEXT: [[INT:%.*]] = moore.string_to_int [[STR]] : i32
// CHECK-NEXT: [[PACKED:%.*]] = moore.sbv_to_packed [[INT]] : array<4 x i8>
// CHECK-NEXT: moore.blocking_assign %arg1, [[PACKED]] : array<4 x i8>
// CHECK: [[STR:%.*]] = moore.fstring_to_string %{{.*}}
// CHECK-NEXT: [[INT:%.*]] = moore.string_to_int [[STR]] : i32
// CHECK-NEXT: [[LOGIC:%.*]] = moore.int_to_logic [[INT]] : i32
// CHECK-NEXT: [[PACKED:%.*]] = moore.sbv_to_packed [[LOGIC]] : array<4 x l8>
// CHECK-NEXT: moore.blocking_assign %arg2, [[PACKED]] : array<4 x l8>
function automatic void format_bytes(string value, ref bytes4_t bits,
                                     ref logic4_t logs);
  $sformat(bits, "%s", value);
  $sformat(logs, "%s", value);
endfunction

typedef enum { ABC, DEFG } names_t;
// CHECK-LABEL: func.func private @enum_name(
// CHECK: [[STR:%.*]] = call @enum.name.names_t(%arg0)
// CHECK-NEXT: [[INT:%.*]] = moore.string_to_int [[STR]] : i32
// CHECK-NEXT: [[PACKED:%.*]] = moore.sbv_to_packed [[INT]] : array<4 x i8>
// CHECK-NEXT: moore.blocking_assign %arg1, [[PACKED]] : array<4 x i8>
function automatic void enum_name(names_t value, ref bytes4_t bytes);
  bytes = bytes4_t'(value.name());
endfunction
