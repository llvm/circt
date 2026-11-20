// RUN: circt-opt %s -export-verilog | FileCheck %s

// Native DPI atoms retain their signed defaults for signless integers.
// CHECK-LABEL: import "DPI-C" context function void dpi_i8(
// CHECK-NEXT: input byte value
// CHECK-NEXT: );
sv.func private @dpi_i8(in %value: i8)
sv.func.dpi.import @dpi_i8

// CHECK-LABEL: import "DPI-C" context function void dpi_si8(
// CHECK-NEXT: input byte value
// CHECK-NEXT: );
sv.func private @dpi_si8(in %value: si8)
sv.func.dpi.import @dpi_si8

// CHECK-LABEL: import "DPI-C" context function void dpi_ui8(
// CHECK-NEXT: input byte unsigned value
// CHECK-NEXT: );
sv.func private @dpi_ui8(in %value: ui8)
sv.func.dpi.import @dpi_ui8

// CHECK-LABEL: import "DPI-C" context function void dpi_i16(
// CHECK-NEXT: input shortint value
// CHECK-NEXT: );
sv.func private @dpi_i16(in %value: i16)
sv.func.dpi.import @dpi_i16

// CHECK-LABEL: import "DPI-C" context function void dpi_si16(
// CHECK-NEXT: input shortint value
// CHECK-NEXT: );
sv.func private @dpi_si16(in %value: si16)
sv.func.dpi.import @dpi_si16

// CHECK-LABEL: import "DPI-C" context function void dpi_ui16(
// CHECK-NEXT: input shortint unsigned value
// CHECK-NEXT: );
sv.func private @dpi_ui16(in %value: ui16)
sv.func.dpi.import @dpi_ui16

// CHECK-LABEL: import "DPI-C" context function void dpi_i32(
// CHECK-NEXT: input int value
// CHECK-NEXT: );
sv.func private @dpi_i32(in %value: i32)
sv.func.dpi.import @dpi_i32

// CHECK-LABEL: import "DPI-C" context function void dpi_si32(
// CHECK-NEXT: input int value
// CHECK-NEXT: );
sv.func private @dpi_si32(in %value: si32)
sv.func.dpi.import @dpi_si32

// CHECK-LABEL: import "DPI-C" context function void dpi_ui32(
// CHECK-NEXT: input int unsigned value
// CHECK-NEXT: );
sv.func private @dpi_ui32(in %value: ui32)
sv.func.dpi.import @dpi_ui32

// CHECK-LABEL: import "DPI-C" context function void dpi_i64(
// CHECK-NEXT: input longint value
// CHECK-NEXT: );
sv.func private @dpi_i64(in %value: i64)
sv.func.dpi.import @dpi_i64

// CHECK-LABEL: import "DPI-C" context function void dpi_si64(
// CHECK-NEXT: input longint value
// CHECK-NEXT: );
sv.func private @dpi_si64(in %value: si64)
sv.func.dpi.import @dpi_si64

// CHECK-LABEL: import "DPI-C" context function void dpi_ui64(
// CHECK-NEXT: input longint unsigned value
// CHECK-NEXT: );
sv.func private @dpi_ui64(in %value: ui64)
sv.func.dpi.import @dpi_ui64

// Non-native widths use packed vectors with explicit signedness.
// CHECK-LABEL: import "DPI-C" context function void dpi_i7(
// CHECK-NEXT: input bit [6:0] value
// CHECK-NEXT: );
sv.func private @dpi_i7(in %value: i7)
sv.func.dpi.import @dpi_i7

// CHECK-LABEL: import "DPI-C" context function void dpi_si7(
// CHECK-NEXT: input bit signed [6:0] value
// CHECK-NEXT: );
sv.func private @dpi_si7(in %value: si7)
sv.func.dpi.import @dpi_si7

// CHECK-LABEL: import "DPI-C" context function void dpi_ui7(
// CHECK-NEXT: input bit [6:0] value
// CHECK-NEXT: );
sv.func private @dpi_ui7(in %value: ui7)
sv.func.dpi.import @dpi_ui7

// CHECK-LABEL: import "DPI-C" context function void dpi_si1(
// CHECK-NEXT: input bit signed value
// CHECK-NEXT: );
sv.func private @dpi_si1(in %value: si1)
sv.func.dpi.import @dpi_si1

// Typedefs retain packed storage, including signed packed array elements.
hw.type_scope @types {
  // CHECK: typedef logic signed signed_scalar;
  hw.typedecl @signed_scalar : si1
  // CHECK: typedef logic signed [31:0] signed_vector;
  hw.typedecl @signed_vector : si32
  // CHECK: typedef logic [31:0] unsigned_vector;
  hw.typedecl @unsigned_vector : ui32
  // CHECK: typedef logic [31:0] signless_vector;
  hw.typedecl @signless_vector : i32
  // CHECK: typedef logic signed [1:0][31:0] signed_array;
  hw.typedecl @signed_array : !hw.array<2xsi32>
}
