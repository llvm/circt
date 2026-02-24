
// Generated header for esi_system types.
#pragma once

#include <cstdint>
#include <any>
#include <string_view>

namespace esi_system {
#pragma pack(push, 1)

struct _struct_a_ui16_b_si8 {
  int8_t b : 8;
  uint16_t a : 16;

  static constexpr std::string_view _ESI_ID = "!hw.struct<a: ui16, b: si8>";
};

using ArgStruct = _struct_a_ui16_b_si8;

struct _struct_p_ui8_q_si8_r__hw_array_2xui8_ {
  uint8_t r[2];
  int8_t q : 8;
  uint8_t p : 8;

  static constexpr std::string_view _ESI_ID = "!hw.struct<p: ui8, q: si8, r: !hw.array<2xui8>>";
};

using OddInner = _struct_p_ui8_q_si8_r__hw_array_2xui8_;

struct _struct_a_ui12_b_si7_inner__hw_typealias__pycde__OddInner___hw_struct_p__ui8__q__si8__r___hw_array_2xui8___ {
  OddInner inner;
  int8_t b : 7;
  uint16_t a : 12;

  static constexpr std::string_view _ESI_ID = "!hw.struct<a: ui12, b: si7, inner: !hw.typealias<@pycde::@OddInner, !hw.struct<p: ui8, q: si8, r: !hw.array<2xui8>>>>";
};

using OddStruct = _struct_a_ui12_b_si7_inner__hw_typealias__pycde__OddInner___hw_struct_p__ui8__q__si8__r___hw_array_2xui8___;

using ResultArray = int8_t[2];

struct _struct_x_si8_y_si8 {
  int8_t y : 8;
  int8_t x : 8;

  static constexpr std::string_view _ESI_ID = "!hw.struct<x: si8, y: si8>";
};

using ResultStruct = _struct_x_si8_y_si8;

struct _struct_address_ui5_data_i64 {
  uint64_t data : 64;
  uint8_t address : 5;

  static constexpr std::string_view _ESI_ID = "!hw.struct<address: ui5, data: i64>";
};

struct _struct_address_ui64_length_ui32_tag_ui8 {
  uint8_t tag : 8;
  uint32_t length : 32;
  uint64_t address : 64;

  static constexpr std::string_view _ESI_ID = "!hw.struct<address: ui64, length: ui32, tag: ui8>";
};

struct _struct_address_ui64_tag_ui8 {
  uint8_t tag : 8;
  uint64_t address : 64;

  static constexpr std::string_view _ESI_ID = "!hw.struct<address: ui64, tag: ui8>";
};

struct _struct_address_ui64_tag_ui8_data__esi_any {
  std::any data;
  uint8_t tag : 8;
  uint64_t address : 64;

  static constexpr std::string_view _ESI_ID = "!hw.struct<address: ui64, tag: ui8, data: !esi.any>";
};

struct _struct_address_ui64_tag_ui8_data_i64_valid_bytes_i8 {
  uint8_t valid_bytes : 8;
  uint64_t data : 64;
  uint8_t tag : 8;
  uint64_t address : 64;

  static constexpr std::string_view _ESI_ID = "!hw.struct<address: ui64, tag: ui8, data: i64, valid_bytes: i8>";
};

struct _struct_tag_ui8_data__esi_any {
  std::any data;
  uint8_t tag : 8;

  static constexpr std::string_view _ESI_ID = "!hw.struct<tag: ui8, data: !esi.any>";
};

struct _struct_tag_ui8_data_i64 {
  uint64_t data : 64;
  uint8_t tag : 8;

  static constexpr std::string_view _ESI_ID = "!hw.struct<tag: ui8, data: i64>";
};

struct _struct_write_i1_offset_ui32_data_i64 {
  uint64_t data : 64;
  uint32_t offset : 32;
  bool write : 1;

  static constexpr std::string_view _ESI_ID = "!hw.struct<write: i1, offset: ui32, data: i64>";
};


#pragma pack(pop)
} // namespace esi_system
