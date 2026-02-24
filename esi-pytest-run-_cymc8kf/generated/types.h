
// Generated header for esi_system types.
#pragma once

#include <cstdint>
#include <any>
#include <string_view>

namespace esi_system {
#pragma pack(push, 1)

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
