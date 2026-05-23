//===- BitAccess.h - Bit-level helpers for generated types ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
// Helpers used by `esiaccel.codegen`-generated headers to read and write
// arbitrary-width integer fields out of a contiguous wire-byte buffer. The
// runtime emits each struct/union as a `std::array<uint8_t, N>` matching the
// on-wire bit layout exactly, and these helpers handle the shift/mask work
// (including signed sign-extension) without depending on C++ bit-fields,
// which have implementation-defined layout that differs between the Itanium
// ABI (GCC/Clang) and MSVC.
//
// `BitOffset` and `Width` are non-type template parameters so the per-field
// constants from the generated code (a) participate in `static_assert`
// checks against `Storage`, and (b) let the optimiser fully unroll the
// per-bit loop.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_BITACCESS_H
#define ESI_BITACCESS_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace esi {
namespace detail {

/// Read `Width` bits starting at `BitOffset` (LSB-first within `bytes`) and
/// return them zero-extended in `Storage`. `Storage` must be an unsigned
/// integer type wide enough to hold `Width` bits.
template <typename Storage, std::size_t BitOffset, std::size_t Width>
inline constexpr Storage readUnsignedBits(const uint8_t *bytes) {
  static_assert(std::is_unsigned<Storage>::value,
                "readUnsignedBits Storage must be unsigned");
  static_assert(Width != 0, "readUnsignedBits Width must be non-zero");
  static_assert(sizeof(Storage) * 8 >= Width,
                "readUnsignedBits Storage is narrower than Width");
  Storage result = 0;
  for (std::size_t i = 0; i < Width; ++i) {
    std::size_t src = BitOffset + i;
    Storage bit = static_cast<Storage>((bytes[src >> 3] >> (src & 7)) & 1u);
    result |= static_cast<Storage>(bit << i);
  }
  return result;
}

/// Read `Width` bits starting at `BitOffset` (LSB-first) and sign-extend
/// into the signed integer type `Signed`. `Signed` must be at least `Width`
/// bits wide.
template <typename Signed, std::size_t BitOffset, std::size_t Width>
inline constexpr Signed readSignedBits(const uint8_t *bytes) {
  static_assert(std::is_signed<Signed>::value,
                "readSignedBits Signed must be signed");
  static_assert(Width != 0, "readSignedBits Width must be non-zero");
  static_assert(sizeof(Signed) * 8 >= Width,
                "readSignedBits Signed is narrower than Width");
  using Unsigned = typename std::make_unsigned<Signed>::type;
  Unsigned u = readUnsignedBits<Unsigned, BitOffset, Width>(bytes);
  if constexpr (Width < sizeof(Unsigned) * 8) {
    // Branch-free sign extend: `(u ^ signBit) - signBit` flips the sign
    // bit and subtracts it back, leaving the high bits all-ones when the
    // sign bit was set and unchanged otherwise. Avoids relying on the
    // arithmetic-shift behaviour of signed right-shift.
    constexpr Unsigned signBit = static_cast<Unsigned>(1) << (Width - 1);
    u = static_cast<Unsigned>((u ^ signBit) - signBit);
  }
  return static_cast<Signed>(u);
}

/// Write the low `Width` bits of `value` into `bytes` starting at
/// `BitOffset` (LSB-first). Other bits in the affected bytes are preserved.
template <typename Storage, std::size_t BitOffset, std::size_t Width>
inline constexpr void writeUnsignedBits(uint8_t *bytes, Storage value) {
  static_assert(std::is_unsigned<Storage>::value,
                "writeUnsignedBits Storage must be unsigned");
  static_assert(Width != 0, "writeUnsignedBits Width must be non-zero");
  static_assert(sizeof(Storage) * 8 >= Width,
                "writeUnsignedBits Storage is narrower than Width");
  for (std::size_t i = 0; i < Width; ++i) {
    std::size_t dst = BitOffset + i;
    std::size_t shift = dst & 7;
    uint8_t mask = static_cast<uint8_t>(1u << shift);
    uint8_t bit = static_cast<uint8_t>((value >> i) & static_cast<Storage>(1));
    bytes[dst >> 3] =
        static_cast<uint8_t>((bytes[dst >> 3] & static_cast<uint8_t>(~mask)) |
                             static_cast<uint8_t>(bit << shift));
  }
}

/// Convenience overload for signed values. Reinterprets the value as
/// unsigned (two's complement) and writes the low `Width` bits.
template <typename Signed, std::size_t BitOffset, std::size_t Width>
inline constexpr void writeSignedBits(uint8_t *bytes, Signed value) {
  static_assert(std::is_signed<Signed>::value,
                "writeSignedBits Signed must be signed");
  static_assert(Width != 0, "writeSignedBits Width must be non-zero");
  static_assert(sizeof(Signed) * 8 >= Width,
                "writeSignedBits Signed is narrower than Width");
  using Unsigned = typename std::make_unsigned<Signed>::type;
  writeUnsignedBits<Unsigned, BitOffset, Width>(bytes,
                                                static_cast<Unsigned>(value));
}

/// Copy `Width` bits out of `src` starting at `BitOffset` into `dst`
/// packed LSB-first from bit 0. Used to embed an aggregate (struct, union,
/// or array) at an arbitrary bit position inside a parent wire buffer:
/// the inner type already stores its bits LSB-first in `dst`, so the
/// destination bits land at the same positions they'd occupy if the inner
/// was read out at byte alignment. `dst` must be zero-initialised on entry
/// (the helper only sets `1` bits) and must hold at least
/// `ceil(Width / 8)` bytes.
template <std::size_t BitOffset, std::size_t Width>
inline constexpr void copyBitsIn(const uint8_t *src, uint8_t *dst) {
  static_assert(Width != 0, "copyBitsIn Width must be non-zero");
  for (std::size_t i = 0; i < Width; ++i) {
    std::size_t s = BitOffset + i;
    uint8_t bit = static_cast<uint8_t>((src[s >> 3] >> (s & 7)) & 1u);
    dst[i >> 3] = static_cast<uint8_t>(dst[i >> 3] | (bit << (i & 7)));
  }
}

/// Inverse of `copyBitsIn`: take `Width` bits packed LSB-first from bit 0
/// of `src` and write them into `dst` starting at `BitOffset`. Other bits
/// in the affected bytes of `dst` are preserved.
template <std::size_t BitOffset, std::size_t Width>
inline constexpr void copyBitsOut(uint8_t *dst, const uint8_t *src) {
  static_assert(Width != 0, "copyBitsOut Width must be non-zero");
  for (std::size_t i = 0; i < Width; ++i) {
    std::size_t d = BitOffset + i;
    std::size_t shift = d & 7;
    uint8_t mask = static_cast<uint8_t>(1u << shift);
    uint8_t bit = static_cast<uint8_t>((src[i >> 3] >> (i & 7)) & 1u);
    dst[d >> 3] =
        static_cast<uint8_t>((dst[d >> 3] & static_cast<uint8_t>(~mask)) |
                             static_cast<uint8_t>(bit << shift));
  }
}

} // namespace detail
} // namespace esi

#endif // ESI_BITACCESS_H
