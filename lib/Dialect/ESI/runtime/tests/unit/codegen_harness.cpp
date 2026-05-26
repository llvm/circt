// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Round-trip harness driven by `tests/unit/test_codegen.py`. The Python
// fixture builds a small manifest, runs `esiaccel.codegen` on it, and
// compiles this file against the resulting `types.h`. The harness asserts
// the generated accessors round-trip user-visible values *and* that the
// underlying wire bytes match the manifest's bit-packed layout exactly.
//
// Reviewing this file is the easiest way to see what the C++ codegen
// promises end-to-end; if anything here changes the runtime semantics
// (rather than just the textual emission), this harness is what will
// catch it.

#include "codegen_harness/types.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ranges>
#include <type_traits>
#include <vector>

#include "esi/Values.h"

using namespace esi_system;

namespace {

// Reinterpret `value` as a `std::array<uint8_t, sizeof(T)>` view of its
// wire bytes. Used to assert the underlying byte layout of generated
// types — the only safe API for that is `reinterpret_cast<uint8_t *>`,
// which the runtime already uses internally via `MessageData::from()`.
template <typename T>
std::array<uint8_t, sizeof(T)> wireBytes(const T &value) {
  std::array<uint8_t, sizeof(T)> out{};
  const auto *src = reinterpret_cast<const uint8_t *>(&value);
  for (std::size_t i = 0; i < sizeof(T); ++i)
    out[i] = src[i];
  return out;
}

template <typename T, std::size_t N>
void expectBytes(const T &value, const std::array<uint8_t, N> &expected,
                 const char *label) {
  static_assert(sizeof(T) == N, "wire-byte assertion size mismatch");
  auto got = wireBytes(value);
  for (std::size_t i = 0; i < N; ++i) {
    if (got[i] != expected[i]) {
      std::fprintf(stderr,
                   "%s: byte %zu mismatch: got 0x%02x expected 0x%02x\n", label,
                   i, got[i], expected[i]);
      std::fprintf(stderr, "  got     :");
      for (auto b : got)
        std::fprintf(stderr, " %02x", b);
      std::fprintf(stderr, "\n  expected:");
      for (auto b : expected)
        std::fprintf(stderr, " %02x", b);
      std::fprintf(stderr, "\n");
      std::abort();
    }
  }
}

// ---------------------------------------------------------------------------
// Path A — 8/16/32/64-bit byte-aligned integers
// ---------------------------------------------------------------------------

void testStandardWidthUnsigned() {
  StdU s;
  s.u8(0xAB).u16(0xCAFE).u32(0xDEADBEEFu).u64(0x0123456789ABCDEFull);
  assert(s.u8() == 0xAB);
  assert(s.u16() == 0xCAFE);
  assert(s.u32() == 0xDEADBEEFu);
  assert(s.u64() == 0x0123456789ABCDEFull);

  // Wire order is reverse of declaration: u64 at byte 0, u32 at byte 8,
  // u16 at byte 12, u8 at byte 14. Each value is little-endian within
  // its own bytes.
  expectBytes(s,
              std::array<uint8_t, 15>{
                  // u64 = 0x0123456789ABCDEF, little-endian:
                  0xEF,
                  0xCD,
                  0xAB,
                  0x89,
                  0x67,
                  0x45,
                  0x23,
                  0x01,
                  // u32 = 0xDEADBEEF, little-endian:
                  0xEF,
                  0xBE,
                  0xAD,
                  0xDE,
                  // u16 = 0xCAFE, little-endian:
                  0xFE,
                  0xCA,
                  // u8 = 0xAB:
                  0xAB,
              },
              "StdU");

  // Default construction zero-initialises the wire bytes.
  StdU empty;
  expectBytes(
      empty,
      std::array<uint8_t, 15>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      "StdU empty");

  // Logical-order ctor matches manifest field order.
  StdU cons(0xAB, 0xCAFE, 0xDEADBEEFu, 0x0123456789ABCDEFull);
  assert(wireBytes(cons) == wireBytes(s));
}

void testStandardWidthSigned() {
  StdS s;
  s.s8(-7).s16(-1000).s32(-1234567).s64(INT64_MIN);
  assert(s.s8() == -7);
  assert(s.s16() == -1000);
  assert(s.s32() == -1234567);
  assert(s.s64() == INT64_MIN);

  // Default-construct a fresh value to confirm round-trip without the
  // chained setter; check INT64_MAX as well to verify both ends of the
  // range.
  StdS pos(127, 32767, 2147483647, INT64_MAX);
  assert(pos.s8() == 127);
  assert(pos.s64() == INT64_MAX);
}

// ---------------------------------------------------------------------------
// Path B — byte-aligned non-standard width (e.g. i24, i48)
// ---------------------------------------------------------------------------

void testByteAlignedOddWidthUnsigned() {
  OddU s;
  s.u24(0xABCDEFu);
  assert(s.u24() == 0xABCDEFu);
  // ui24 is 3 bytes little-endian on the wire.
  expectBytes(s, std::array<uint8_t, 3>{0xEF, 0xCD, 0xAB}, "OddU");
}

void testByteAlignedOddWidthSigned() {
  OddS s;
  s.s24(-1);
  assert(s.s24() == -1);
  expectBytes(s, std::array<uint8_t, 3>{0xFF, 0xFF, 0xFF}, "OddS -1");

  s.s24(-8388608); // si24 min
  assert(s.s24() == -8388608);
  expectBytes(s, std::array<uint8_t, 3>{0x00, 0x00, 0x80}, "OddS min");

  s.s24(8388607); // si24 max
  assert(s.s24() == 8388607);
  expectBytes(s, std::array<uint8_t, 3>{0xFF, 0xFF, 0x7F}, "OddS max");

  // Negative value with all three bytes non-zero. Round-trip + bit pattern.
  s.s24(-1234567);
  assert(s.s24() == -1234567);
}

// ---------------------------------------------------------------------------
// Path C — sub-byte alignment (uses the BitAccess helpers)
// ---------------------------------------------------------------------------

void testSubByteUnsigned() {
  SubU s;
  s.u3(0b101).u12(0xABC);
  assert(s.u3() == 0b101);
  assert(s.u12() == 0xABC);

  // Wire layout: u12 (low 12 bits) then u3 (next 3 bits), tightly packed.
  // bit 0..11  = 0xABC -> bytes 0=0xBC, byte 1 low nibble = 0xA, high
  //                      nibble starts u3
  // bit 12..14 = 0b101 -> byte 1 high 3 bits = 0b101 << 4 = 0x50
  // total 15 bits in 2 bytes: byte 0 = 0xBC, byte 1 = 0x5A.
  expectBytes(s, std::array<uint8_t, 2>{0xBC, 0x5A}, "SubU");
}

void testSubByteSigned() {
  SubS s;
  s.s5(-3).s7(-50);
  assert(s.s5() == -3);
  assert(s.s7() == -50);

  // s5 = -3 (5-bit two's complement = 0x1D); s7 = -50 (7-bit = 0x4E).
  // Wire (reversed): s7 in bit 0..6, s5 in bit 7..11.
  // bit 0..6 = 0b1001110 (0x4E)
  // bit 7    = s5 bit 0 (LSB of -3 = 1)
  // bit 8..11 = s5 bit 1..4
  // s5 = 0x1D = 0b11101 -> bits 7..11 = 11101
  // byte 0 = (0x4E) | (1 << 7) = 0xCE
  // byte 1 = 0b00001110 = 0x0E
  expectBytes(s, std::array<uint8_t, 2>{0xCE, 0x0E}, "SubS");
}

void testBoolField() {
  BoolField f;
  f.flag(true).pad(0x5A);
  assert(f.flag() == true);
  assert(f.pad() == 0x5A);
  // flag is at bit 7 (after the ui7 pad, which is wire-reversed first).
  // byte 0 = 0x80 | 0x5A = 0xDA
  expectBytes(f, std::array<uint8_t, 1>{0xDA}, "BoolField true");

  f.flag(false);
  expectBytes(f, std::array<uint8_t, 1>{0x5A}, "BoolField false");
}

// ---------------------------------------------------------------------------
// Nested struct fields (aggregate accessor)
// ---------------------------------------------------------------------------

void testNestedStruct() {
  Inner in;
  in.x(0x12).y(0x34);

  Outer o;
  o.label(0xAB).inner(in);
  assert(o.label() == 0xAB);
  assert(o.inner().x() == 0x12);
  assert(o.inner().y() == 0x34);

  // Wire order: inner first (in declaration-reversed order), then label.
  // Inner has y at byte 0, x at byte 1 (reversed inside inner too).
  expectBytes(o, std::array<uint8_t, 3>{0x34, 0x12, 0xAB}, "Outer");

  // Modifying a returned inner does NOT alias back into the outer
  // (accessor returns a value, not a reference) — the caller has to
  // hand the modified inner back via `outer.inner(modified)`.
  Outer o2;
  o2.inner(in);
  auto copy = o2.inner();
  copy.x(0xFF);
  assert(o2.inner().x() == 0x12); // unchanged
  o2.inner(copy);
  assert(o2.inner().x() == 0xFF);
}

void testNestedStructMisaligned() {
  // `Misaligned` wire layout (LSB-first):
  //   bits  0..2  = tag (ui3)         = 0b101                       (= 5)
  //   bits  3..10 = inner.y (ui8 LSB at outer bit 3) = 0x34
  //   bits 11..18 = inner.x (ui8)                     = 0x12
  //   bits 19..23 = padding (0)
  // Inner is byte-aligned internally (x at inner bit 8, y at inner bit 0
  // with the standard reverse), but the inner aggregate as a whole lands
  // at outer bit 3 — exercising the `copyBitsIn`/`copyBitsOut` path
  // rather than the byte-aligned fast copy.
  MisInner mi;
  mi.x(0x12).y(0x34);

  Misaligned m;
  m.tag(0b101).inner(mi);
  assert(m.tag() == 0b101);
  assert(m.inner().x() == 0x12);
  assert(m.inner().y() == 0x34);

  // Hand-derived bytes from the bit layout above.
  expectBytes(m, std::array<uint8_t, 3>{0xA5, 0x91, 0x00},
              "Misaligned all-set");

  // Setting tag again after writing inner must not disturb inner's bits
  // (the writer is supposed to mask only the tag's bit range).
  m.tag(0b000);
  assert(m.inner().x() == 0x12);
  assert(m.inner().y() == 0x34);
  expectBytes(m, std::array<uint8_t, 3>{0xA0, 0x91, 0x00},
              "Misaligned tag cleared");

  // Setting inner again after writing tag must not disturb tag.
  m.tag(0b101);
  MisInner mi2;
  mi2.x(0xFF).y(0x00);
  m.inner(mi2);
  assert(m.tag() == 0b101);
  assert(m.inner().x() == 0xFF);
  assert(m.inner().y() == 0x00);
  // bits  0..2  = 0b101                = 5
  // bits  3..10 = 0x00                 = 0
  // bits 11..18 = 0xFF                 = all 1s spanning byte 1 high and
  //                                      byte 2 low bits
  // byte 0: tag(3) | y[0..4]<<3 = 0b00000101 = 0x05
  // byte 1: y[5..7] | x[0..4]<<3 = 0 | 0b11111<<3 = 0b11111000 = 0xF8
  // byte 2: x[5..7] | pad = 0b00000111 = 0x07
  expectBytes(m, std::array<uint8_t, 3>{0x05, 0xF8, 0x07},
              "Misaligned inner all-ones");
}

// ---------------------------------------------------------------------------
// Array fields: whole-array and indexed accessors
// ---------------------------------------------------------------------------

void testArrayOfIntegers() {
  Arr4 a;
  a.r({0x10, 0x20, 0x30, 0x40});
  auto r = a.r();
  assert(r[0] == 0x10 && r[1] == 0x20 && r[2] == 0x30 && r[3] == 0x40);
  // Indexed read/write convenience overloads.
  assert(a.r(2) == 0x30);
  a.r(2, 0x99);
  assert(a.r(2) == 0x99);

  // Arrays are laid out element-0-first on the wire (matching the
  // existing `std::array` layout), not reversed.
  expectBytes(a, std::array<uint8_t, 4>{0x10, 0x20, 0x99, 0x40}, "Arr4");
}

// ---------------------------------------------------------------------------
// Unions
// ---------------------------------------------------------------------------

void testUnion() {
  UnionTwo u;
  u.big(0xCAFE);
  assert(u.big() == 0xCAFE);
  // Narrow variant at MSB end: small = high byte of big (little-endian
  // layout means high byte is at byte_offset = union_bytes - 1 = 1).
  assert(u.small() == 0xCA);
  expectBytes(u, std::array<uint8_t, 2>{0xFE, 0xCA}, "Union big");

  u.small(0xA5);
  assert(u.small() == 0xA5);
  // Writing the narrow variant overwrites only its byte slot at the
  // MSB end; the LSB byte retains its previous content.
  expectBytes(u, std::array<uint8_t, 2>{0xFE, 0xA5}, "Union small");
}

// ---------------------------------------------------------------------------
// Window helpers (header / data frame round-trip)
// ---------------------------------------------------------------------------

void testWindowList() {
  std::vector<uint32_t> elements = {0xAAAA0001u, 0xBBBB0002u, 0xCCCC0003u};
  ListWindow win(0xDEAD, elements);

  // Header-field accessor reads the static `tag`. Count comes from the
  // `<list>_count()` accessor and matches the number of data frames.
  assert(win.tag() == 0xDEAD);
  assert(win.items_count() == elements.size());

  // Vector helper materialises the data into a flat std::vector.
  auto got = win.items_vector();
  assert(got.size() == elements.size());
  for (std::size_t i = 0; i < elements.size(); ++i)
    assert(got[i] == elements[i]);

  // Range accessor produces the same values lazily.
  std::size_t i = 0;
  for (uint32_t v : win.items())
    assert(v == elements[i++]);

  // The window splits into three wire segments: header, data, footer.
  assert(win.numSegments() == 3);
  auto header_seg = win.segment(0);
  auto data_seg = win.segment(1);
  auto footer_seg = win.segment(2);
  assert(data_seg.size == elements.size() * sizeof(uint32_t));
  // Header carries `tag` at the MSB end and the count at the LSB end.
  // Layout: count (ui16, 2 bytes) then tag (ui16, 2 bytes) in reversed
  // declaration order; header bytes = [count_lo, count_hi, tag_lo, tag_hi].
  assert(header_seg.size == 4);
  assert(header_seg.data[0] == (static_cast<uint8_t>(elements.size()) & 0xFF));
  assert(header_seg.data[1] == 0); // count high byte (size < 256)
  assert(header_seg.data[2] == 0xAD);
  assert(header_seg.data[3] == 0xDE);
  // Footer is a zero-count header with the same layout.
  assert(footer_seg.size == 4);
  assert(footer_seg.data[0] == 0);
  assert(footer_seg.data[1] == 0);
}

// ---------------------------------------------------------------------------
// Path D — value-class fields (esi::MutableBitVector, esi::Int, esi::UInt)
// ---------------------------------------------------------------------------

// Build a wide MutableBitVector from an arbitrary-length byte buffer
// (LSB-first wire order). Useful for hand-constructing > 64-bit values in
// the harness.
esi::MutableBitVector bvFromBytes(std::initializer_list<uint8_t> bytes,
                                  std::size_t width) {
  std::vector<uint8_t> storage(bytes.begin(), bytes.end());
  // The MutableBitVector(vector<byte>&&, width) ctor requires the storage
  // size to cover `width` bits. Pad with zeros if the caller supplied
  // fewer bytes than that.
  std::size_t need = (width + 7) / 8;
  if (storage.size() < need)
    storage.resize(need, 0);
  return esi::MutableBitVector(std::move(storage), width);
}

void testWideUnsigned() {
  // WideU has two byte-aligned wide-int fields: ui128 at the LSB end
  // (bits 0..127), ui96 at bits 128..223. Total 224 bits = 28 bytes.
  WideU s;
  // Construct a 96-bit value: low half = 0x0123456789ABCDEF,
  // high half = 0xCAFEBABEu (32 bits worth of upper bytes).
  auto u96 = bvFromBytes(
      {
          0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, // low 64 bits
          0xBE, 0xBA, 0xFE, 0xCA,                         // bits 64..95
      },
      96);
  auto u128 = bvFromBytes(
      {
          0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, // low 64 bits
          0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, // high 64 bits
      },
      128);

  s.u96(u96).u128(u128);

  // Round-trip: read the field back out and compare byte-for-byte.
  auto got96 = s.u96();
  assert(got96.width() == 96);
  for (std::size_t i = 0; i < 96; ++i)
    assert(got96.getBit(i) == u96.getBit(i));
  auto got128 = s.u128();
  assert(got128.width() == 128);
  for (std::size_t i = 0; i < 128; ++i)
    assert(got128.getBit(i) == u128.getBit(i));

  // Wire layout: ui128 at byte 0..15 (LSB), then ui96 at byte 16..27.
  std::array<uint8_t, 28> expected{
      // u128 first (LSB):
      0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, //
      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, //
      // u96 next:
      0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, //
      0xBE, 0xBA, 0xFE, 0xCA,                         //
  };
  expectBytes(s, expected, "WideU");

  // Constructor takes the same value-class params, in manifest order.
  WideU cons(u96, u128);
  assert(wireBytes(cons) == wireBytes(s));
}

void testWideSigned() {
  // si128 max = all ones except the sign bit = 0x7FFF...FF.
  // si128 -1  = all ones.
  WideS s;
  auto s96_zero = bvFromBytes({}, 96);
  auto s128_neg_one = bvFromBytes(
      {
          0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
          0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
      },
      128);
  s.s96(s96_zero).s128(s128_neg_one);

  auto got = s.s128();
  for (std::size_t i = 0; i < 128; ++i)
    assert(got.getBit(i)); // -1 has every bit set
  auto got96 = s.s96();
  for (std::size_t i = 0; i < 96; ++i)
    assert(!got96.getBit(i));

  // Wire bytes: low 16 bytes all-ones (s128), then 12 zero bytes (s96).
  std::array<uint8_t, 28> expected{};
  for (std::size_t i = 0; i < 16; ++i)
    expected[i] = 0xFF;
  expectBytes(s, expected, "WideS s128=-1, s96=0");
}

void testBitsField() {
  // BitsType > 64 routes through Path D (non-owning `esi::BitVector`
  // view); narrower Bits stay on the native int paths and are covered
  // by the other tests above.
  BitsField f;
  auto wide = bvFromBytes(
      {
          0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, //
          0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, //
      },
      128);
  f.wide(wide);

  auto got_wide = f.wide();
  assert(got_wide.width() == 128);
  for (std::size_t i = 0; i < 128; ++i)
    assert(got_wide.getBit(i) == wide.getBit(i));
}

void testWideMisaligned() {
  // WideMisaligned: tag (ui3) at bits 0..2, payload (ui128) at bits
  // 3..130 — i.e. a wide value at a non-byte-aligned offset. Hits the
  // `copyBitsIn` / `copyBitsOut` arm of Path D.
  WideMisaligned m;
  auto payload = bvFromBytes(
      {
          0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, //
          0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, //
      },
      128);
  m.tag(0b101).payload(payload);
  assert(m.tag() == 0b101);

  auto got = m.payload();
  assert(got.width() == 128);
  for (std::size_t i = 0; i < 128; ++i)
    assert(got.getBit(i) == payload.getBit(i));

  // Re-setting `tag` must not disturb `payload`, and vice versa.
  m.tag(0b010);
  auto got2 = m.payload();
  for (std::size_t i = 0; i < 128; ++i)
    assert(got2.getBit(i) == payload.getBit(i));
  assert(m.tag() == 0b010);

  // A narrower input value must be zero-extended; a wider input is
  // truncated. Verify the former here (the latter is exercised
  // implicitly because the Path D setter clamps to bit_width).
  auto narrow = bvFromBytes({0xAA}, 8); // only 8 bits of input
  m.payload(narrow);
  auto got3 = m.payload();
  // Low 8 bits == 0xAA, upper 120 bits == 0.
  for (std::size_t i = 0; i < 8; ++i)
    assert(got3.getBit(i) == ((0xAA >> i) & 1));
  for (std::size_t i = 8; i < 128; ++i)
    assert(!got3.getBit(i));
  assert(m.tag() == 0b010); // still preserved
}

// ---------------------------------------------------------------------------
// Array of view-class elements: indexed + lazy whole-array accessors
// ---------------------------------------------------------------------------

// Round-trip a 3-element array of `ui128` through both the per-element
// indexed accessors (`items(i)` / `items(i, v)`) and the lazy
// whole-array accessor (`items()` returning a
// `std::views::iota | std::views::transform` range of per-element
// views, plus the `std::array<view, N>` whole-array setter and matching
// ctor). The byte-copy whole-array path used for native-int arrays
// doesn't apply here because the view's storage layout differs from
// the wire layout.
void testArrayOfViewsAligned() {
  ArrViews a;
  std::array<esi::MutableBitVector, 3> source;
  for (std::size_t i = 0; i < 3; ++i) {
    source[i] = bvFromBytes(
        {
            static_cast<uint8_t>(0x10 + i),
            static_cast<uint8_t>(0x20 + i),
            static_cast<uint8_t>(0x30 + i),
            static_cast<uint8_t>(0x40 + i),
            static_cast<uint8_t>(0x50 + i),
            static_cast<uint8_t>(0x60 + i),
            static_cast<uint8_t>(0x70 + i),
            static_cast<uint8_t>(0x80 + i),
            static_cast<uint8_t>(0x91 + i),
            static_cast<uint8_t>(0xA2 + i),
            static_cast<uint8_t>(0xB3 + i),
            static_cast<uint8_t>(0xC4 + i),
            static_cast<uint8_t>(0xD5 + i),
            static_cast<uint8_t>(0xE6 + i),
            static_cast<uint8_t>(0xF7 + i),
            static_cast<uint8_t>(0x08 + i),
        },
        128);
    a.items(i, source[i]);
  }
  for (std::size_t i = 0; i < 3; ++i) {
    auto got = a.items(i);
    assert(got.width() == 128);
    for (std::size_t b = 0; b < 128; ++b)
      assert(got.getBit(b) == source[i].getBit(b));
  }

  // Independence: writing element 1 must not disturb elements 0 or 2.
  auto fresh = bvFromBytes({0xAA}, 128);
  a.items(1, fresh);
  for (std::size_t b = 0; b < 128; ++b) {
    assert(a.items(0).getBit(b) == source[0].getBit(b));
    assert(a.items(2).getBit(b) == source[2].getBit(b));
  }

  // Lazy whole-array accessor (std::views::iota | std::views::transform):
  // yields per-element views on demand. Random-access, so size() and
  // r[i] both work, and we can range-for it.
  auto range = a.items();
  assert(std::ranges::size(range) == 3);
  std::size_t i = 0;
  for (auto view : range) {
    assert(view.width() == 128);
    for (std::size_t b = 0; b < 128; ++b)
      assert(view.getBit(b) == a.items(i).getBit(b));
    ++i;
  }
  assert(i == 3);

  // Whole-array setter taking `std::array<view, N>` and the matching
  // ctor: build a fresh struct from the array argument and confirm the
  // round-trip matches.
  std::array<esi::UIntView, 3> bulk = {
      esi::UIntView(source[0]),
      esi::UIntView(source[1]),
      esi::UIntView(source[2]),
  };
  ArrViews ctor_built(bulk);
  for (std::size_t k = 0; k < 3; ++k) {
    auto got = ctor_built.items(k);
    for (std::size_t b = 0; b < 128; ++b)
      assert(got.getBit(b) == source[k].getBit(b));
  }
  // Whole-array setter on an existing instance.
  ArrViews bulk_set;
  bulk_set.items(bulk);
  for (std::size_t k = 0; k < 3; ++k) {
    auto got = bulk_set.items(k);
    for (std::size_t b = 0; b < 128; ++b)
      assert(got.getBit(b) == source[k].getBit(b));
  }
}

void testArrayOfViewsMisaligned() {
  // `items` is the FIRST manifest field, so on the wire it lands at the
  // top of the buffer with `tag` (ui3) at bits 0..2 and the 3 ui128
  // elements at bits 3..130, 131..258, 259..386. Every element is
  // bit-misaligned -- exercises the runtime per-bit setter loop and the
  // sub-byte BitVector view constructor.
  ArrViewsMis m;
  m.tag(0b101);
  std::array<esi::MutableBitVector, 3> source;
  for (std::size_t i = 0; i < 3; ++i) {
    source[i] = bvFromBytes(
        {
            static_cast<uint8_t>(0xC0 + i),
            static_cast<uint8_t>(0xC1 + i),
            static_cast<uint8_t>(0xC2 + i),
            static_cast<uint8_t>(0xC3 + i),
            static_cast<uint8_t>(0xC4 + i),
            static_cast<uint8_t>(0xC5 + i),
            static_cast<uint8_t>(0xC6 + i),
            static_cast<uint8_t>(0xC7 + i),
            static_cast<uint8_t>(0xC8 + i),
            static_cast<uint8_t>(0xC9 + i),
            static_cast<uint8_t>(0xCA + i),
            static_cast<uint8_t>(0xCB + i),
            static_cast<uint8_t>(0xCC + i),
            static_cast<uint8_t>(0xCD + i),
            static_cast<uint8_t>(0xCE + i),
            static_cast<uint8_t>(0xCF + i),
        },
        128);
    m.items(i, source[i]);
  }
  assert(m.tag() == 0b101);
  for (std::size_t i = 0; i < 3; ++i) {
    auto got = m.items(i);
    assert(got.width() == 128);
    for (std::size_t b = 0; b < 128; ++b)
      assert(got.getBit(b) == source[i].getBit(b));
  }
}

} // namespace

int main() {
  testStandardWidthUnsigned();
  testStandardWidthSigned();
  testByteAlignedOddWidthUnsigned();
  testByteAlignedOddWidthSigned();
  testSubByteUnsigned();
  testSubByteSigned();
  testBoolField();
  testNestedStruct();
  testNestedStructMisaligned();
  testArrayOfIntegers();
  testUnion();
  testWindowList();
  testWideUnsigned();
  testWideSigned();
  testBitsField();
  testWideMisaligned();
  testArrayOfViewsAligned();
  testArrayOfViewsMisaligned();
  std::printf("OK\n");
  return 0;
}
