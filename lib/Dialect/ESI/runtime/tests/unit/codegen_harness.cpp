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
#include <type_traits>
#include <vector>

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
  std::printf("OK\n");
  return 0;
}
