//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lib/Dialect/LLHD/Transforms/DeseqDNF.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace llhd;

namespace circt {
namespace llhd {

/// Print a given DNF with a custom value printing function that interprets the
/// raw pointer behind `Value` as a character.
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DNF &dnf) {
  dnf.print(os, [&](Value value) {
    size_t raw = (size_t)value.getAsOpaquePointer();
    assert((raw & 0b1111) == 0);
    raw >>= 4;
    assert(raw > 0);
    --raw;
    assert(raw < 26);
    os << (char)(raw + 'a');
  });
  return os;
}

/// Convert a DNF into a string version of `<<`.
static std::string str(const DNF &dnf) {
  std::string buffer;
  llvm::raw_string_ostream(buffer) << dnf;
  return buffer;
}

/// Compare a DNF against a string.
static bool operator==(const DNF &a, StringRef b) { return str(a) == b; }

} // namespace llhd
} // namespace circt

namespace {

// A few values we can pass into DNFs, that print back to a letter.
Value val(size_t idx) {
  assert(idx >= 'a' && idx <= 'z');
  return Value::getFromOpaquePointer((void *)((idx - 'a' + 1) << 4));
}
auto a = val('a');
auto b = val('b');
auto c = val('c');
auto d = val('d');

TEST(DNFTest, Atoms) {
  ASSERT_EQ(DNF(), "false");
  ASSERT_EQ(DNF(true), "true");
  ASSERT_EQ(DNF(false), "false");
  ASSERT_EQ(DNF(a), "a");
  ASSERT_EQ(DNF(AndTerm::id(a)), "a");
  ASSERT_EQ(DNF(AndTerm::past(a)), "@a");
  ASSERT_EQ(DNF(AndTerm::posEdge(a)), "/a");
  ASSERT_EQ(DNF(AndTerm::negEdge(a)), "\\a");
}

TEST(DNFTest, Negation) {
  ASSERT_EQ(~DNF(a), "!a");
  ASSERT_EQ(~(DNF(a) & DNF(b) | DNF(c) & DNF(d)),
            "!a&!c | !a&!d | !b&!c | !b&!d");
}

TEST(DNFTest, Or) {
  ASSERT_EQ(DNF(a) | DNF(b), "a | b");
  ASSERT_EQ(DNF(b) | DNF(a), "a | b");
}

TEST(DNFTest, And) {
  ASSERT_EQ(DNF(a) & DNF(b), "a&b");
  ASSERT_EQ(DNF(b) & DNF(a), "a&b");
}

TEST(DNFTest, Xor) {
  ASSERT_EQ(DNF(a) ^ DNF(b), "a&!b | !a&b");
  ASSERT_EQ(DNF(b) ^ DNF(a), "a&!b | !a&b");
}

TEST(DNFTest, Distributivity) {
  ASSERT_EQ(DNF(c) & (DNF(a) | DNF(b)), "a&c | b&c");
  ASSERT_EQ((DNF(a) | DNF(b)) & DNF(c), "a&c | b&c");
}

TEST(DNFTest, UniqueTerms) {
  auto abc = DNF(a) & DNF(b) & DNF(c);
  ASSERT_EQ(DNF(a) | DNF(a), DNF(a));
  ASSERT_EQ(DNF(a) & DNF(a), DNF(a));
  ASSERT_EQ(abc | abc, abc);
  ASSERT_EQ(abc & abc, abc);
}

TEST(DNFTest, OrSubset) {
  auto abc = DNF(a) & DNF(b) & DNF(c);
  auto abcd = abc & DNF(d);
  ASSERT_EQ(abc | abcd, abc);
  ASSERT_EQ(abcd | abc, abc);

  abcd = DNF(d) & abc;
  ASSERT_EQ(abc | abcd, abc);
  ASSERT_EQ(abcd | abc, abc);

  // /a & a = /a
  // \a & ~a = \a
  ASSERT_EQ(DNF(AndTerm::posEdge(a)) & DNF(a), DNF(AndTerm::posEdge(a)));
  ASSERT_EQ(DNF(AndTerm::negEdge(a)) & ~DNF(a), DNF(AndTerm::negEdge(a)));

  // /a&/b | /a&b = /a&b
  auto papb = DNF(AndTerm::posEdge(a)) & DNF(AndTerm::posEdge(b));
  auto pab = DNF(AndTerm::posEdge(a)) & DNF(b);
  ASSERT_EQ(papb | pab, pab);
}

TEST(DNFTest, OrComplement) {
  // a | !a = true
  ASSERT_EQ(DNF(a) | ~DNF(a), DNF(true));
  // a&b&c | a&!b&c = a&c
  ASSERT_EQ((DNF(a) & DNF(b) & DNF(c)) | (DNF(a) & ~DNF(b) & DNF(c)),
            DNF(a) & DNF(c));
}

TEST(DNFTest, OrComplementAndSubset) {
  auto abc = DNF(a) & DNF(b) & DNF(c);
  auto acd = DNF(a) & DNF(c) & DNF(d);
  auto anbc = DNF(a) & ~DNF(b) & DNF(c);
  // a&b&c | a&c&d | a&!b&c = a&c | a&c&d = a&c
  ASSERT_EQ((abc | acd) | anbc, DNF(a) & DNF(c));
}

TEST(DNFTest, AndComplement) {
  ASSERT_EQ(DNF(a) & ~DNF(a), DNF(false));
  ASSERT_EQ(DNF(AndTerm::past(a)) & ~DNF(AndTerm::past(a)), DNF(false));
  ASSERT_EQ(DNF(AndTerm::posEdge(a)) & ~DNF(AndTerm::posEdge(a)), DNF(false));
  ASSERT_EQ(DNF(AndTerm::negEdge(a)) & ~DNF(AndTerm::negEdge(a)), DNF(false));

  // /a & ~a = false
  // \a & a = false
  ASSERT_EQ(DNF(AndTerm::posEdge(a)) & ~DNF(a), DNF(false));
  ASSERT_EQ(DNF(AndTerm::negEdge(a)) & DNF(a), DNF(false));
}

TEST(DNFTest, DetectEdges) {
  auto pastA = DNF(AndTerm::past(a));
  ASSERT_EQ(~pastA & DNF(a) & DNF(b), DNF(AndTerm::posEdge(a)) & DNF(b));
  ASSERT_EQ(pastA & ~DNF(a) & DNF(b), DNF(AndTerm::negEdge(a)) & DNF(b));
}

} // namespace
