//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lib/Dialect/LLHD/Transforms/DeseqUtils.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace llhd;
using namespace deseq;

namespace circt {
namespace llhd {
namespace deseq {
static bool operator==(const TruthTable &a, StringRef b) {
  std::string buffer;
  llvm::raw_string_ostream(buffer) << a;
  return buffer == b;
}
} // namespace deseq
} // namespace llhd
} // namespace circt

namespace {

const auto F = TruthTable::getConst(3, false);
const auto T = TruthTable::getConst(3, true);
const auto p = TruthTable::getPoison();
const auto x = TruthTable::getTerm(3, 0);
const auto a1 = TruthTable::getTerm(3, 1);
const auto a2 = TruthTable::getTerm(3, 2);

// Basic operations.
TEST(TruthTable, Basics) {
  ASSERT_EQ(a1, "a1");
  ASSERT_EQ(~a1, "!a1");
  ASSERT_EQ(a1 & a2, "a1&a2");
  ASSERT_EQ(a1 | a2, "a1 | a2");
  ASSERT_EQ(a1 ^ a2, "a1&!a2 | !a1&a2");
}

// Unknown marker semantics.
TEST(TruthTable, Unknown) {
  ASSERT_EQ(x, "x");
  ASSERT_EQ(~x, "x");
  ASSERT_EQ(x & x, "x");
  ASSERT_EQ(x | x, "x");
  ASSERT_EQ(x ^ x, "x");
}

// Identities.
TEST(TruthTable, Identities) {
  ASSERT_EQ(a1 & F, "false");
  ASSERT_EQ(a1 & T, "a1");
  ASSERT_EQ(a1 | F, "a1");
  ASSERT_EQ(a1 | T, "true");
  ASSERT_EQ(a1 & ~a1, "false");
  ASSERT_EQ(a1 | ~a1, "true");
  ASSERT_EQ(a1 | a1, "a1");
  ASSERT_EQ(a1 & a1, "a1");
  ASSERT_EQ(a1 ^ a1, "false");
  ASSERT_EQ(a1 ^ ~a1, "true");
}

// Basic operations involving unknown markers.
TEST(TruthTable, BasicsWithUnknown) {
  ASSERT_EQ(x & a1, "x&a1");
  ASSERT_EQ((x & a1) & a2, "x&a1&a2");
  ASSERT_EQ((x & a1) & (x & a2), "x&a1&a2");
  ASSERT_EQ(~(x & a1), "!a1 | x");
  ASSERT_EQ(x ^ a1, "x");
  ASSERT_EQ((x & a1) ^ a1, "x&a1");
  ASSERT_EQ((x & a1) ^ (x & a1), "x&a1");
}

// Identities involving unknown markers.
TEST(TruthTable, IdentitiesWithUnknown) {
  ASSERT_EQ((x & a1) & ~a1, "false");
  ASSERT_EQ((x & ~a1) & a1, "false");
  ASSERT_EQ((x & ~a1) & (x & a1), "false");
  ASSERT_EQ((x & a1) | a1, "a1");
}

// Poison.
TEST(TruthTable, Poison) {
  ASSERT_EQ(p, "poison");
  ASSERT_EQ(~p, "poison");
  ASSERT_EQ(p & x, "poison");
  ASSERT_EQ(x & p, "poison");
  ASSERT_EQ(p | x, "poison");
  ASSERT_EQ(x | p, "poison");
  ASSERT_EQ(p ^ x, "poison");
  ASSERT_EQ(x ^ p, "poison");
  ASSERT_EQ(p & a1, "poison");
  ASSERT_EQ(p | a1, "poison");
  ASSERT_EQ(p ^ a1, "poison");
  ASSERT_EQ(a1 & p, "poison");
  ASSERT_EQ(a1 | p, "poison");
  ASSERT_EQ(a1 ^ p, "poison");
}

} // namespace
