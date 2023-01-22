//===- FFIContext.cpp - FIRRTL FFI context unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLFFIContext.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace chirrtl;

namespace {

TEST(ParseUntilUnbalanced, FFIContext) {
  using details::parseUntilUnbalanced;

#define MK_PAIR(a, b) (std::make_pair(StringRef{(a)}, StringRef{(b)}))

  EXPECT_EQ(parseUntilUnbalanced("<abcdefg>", '<', '>'),
            MK_PAIR("abcdefg", ""));
  EXPECT_EQ(parseUntilUnbalanced("<abcd<efg>>", '<', '>'),
            MK_PAIR("abcd<efg>", ""));
  EXPECT_EQ(parseUntilUnbalanced("<ab<cde>fg>", '<', '>'),
            MK_PAIR("ab<cde>fg", ""));
  EXPECT_EQ(parseUntilUnbalanced("<<abc>defg>", '<', '>'),
            MK_PAIR("<abc>defg", ""));
  EXPECT_EQ(parseUntilUnbalanced("<abc<defg<<hijklmn>opq>>uvw>xyz", '<', '>'),
            MK_PAIR("abc<defg<<hijklmn>opq>>uvw", "xyz"));
  EXPECT_EQ(parseUntilUnbalanced("<abc<defg<<hijklmn>opq>uvw>xyz", '<', '>'),
            std::nullopt);
  EXPECT_EQ(parseUntilUnbalanced("<abc<defg>", '<', '>'), std::nullopt);
  EXPECT_EQ(parseUntilUnbalanced("<abc>defg>", '<', '>'),
            MK_PAIR("abc", "defg>"));
  EXPECT_EQ(parseUntilUnbalanced("<>", '<', '>'), MK_PAIR("", ""));
}

} // namespace
