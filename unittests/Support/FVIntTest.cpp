//===- FVIntTest.cpp - Four-valued integer unit tests ===------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the `FVInt` class.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/FVInt.h"
#include "gtest/gtest.h"

using namespace circt;

namespace {

TEST(FVIntTest, Resizing) {
  ASSERT_EQ(FVInt::fromString("01", 2).zext(5), FVInt::fromString("00001", 2));
  ASSERT_EQ(FVInt::fromString("01", 2).sext(5), FVInt::fromString("00001", 2));
  ASSERT_EQ(FVInt::fromString("10", 2).zext(5), FVInt::fromString("00010", 2));
  ASSERT_EQ(FVInt::fromString("10", 2).sext(5), FVInt::fromString("11110", 2));
  ASSERT_EQ(FVInt::fromString("X1", 2).zext(5), FVInt::fromString("000X1", 2));
  ASSERT_EQ(FVInt::fromString("X1", 2).sext(5), FVInt::fromString("XXXX1", 2));
  ASSERT_EQ(FVInt::fromString("Z1", 2).zext(5), FVInt::fromString("000Z1", 2));
  ASSERT_EQ(FVInt::fromString("Z1", 2).sext(5), FVInt::fromString("ZZZZ1", 2));
}

TEST(FVIntTest, Basics) {
  ASSERT_TRUE(FVInt::getZero(42).isZero());
  ASSERT_TRUE(FVInt::getAllOnes(42).isAllOnes());
  ASSERT_TRUE(FVInt::getAllX(42).isAllX());
  ASSERT_TRUE(FVInt::getAllZ(42).isAllZ());

  ASSERT_FALSE(FVInt::getZero(42).hasUnknown());
  ASSERT_FALSE(FVInt::getAllOnes(42).hasUnknown());
  ASSERT_TRUE(FVInt::getAllX(42).hasUnknown());
  ASSERT_TRUE(FVInt::getAllZ(42).hasUnknown());

  auto x = FVInt::fromString("01XZ", 2);
  ASSERT_EQ(x.toAPInt(false), 0b0100);
  ASSERT_EQ(x.toAPInt(true), 0b0111);
  ASSERT_EQ(x.getBit(0), FVInt::Z);
  ASSERT_EQ(x.getBit(1), FVInt::X);
  ASSERT_EQ(x.getBit(2), FVInt::V1);
  ASSERT_EQ(x.getBit(3), FVInt::V0);
  ASSERT_EQ(FVInt::V1, 1);
  ASSERT_EQ(FVInt::V0, 0);

  ASSERT_EQ(FVInt(32, 9001), FVInt(32, 9001));
  ASSERT_EQ(FVInt(32, 9001), 9001);
  ASSERT_EQ(9001, FVInt(32, 9001));

  ASSERT_NE(FVInt(32, 9001), FVInt(32, 1337));
  ASSERT_NE(FVInt(32, 9001), 1337);
  ASSERT_NE(9001, FVInt(32, 1337));
}

TEST(FVIntTest, StringConversion) {
  auto v = FVInt::fromString("ZX1001XZ", 2);
  ASSERT_EQ(v.getZeroBits(), 0b00011000);
  ASSERT_EQ(v.getOneBits(), 0b00100100);
  ASSERT_EQ(v.getXBits(), 0b01000010);
  ASSERT_EQ(v.getZBits(), 0b10000001);

  ASSERT_EQ(FVInt::getZero(0).toString(2), StringRef("0"));
  ASSERT_EQ(FVInt::getZero(0).toString(8), StringRef("0"));
  ASSERT_EQ(FVInt::getZero(0).toString(10), StringRef("0"));
  ASSERT_EQ(FVInt::getZero(0).toString(16), StringRef("0"));

  // Parsing/printing without unknown values.
  ASSERT_EQ(FVInt::fromString("10101100", 2).toString(2),
            StringRef("10101100"));
  ASSERT_EQ(FVInt::fromString("1234567", 8).toString(8), StringRef("1234567"));
  ASSERT_EQ(FVInt::fromString("1234567890", 10).toString(10),
            StringRef("1234567890"));
  ASSERT_EQ(FVInt::fromString("1234567890ABCDEF", 16).toString(16),
            "1234567890ABCDEF");
  ASSERT_EQ(FVInt::fromString("1234567890abcdef", 16).toString(16, false),
            "1234567890abcdef");

  // Parsing/printing with unknown values.
  ASSERT_EQ(FVInt::fromString("10XZ1XZ0", 2).toString(2),
            StringRef("10XZ1XZ0"));
  ASSERT_EQ(FVInt::fromString("10xz1xz0", 2).toString(2, false),
            StringRef("10xz1xz0"));
  ASSERT_EQ(FVInt::fromString("1234XZ567", 8).toString(8),
            StringRef("1234XZ567"));
  ASSERT_EQ(FVInt::fromString("1234xz567", 8).toString(8, false),
            StringRef("1234xz567"));
  ASSERT_EQ(FVInt::fromString("12345XZ67890ABCDEF", 16).toString(16),
            StringRef("12345XZ67890ABCDEF"));
  ASSERT_EQ(FVInt::fromString("12345xz67890abcdef", 16).toString(16, false),
            StringRef("12345xz67890abcdef"));

  // Narrow <4 bit integers printed as hex.
  ASSERT_EQ(FVInt::fromString("10", 2).toString(16), StringRef("2"));
}

TEST(FVIntTest, LogicOps) {
  auto a = FVInt::fromString("01XZ01XZ01XZ01XZ", 2);
  auto b = FVInt::fromString("00001111XXXXZZZZ", 2);
  auto c = FVInt::fromString("01XZ", 2);

  ASSERT_EQ(~c, FVInt::fromString("10XX", 2));
  ASSERT_EQ(a & b, FVInt::fromString("000001XX0XXX0XXX", 2));
  ASSERT_EQ(a | b, FVInt::fromString("01XX1111X1XXX1XX", 2));
  ASSERT_EQ(a ^ b, FVInt::fromString("01XX10XXXXXXXXXX", 2));
}

TEST(FVIntTest, ArithmeticOps) {
  auto a = FVInt::fromString("123").zext(32);
  auto b = FVInt::fromString("234").zext(32);
  auto c = FVInt::fromString("1XZ", 16).zext(32);

  ASSERT_EQ(-a, uint32_t(-123));
  ASSERT_TRUE((-c).isAllX());

  ASSERT_EQ(a + 1, FVInt::fromString("124").zext(32));
  ASSERT_EQ(1 + b, FVInt::fromString("235").zext(32));
  ASSERT_EQ(a + b, FVInt::fromString("357").zext(32));
  ASSERT_TRUE((a + c).isAllX());

  ASSERT_EQ(a - 1, FVInt::fromString("122").zext(32));
  ASSERT_EQ(234 - a, FVInt::fromString("111").zext(32));
  ASSERT_EQ(b - a, FVInt::fromString("111").zext(32));
  ASSERT_TRUE((a - c).isAllX());

  ASSERT_EQ(a * 2, FVInt::fromString("246").zext(32));
  ASSERT_EQ(2 * b, FVInt::fromString("468").zext(32));
  ASSERT_EQ(a * b, FVInt::fromString("28782").zext(32));
  ASSERT_TRUE((a * c).isAllX());
}

} // namespace
