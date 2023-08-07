//===- RLELogicTests.cpp - Run-length encoded logic tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWLogicDigits.h"
#include "circt/Dialect/HW/HWRLELogic.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace hw;
using namespace logicdigits;

using namespace llvm;

static uint8_t getRunLength(RLELogic::RunLengthCode code) {
  return (code >> 4) + 1;
}
static uint8_t getRawDigit(RLELogic::RunLengthCode code) { return code & 0x0F; }

static constexpr char l9Values[10] = "UX01ZWLH-";

static const char randomL9[401] =
    "1-XU1WZZWULWHU0Z00WLZXU0-ZLH1LH--1-UUZ01WX1WWLZWZW0LH-X1U111-WULZ-"
    "WZWHH1111H1U1L-1W-ZUHL0XZL-0W1X0UZXLL-00ZUUXLZ10L0L0ZUH1HXXZLLWL0-010-"
    "X0ZWU0X1ZX-XWL1XLH--0WL001ZLLLUXUZ-XZ-ZL-UZW1HW-U-U-W-"
    "HUUZWZ1U10HXZ0ZH1HLUH-1ZLLULU0HWUUHHHH0HZLU1L0110-UWZ1L01XHXHH---X0HLXZ-"
    "LHHLL-WLU-W-H-WZUXHZX-0XWLU1UU0UX1H-XW1XH-1WHLZXZWH1-ULXWZ1ZUL01WX0W1L-"
    "XHHZ10W-1HWHLXLX0HZWWULWWZZ-00-LXUUW0XX1-Z1-UW-ZLZZLLU1UULLWLWX0H1U";
static const char randomL4[401] =
    "00XZ11Z1X010ZZ10XZX0X11Z0110X0Z1Z1ZZ010ZXX1XXZ001ZXX10ZXZ00X1XXXZXZXZX11Z1"
    "1Z110XZ0ZX00XZZ0101XXZ01XZ110ZZ11ZX1XZZZXXZXZ1XX110XX0XZZX0011ZZZ1101X1ZZX"
    "Z0Z1Z0X11101Z1XXZ11ZXX0ZZ0Z1XXZ1X001ZZ0Z1ZX1XXZZ1011XZZXZZXX1010XX0ZX00ZZ0"
    "00ZXX00Z0XZZZX0ZXXZ1ZXZ1X0XZXX01X11XX01ZZZXZZ01ZZ000XX1X00X0Z11ZXXXXXX10X0"
    "0100ZXZZXX11X1Z0ZX01011XXX11ZZZXX11X1X10XZZZ100X0Z1XX0ZZZZXXZ0XZZXZ1XXZ01Z"
    "100Z001X0X1ZZ00X10Z01ZZZ1ZXX1X";
static const char randomL2[401] =
    "00000101011001101000111111111000110001110011001111010111010000011011111000"
    "10111011010100000101000111101110001101110011110001010111101000110000010010"
    "11100100011100001000010000110001001000011010100011011011110011100111000011"
    "01100011110110010011100111011011101100111111011111000100000010111010011000"
    "01100110100110101100000100100110001000001101010001001000111111100000110001"
    "011111010100111011011001011001";

static std::string strRepeat(size_t n, const std::string &pat) {
  std::string str;
  str.reserve(pat.size() * n);
  for (size_t i = 0; i < n; i++) {
    str.append(pat);
  }
  return str;
}

// Verify encoding invariants
static const RLELogic &verif(const RLELogic &rlelog) {
  if (!rlelog.isValid()) {
    EXPECT_EQ(rlelog.getByteCount(), 0U);
    EXPECT_EQ(rlelog.getDigitMask(), 0U);
    return rlelog;
  }

  auto bytes = rlelog.getByteCount();
  EXPECT_NE(bytes, 0U);
  EXPECT_NE(rlelog.getDigitMask(), 0U);
  const auto *ptr = rlelog.getCodePointer();

  uint16_t verifMask = 0;
  for (size_t i = 0; i < bytes; i++) {
    auto rawDigit = getRawDigit(ptr[i]);
    EXPECT_TRUE(isValidLogicDigit((LogicDigit)rawDigit));
    verifMask |= 1 << (rawDigit - 1);
    // If two subsequent digits are equal, the former must have maximum run
    // length
    if (i != 0) {
      auto prevDigit = getRawDigit(ptr[i - 1]);
      if (prevDigit == rawDigit) {
        EXPECT_EQ(getRunLength(ptr[i - 1]), 16U);
      }
    }
  }
  EXPECT_EQ(rlelog.getDigitMask(), verifMask);

  // Last digit must have encoded run length = 1
  EXPECT_EQ(getRunLength(ptr[bytes - 1]), 1U);
  // Last digit must not be repeated
  if (bytes > 1) {
    EXPECT_NE(getRawDigit(ptr[bytes - 1]), getRawDigit(ptr[bytes - 2]));
  }

  return rlelog;
}

static void parsePrintTest(const std::string &str) {
  auto rlelog = verif(RLELogic::encode(StringRef(str)));
  EXPECT_STREQ(rlelog.toString(str.size()).c_str(), str.c_str());
}

TEST(RLELogicTest, ParsePrintTest) {
  parsePrintTest("X");
  parsePrintTest("ZX");
  parsePrintTest("ZXX");
  parsePrintTest("WWX");
  parsePrintTest("WWXX");
  parsePrintTest("-UXZWHL10");
  parsePrintTest("01LHWZXU-");

  for (size_t n = 1; n < 150; n++) {
    parsePrintTest(std::string(n, '-'));
    parsePrintTest(strRepeat(n, "10"));
    parsePrintTest(strRepeat(n, "UWU"));
    parsePrintTest("Z" + std::string(n, '0'));
    parsePrintTest(std::string(n, 'L') + std::string(n, 'H') +
                   std::string(n, 'X') + std::string(n, 'L'));

    std::string randl9(randomL9 + n, n);
    std::string randl4(randomL4 + n, n);
    std::string randl2(randomL2 + n, n);
    parsePrintTest(randl9);
    parsePrintTest(randl4);
    parsePrintTest(randl2);
  }
}

TEST(RLELogicTest, ContainsTest) {
  RLELogic rlelog = verif(RLELogic::encode("Z0XHZZZZZ000-ZXXXXXXXZZZHHHHH"));

  EXPECT_FALSE((rlelog.containsOnly<'U'>()));
  EXPECT_FALSE((rlelog.containsOnly<'0'>()));
  EXPECT_TRUE((rlelog.containsOnly<'0', 'Z', 'X', 'H', '-'>()));
  EXPECT_TRUE((rlelog.containsOnly<'0', 'Z', 'X', 'U', 'H', '-'>()));
  EXPECT_FALSE((rlelog.containsOnly<'0', 'Z', 'X', 'U', 'H'>()));
  EXPECT_FALSE((rlelog.containsOnly<'Z', 'X', 'U', 'H', '-'>()));

  EXPECT_TRUE((rlelog.containsAny<'-'>()));
  EXPECT_FALSE((rlelog.containsAny<'U'>()));
  EXPECT_FALSE((rlelog.containsAny<'U', 'W'>()));
  EXPECT_TRUE((rlelog.containsAny<'U', 'W', '-'>()));
  EXPECT_TRUE((rlelog.containsAny<'0', 'Z', 'X', 'H', '-'>()));
  EXPECT_TRUE((rlelog.containsAny<'0', 'Z', 'U'>()));

  EXPECT_FALSE(rlelog.containsOnly(rlelog.getDigitMask(), LogicDigit::LD_U));
  EXPECT_FALSE(rlelog.containsOnly(rlelog.getDigitMask(), LogicDigit::LD_0));
  EXPECT_TRUE(rlelog.containsOnly(rlelog.getDigitMask(), LogicDigit::LD_0,
                                  LogicDigit::LD_Z, LogicDigit::LD_X,
                                  LogicDigit::LD_H, LogicDigit::LD_DC));
  EXPECT_TRUE(rlelog.containsOnly(
      rlelog.getDigitMask(), LogicDigit::LD_0, LogicDigit::LD_Z,
      LogicDigit::LD_X, LogicDigit::LD_U, LogicDigit::LD_H, LogicDigit::LD_DC));
  EXPECT_FALSE(rlelog.containsOnly(rlelog.getDigitMask(), LogicDigit::LD_0,
                                   LogicDigit::LD_Z, LogicDigit::LD_X,
                                   LogicDigit::LD_U, LogicDigit::LD_H));
  EXPECT_FALSE(rlelog.containsOnly(rlelog.getDigitMask(), LogicDigit::LD_Z,
                                   LogicDigit::LD_X, LogicDigit::LD_U,
                                   LogicDigit::LD_H, LogicDigit::LD_DC));

  EXPECT_TRUE(rlelog.containsAny(rlelog.getDigitMask(), LogicDigit::LD_DC));
  EXPECT_FALSE(rlelog.containsAny(rlelog.getDigitMask(), LogicDigit::LD_U));
  EXPECT_FALSE(rlelog.containsAny(rlelog.getDigitMask(), LogicDigit::LD_U,
                                  LogicDigit::LD_W));
  EXPECT_TRUE(rlelog.containsAny(rlelog.getDigitMask(), LogicDigit::LD_U,
                                 LogicDigit::LD_W, LogicDigit::LD_DC));
  EXPECT_TRUE(rlelog.containsAny(rlelog.getDigitMask(), LogicDigit::LD_0,
                                 LogicDigit::LD_Z, LogicDigit::LD_X,
                                 LogicDigit::LD_H, LogicDigit::LD_DC));
  EXPECT_TRUE(rlelog.containsAny(rlelog.getDigitMask(), LogicDigit::LD_0,
                                 LogicDigit::LD_Z, LogicDigit::LD_U));
}

TEST(RLELogicTest, FilledTest) {
  auto rle0 = verif(RLELogic::filled<l9Values[2]>());
  auto rle1 = verif(RLELogic::filled<l9Values[3]>());
  auto rleX = verif(RLELogic::filled<l9Values[1]>());
  auto rleZ = verif(RLELogic::filled<l9Values[4]>());
  auto rleU = verif(RLELogic::filled<l9Values[0]>());
  auto rleW = verif(RLELogic::filled<l9Values[5]>());
  auto rleL = verif(RLELogic::filled<l9Values[6]>());
  auto rleH = verif(RLELogic::filled<l9Values[7]>());
  auto rleDC = verif(RLELogic::filled<l9Values[8]>());

  EXPECT_TRUE(rle0.isInteger());
  EXPECT_TRUE(rle0.isIntegerLike());
  EXPECT_FALSE(rle0.containsAnyUnknownDigits());
  EXPECT_FALSE(rle0.containsOnlyUnknownDigits());

  EXPECT_TRUE(rle1.isInteger());
  EXPECT_TRUE(rle1.isIntegerLike());
  EXPECT_FALSE(rle0.containsAnyUnknownDigits());
  EXPECT_FALSE(rle0.containsOnlyUnknownDigits());

  EXPECT_FALSE(rleX.isInteger());
  EXPECT_FALSE(rleX.isIntegerLike());
  EXPECT_TRUE(rleX.containsAnyUnknownDigits());
  EXPECT_TRUE(rleX.containsOnlyUnknownDigits());

  EXPECT_FALSE(rleZ.isInteger());
  EXPECT_FALSE(rleZ.isIntegerLike());
  EXPECT_TRUE(rleZ.containsAnyUnknownDigits());
  EXPECT_TRUE(rleZ.containsOnlyUnknownDigits());

  EXPECT_FALSE(rleU.isInteger());
  EXPECT_FALSE(rleU.isIntegerLike());
  EXPECT_TRUE(rleU.containsAnyUnknownDigits());
  EXPECT_TRUE(rleU.containsOnlyUnknownDigits());

  EXPECT_FALSE(rleW.isInteger());
  EXPECT_FALSE(rleW.isIntegerLike());
  EXPECT_TRUE(rleW.containsAnyUnknownDigits());
  EXPECT_TRUE(rleW.containsOnlyUnknownDigits());

  EXPECT_FALSE(rleDC.isInteger());
  EXPECT_FALSE(rleDC.isIntegerLike());
  EXPECT_TRUE(rleDC.containsAnyUnknownDigits());
  EXPECT_TRUE(rleDC.containsOnlyUnknownDigits());

  EXPECT_FALSE(rleL.isInteger());
  EXPECT_TRUE(rleL.isIntegerLike());
  EXPECT_FALSE(rleL.containsAnyUnknownDigits());
  EXPECT_FALSE(rleL.containsOnlyUnknownDigits());

  EXPECT_FALSE(rleH.isInteger());
  EXPECT_TRUE(rleH.isIntegerLike());
  EXPECT_FALSE(rleH.containsAnyUnknownDigits());
  EXPECT_FALSE(rleH.containsOnlyUnknownDigits());

  for (auto c : std::string(l9Values)) {
    auto filled = verif(RLELogic::filled(charToLogicDigit(c)));
    auto rep = verif(RLELogic::encode(std::string(1022, c)));
    EXPECT_EQ(rep.getByteCount(), 1U);
    EXPECT_TRUE(filled == rep);
    EXPECT_TRUE(rep == filled);
    EXPECT_FALSE(filled != rep);
    EXPECT_FALSE(rep != filled);

    EXPECT_EQ(rep == rle0, c == '0');
    EXPECT_EQ(rep == rle1, c == '1');
    EXPECT_EQ(rep == rleX, c == 'X');
    EXPECT_EQ(rep == rleW, c == 'W');
    EXPECT_EQ(rep == rleU, c == 'U');
    EXPECT_EQ(rep == rleL, c == 'L');
    EXPECT_EQ(rep == rleH, c == 'H');
    EXPECT_EQ(rep == rleZ, c == 'Z');
    EXPECT_EQ(rep == rleDC, c == '-');
  }

  auto rleInval = verif(RLELogic::filled(LogicDigit::Invalid));
  EXPECT_FALSE(rleInval.isValid());
}

TEST(RLELogicTest, CopyAndMoveTest) {
  auto smallRef = std::string(randomL2, 63);
  auto largeRef = std::string(randomL9 + 7, 300);
  RLELogic smallOrig = verif(RLELogic::encode(smallRef));
  RLELogic largeOrig = verif(RLELogic::encode(largeRef));
  RLELogic smallCopy(smallOrig);
  verif(smallCopy);
  RLELogic largeCopy(largeOrig);
  verif(largeCopy);
  EXPECT_STREQ(smallCopy.toString(63).c_str(), smallRef.c_str());
  EXPECT_STREQ(largeCopy.toString(300).c_str(), largeRef.c_str());

  RLELogic smallCopyAssigned = RLELogic::filled<'Z'>();
  verif(smallCopyAssigned);
  EXPECT_STREQ(smallCopyAssigned.toString(2).c_str(), "ZZ");
  smallCopyAssigned = smallOrig;
  verif(smallCopyAssigned);
  EXPECT_STREQ(smallCopyAssigned.toString(63).c_str(), smallRef.c_str());

  RLELogic verySmallCopy(RLELogic::filled<'1'>());
  verif(verySmallCopy);
  EXPECT_TRUE(verySmallCopy.isInteger());
  RLELogic largeAnother(largeOrig);

  RLELogic smallMoved(std::move(smallCopy));
  RLELogic largeMoved(std::move(largeCopy));
  EXPECT_FALSE(smallCopy.isValid());
  EXPECT_FALSE(largeCopy.isValid());
  EXPECT_TRUE(smallMoved.isValid());
  EXPECT_TRUE(largeMoved.isValid());
  verif(smallMoved);
  verif(largeMoved);
  verif(smallCopy);
  verif(largeCopy);

  EXPECT_STREQ(smallMoved.toString(63).c_str(), smallRef.c_str());
  EXPECT_STREQ(largeMoved.toString(300).c_str(), largeRef.c_str());

  RLELogic verySmallMoved(smallOrig);
  largeMoved = RLELogic::filled<'X'>();

  verySmallMoved = std::move(verySmallCopy);
  largeMoved = std::move(largeAnother);
  EXPECT_FALSE(verySmallCopy.isValid());
  EXPECT_FALSE(largeAnother.isValid());
  EXPECT_TRUE(verySmallMoved.isValid());
  EXPECT_TRUE(largeMoved.isValid());
  verif(verySmallMoved);
  verif(largeMoved);

  EXPECT_STREQ(smallMoved.toString(63).c_str(), smallRef.c_str());
  EXPECT_STREQ(largeMoved.toString(300).c_str(), largeRef.c_str());
}

TEST(RLELogicTest, InvalidTest) {
  auto rleZeroLength = verif(RLELogic::encode(""));
  EXPECT_FALSE(rleZeroLength.isValid());
  auto rleInvalDigit = verif(RLELogic::encode("000011111ZZZZZMZZXXXXZZZZXXX"));
  EXPECT_FALSE(rleInvalDigit.isValid());
}

static void expectEqual(const RLELogic &a, const RLELogic &b) {
  verif(a);
  verif(b);
  EXPECT_TRUE(a == b);
  EXPECT_TRUE(b == a);
  EXPECT_FALSE(a != b);
  EXPECT_FALSE(b != a);
  EXPECT_EQ(hashValue(a), hashValue(b));
}

static void expectNotEqual(const RLELogic &a, const RLELogic &b) {
  verif(a);
  verif(b);
  EXPECT_FALSE(a == b);
  EXPECT_FALSE(b == a);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(b != a);
  // Assume we don't hit a collision by accident
  EXPECT_NE(hashValue(a), hashValue(b));
}

TEST(RLELogicTest, EqualsTest) {
  expectNotEqual(RLELogic::encode(""), RLELogic::encode("0"));
  expectEqual(RLELogic::encode("Z"), RLELogic::encode("Z"));
  expectEqual(RLELogic::encode("WW"), RLELogic::encode("W"));
  expectNotEqual(RLELogic::encode("10"), RLELogic::encode("0"));
  expectEqual(RLELogic::encode("1010101"), RLELogic::encode("11010101"));
  expectEqual(RLELogic::encode("01010101"), RLELogic::encode("001010101"));
  expectNotEqual(RLELogic::encode("1010101"), RLELogic::encode("101-101"));
  expectNotEqual(RLELogic::encode("1010101"), RLELogic::encode("1011101"));
  expectNotEqual(RLELogic::encode("01010101"), RLELogic::encode("0101011101"));

  auto testStrA = strRepeat(10, "XHL") + std::string(256, '0');
  auto testStrB = std::string(50, 'X') + testStrA;
  auto testStrC = std::string(50, 'Z') + testStrA;
  std::string testStrD(testStrA);
  std::string testStrE(testStrA);
  testStrD[250] = '-';
  testStrE[250] = 'H';
  expectEqual(RLELogic::encode(testStrA), RLELogic::encode(testStrA));
  expectEqual(RLELogic::encode(testStrA), RLELogic::encode(testStrB));
  expectNotEqual(RLELogic::encode(testStrA), RLELogic::encode(testStrC));
  expectNotEqual(RLELogic::encode(testStrA), RLELogic::encode(testStrD));
  expectNotEqual(RLELogic::encode(testStrD), RLELogic::encode(testStrE));

  expectNotEqual(RLELogic::encode(""), RLELogic::encode("S"));
  expectEqual(RLELogic::encode("S"), RLELogic::encode("FooBar"));

  RLELogic emptyKey =
      llvm::DenseMapInfo<circt::hw::RLELogic, void>::getEmptyKey();
  RLELogic tombstoneKey =
      llvm::DenseMapInfo<circt::hw::RLELogic, void>::getTombstoneKey();
  verif(emptyKey);
  verif(tombstoneKey);
  expectEqual(emptyKey, emptyKey);
  expectEqual(tombstoneKey, tombstoneKey);
  expectNotEqual(emptyKey, tombstoneKey);
  expectNotEqual(emptyKey, RLELogic::encode(""));
  expectNotEqual(tombstoneKey, RLELogic::encode(""));
  expectNotEqual(emptyKey, RLELogic::encode("0"));
  expectNotEqual(tombstoneKey, RLELogic::encode("0"));
}

TEST(RLELogicTest, SizeTest) {
  auto rle = verif(RLELogic::encode(""));
  EXPECT_EQ(rle.getByteCount(), 0U);

  rle = verif(RLELogic::encode("WW"));
  EXPECT_EQ(rle.getByteCount(), 1U);

  rle = verif(RLELogic::encode(std::string(256, 'U')));
  EXPECT_EQ(rle.getByteCount(), 1U);

  rle = verif(RLELogic::encode("10"));
  EXPECT_EQ(rle.getByteCount(), 2U);

  rle = verif(RLELogic::encode("010"));
  EXPECT_EQ(rle.getByteCount(), 3U);

  rle = verif(RLELogic::encode("000011110000000000"));
  EXPECT_EQ(rle.getByteCount(), 3U);

  rle = verif(RLELogic::encode(std::string(17, 'Z') + std::string(17, 'U') +
                               std::string(16, 'Z') + std::string(16, '1')));
  EXPECT_EQ(rle.getByteCount(), 5U);

  rle = verif(RLELogic::encode("0" + std::string(256, 'U')));
  EXPECT_EQ(rle.getByteCount(), 17U);

  rle = verif(RLELogic::encode(strRepeat(100, "LH")));
  EXPECT_EQ(rle.getByteCount(), 200U);

  rle = verif(RLELogic::encode(strRepeat(100, "LLLLLLLLLHHHH")));
  EXPECT_EQ(rle.getByteCount(), 200U);
}

TEST(RLELogicTest, BoundedIteratorTest) {
  auto rlelogW = RLELogic::filled<'W'>();

  auto logWIter0 = rlelogW.boundedIterator(0);
  EXPECT_EQ(logWIter0.begin(), logWIter0.end());

  auto logWIter1 = rlelogW.boundedIterator(1);
  auto logWIter1Cont = logWIter1.begin();
  EXPECT_NE(logWIter1Cont, logWIter1.end());
  EXPECT_EQ(*logWIter1Cont, LogicDigit::LD_W);
  logWIter1Cont++;
  EXPECT_EQ(logWIter1Cont, logWIter1.end());

  auto logWIter25 = rlelogW.boundedIterator(25);
  auto logWIter25Cont = logWIter25.begin();
  for (unsigned i = 0; i < 25; i++) {
    EXPECT_NE(logWIter25Cont, logWIter25.end());
    EXPECT_EQ(*logWIter25Cont, LogicDigit::LD_W);
    logWIter25Cont++;
  }
  EXPECT_EQ(logWIter25Cont, logWIter25.end());

  std::string refString(randomL2, 200);
  auto randLog = RLELogic::encode(refString);
  size_t i = 0;
  for (auto digit : randLog.boundedIterator(175)) {
    EXPECT_EQ(digit, charToLogicDigit(refString[200 - i - 1]));
    i++;
  }
  EXPECT_EQ(i, 175U);
}

TEST(RLELogicTest, SeekTest) {

  auto rlelog0 = RLELogic::filled<'0'>();
  RLELogic::Offset offset;
  EXPECT_EQ(offset.bytes, 0U);
  EXPECT_EQ(offset.runLength, 0U);

  rlelog0.seek(0, offset);
  EXPECT_EQ(offset.bytes, 0U);
  EXPECT_EQ(offset.runLength, 0U);

  rlelog0.seek(1, offset);
  EXPECT_EQ(offset.bytes, 0U);
  EXPECT_EQ(offset.runLength, 1U);

  rlelog0.seek(10, offset);
  EXPECT_EQ(offset.bytes, 0U);
  EXPECT_EQ(offset.runLength, 11U);

  auto rlelog =
      RLELogic::encode("U" + std::string(40, 'X') + std::string(3, 'W') +
                       "101" + std::string(15, 'L') + "H");

  offset = {0, 0};
  rlelog.seek(0, offset);
  EXPECT_EQ(offset.bytes, 0U);
  EXPECT_EQ(offset.runLength, 0U);

  rlelog.seek(1, offset);
  EXPECT_EQ(offset.bytes, 1U);
  EXPECT_EQ(offset.runLength, 0U);

  rlelog.seek(5, offset);
  EXPECT_EQ(offset.bytes, 1U);
  EXPECT_EQ(offset.runLength, 5U);

  rlelog.seek(3, offset);
  EXPECT_EQ(offset.bytes, 1U);
  EXPECT_EQ(offset.runLength, 8U);

  rlelog.seek(11, offset);
  EXPECT_EQ(offset.bytes, 5U);
  EXPECT_EQ(offset.runLength, 1U);

  auto iter = rlelog.infiniteIterator(offset);
  EXPECT_EQ(*iter, LogicDigit::LD_W);
  iter++;
  EXPECT_EQ(*iter, LogicDigit::LD_W);
  iter++;
  EXPECT_EQ(*iter, LogicDigit::LD_X);

  rlelog.seek(2, offset);
  EXPECT_EQ(offset.bytes, 6U);
  EXPECT_EQ(offset.runLength, 0U);

  rlelog.seek(31, offset);
  EXPECT_EQ(offset.bytes, 7U);
  EXPECT_EQ(offset.runLength, 15U);

  rlelog.seek(50, offset);
  EXPECT_EQ(offset.bytes, 9U);
  EXPECT_EQ(offset.runLength, 41U);
}

TEST(RLELogicTest, XORTest) {
  std::string l9str(l9Values);
  RLELogic l9log = RLELogic::encode(l9str);

  SmallVector<LogicDigit, 9> buffer;

  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'U'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UUUUUUUUU");

  buffer.clear();
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'X'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UXXXXXXXX");

  buffer.clear();
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'0'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UX01XX01X");

  buffer.clear();
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'1'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UX10XX10X");

  buffer.clear();
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'Z'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UXXXXXXXX");

  buffer.clear();
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'W'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UXXXXXXXX");

  buffer.clear();
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'L'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UX01XX01X");

  buffer.clear();
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'H'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UX10XX10X");

  buffer.clear();
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 9, l9log,
                     RLELogic::filled<'-'>());
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UXXXXXXXX");

  buffer.clear();
  RLELogic::unaryOp(logicdigits::lutIeee1164Xor[(unsigned)LogicDigit::LD_1],
                    buffer, 9, l9log);
  EXPECT_STREQ(RLELogic::encode(buffer).toString(9).c_str(), "UX10XX10X");

  buffer.clear();
  std::string l2RandStr(randomL2, 128);
  RLELogic l2Rand = RLELogic::encode(l2RandStr);
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 200, l2Rand, l2Rand);
  EXPECT_EQ(RLELogic::encode(buffer), RLELogic::filled<'0'>());

  buffer.clear();
  RLELogic l9RandA = RLELogic::encode(std::string(randomL9, 100));
  RLELogic l9RandB = RLELogic::encode(std::string(randomL9 + 100, 73));
  RLELogic::binaryOp(logicdigits::lutIeee1164Xor, buffer, 128, l9RandA,
                     l9RandB);
  RLELogic l9Xor = RLELogic::encode(buffer);
  EXPECT_TRUE((l9Xor.containsOnly<'0', '1', 'X', 'U'>()));
}
