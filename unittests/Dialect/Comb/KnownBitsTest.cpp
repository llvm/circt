//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/KnownBits.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace comb;

namespace {

const char *ir = R"MLIR(
    hw.module @basic(in %arg0: i8, in %arg1: i8, in %arg2: i4, in %arg3: i4){
      // === CONSTANT TESTS ===
      %c0_i4 = hw.constant 0 : i4        // 0000
      %c255_i8 = hw.constant 255 : i8    // 11111111
      %c170_i8 = hw.constant 170 : i8    // 10101010
      %c85_i8 = hw.constant 85 : i8      // 01010101
      %c1_i8 = hw.constant 1 : i8        // 00000001
      %c128_i8 = hw.constant 128 : i8    // 10000000

      // Utility constants for tests
      %c0_i8 = hw.constant 0 : i8        // 0000000
      %c2_i8 = hw.constant 2 : i8        // 010
      %c3_i8 = hw.constant 3 : i8        // 011
      %c15_i8 = hw.constant 15 : i8      // 00001111
      %c240_i8 = hw.constant 240 : i8    // 11110000
      

      // === EXTRACT TESTS ===
      // Extract from constant - should know all bits
      %extract1 = comb.extract %c170_i8 from 2 : (i8) -> i4  // Should be 1010

      // Extract single bit from constant
      %extract2 = comb.extract %c128_i8 from 7 : (i8) -> i1  // Should be 1

      // Extract from unknown input
      %extract3 = comb.extract %arg0 from 0 : (i8) -> i3     // Lower 3 bits unknown

      // === CONCAT TESTS ===
      // Concat constants - should know all bits
      %concat1 = comb.concat %c15_i8, %c0_i4 : i8, i4  // {00001111, 0000}

      // Concat known and unknown
      %concat2 = comb.concat %c255_i8, %arg0 : i8, i8        // {11111111, ????????}

      // === LEFT SHIFT TESTS ===
      // Shift constant by constant - should know all bits
      %shl1 = comb.shl %c85_i8, %c2_i8 : i8                 // 01010101 << 2 = 01010100

      // Shift unknown by constant - should know low-bits
      %shl2 = comb.shl %arg0, %c3_i8 : i8                   // ???????? << 3 = ?????000

      // Shift constant by unknown amount
      %shl3 = comb.shl %c1_i8, %arg1 : i8                   // 00000001 << ?

      // === RIGHT SHIFT TESTS ===
      // Logical right shift constant by constant
      %shr1 = comb.shru %c170_i8, %c2_i8 : i8               // 10101010 >> 2 = 00101010

      // Shift unknown by constant - should know leading zeros
      %shr2 = comb.shru %arg0, %c3_i8 : i8                  // ???????? >> 3 = 000?????

      // === AND TESTS ===
      // AND with all-zeros - result is always zero
      %and1 = comb.and %arg0, %c0_i8 : i8                   // ???????? & 00000000 = 00000000

      // AND with alternating pattern
      %and2 = comb.and %arg0, %c170_i8 : i8                 // ???????? & 10101010 = ?0?0?0?0

      // AND two constants
      %and3 = comb.and %c85_i8, %c170_i8 : i8               // 01010101 & 10101010 = 00000000

      // === OR TESTS ===
      // OR with all-ones - result is always all-ones
      %or1 = comb.or %arg0, %c255_i8 : i8                   // ???????? | 11111111 = 11111111

      // OR with alternating pattern
      %or2 = comb.or %arg0, %c85_i8 : i8                    // ???????? | 01010101 = ?1?1?1?1

      // OR two constants
      %or3 = comb.or %c15_i8, %c240_i8 : i8                 // 00001111 | 11110000 = 11111111

      // === XOR TESTS ===
      // XOR with all-zeros - result is unchanged
      %xor1 = comb.xor %arg0, %c0_i8 : i8                   // ???????? ^ 00000000 = ????????

      // XOR with alternating pattern
      %xor2 = comb.xor %arg0, %c170_i8 : i8                 // ???????? ^ 10101010 = ????????

      // XOR two constants
      %xor3 = comb.xor %c85_i8, %c170_i8 : i8               // 01010101 ^ 10101010 = 11111111

    }

    )MLIR";

TEST(KnownBitsTest, BasicTest) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();

  // Parse the IR string into a module
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  // Find the 'basic' module
  SymbolTable symbolTable(module.get());
  auto basicModule = symbolTable.lookup<hw::HWModuleOp>("basic");
  ASSERT_TRUE(basicModule);
  auto it = basicModule.getBodyBlock()->begin();

  // === CONSTANT TESTS ===
  auto c04 = cast<hw::ConstantOp>(*it++);  // 0000
  auto c255 = cast<hw::ConstantOp>(*it++); // 11111111
  auto c170 = cast<hw::ConstantOp>(*it++); // 10101010
  auto c85 = cast<hw::ConstantOp>(*it++);  // 01010101
  auto c1 = cast<hw::ConstantOp>(*it++);   // 00000001
  auto c128 = cast<hw::ConstantOp>(*it++); // 10000000

  ASSERT_TRUE(comb::computeKnownBits(c04).isZero());
  ASSERT_TRUE(comb::computeKnownBits(c255).isAllOnes());
  EXPECT_EQ(comb::computeKnownBits(c170).Zero, APInt(8, 85));
  EXPECT_EQ(comb::computeKnownBits(c170).One, APInt(8, 170));
  EXPECT_EQ(comb::computeKnownBits(c85).Zero, APInt(8, 170));
  EXPECT_EQ(comb::computeKnownBits(c85).One, APInt(8, 85));
  EXPECT_EQ(comb::computeKnownBits(c1).Zero, APInt(8, 254));
  EXPECT_EQ(comb::computeKnownBits(c1).One, APInt(8, 1));
  EXPECT_EQ(comb::computeKnownBits(c128).Zero, APInt(8, 127));
  EXPECT_EQ(comb::computeKnownBits(c128).One, APInt(8, 128));

  // Skip remaining utility constants
  while (isa<hw::ConstantOp>(*it))
    ++it;

  // === EXTRACT TESTS ===
  auto extract1 = cast<comb::ExtractOp>(*it++); // 1010
  auto extract2 = cast<comb::ExtractOp>(*it++); // 1
  auto extract3 = cast<comb::ExtractOp>(*it++); // ???

  EXPECT_EQ(comb::computeKnownBits(extract1).Zero, APInt(4, 5)); // 0101
  EXPECT_EQ(comb::computeKnownBits(extract1).One, APInt(4, 10)); // 1010
  ASSERT_TRUE(comb::computeKnownBits(extract2).isAllOnes());     // 1
  ASSERT_TRUE(comb::computeKnownBits(extract3).isUnknown());

  // === CONCATENATION TESTS ===
  auto concat1 = cast<comb::ConcatOp>(*it++); // {00001111, 0000}
  auto concat2 = cast<comb::ConcatOp>(*it++); // {11111111, ????????}
  EXPECT_EQ(comb::computeKnownBits(concat1).One, APInt(12, 240));
  EXPECT_EQ(comb::computeKnownBits(concat2).Zero, APInt(16, 0));
  EXPECT_EQ(comb::computeKnownBits(concat2).One, APInt(16, 65280));

  // === LOGICAL LEFT SHIFT TESTS ===
  auto shl1 = cast<comb::ShlOp>(*it++); // 01010101 << 2 = 01010100
  auto shl2 = cast<comb::ShlOp>(*it++); // ???????? << 3 = ?????000
  auto shl3 = cast<comb::ShlOp>(*it++); // 00000001 << ?
  EXPECT_EQ(comb::computeKnownBits(shl1).Zero, APInt(8, 171));
  EXPECT_EQ(comb::computeKnownBits(shl1).One, APInt(8, 84));
  EXPECT_EQ(comb::computeKnownBits(shl2).Zero, APInt(8, 7));
  ASSERT_TRUE(comb::computeKnownBits(shl2).One.isZero());
  ASSERT_TRUE(comb::computeKnownBits(shl3).isUnknown());

  // === LOGICAL RIGHT SHIFT TESTS ===
  auto shr1 = cast<comb::ShrUOp>(*it++); // 10101010 >> 2 = 00101010
  auto shr2 = cast<comb::ShrUOp>(*it++); // ???????? >> 3 = 000?????
  EXPECT_EQ(comb::computeKnownBits(shr1).Zero, APInt(8, 213));
  EXPECT_EQ(comb::computeKnownBits(shr1).One, APInt(8, 42));
  EXPECT_EQ(comb::computeKnownBits(shr2).Zero, APInt(8, 224));
  ASSERT_TRUE(comb::computeKnownBits(shr2).One.isZero());

  // === AND TESTS ===
  auto and1 = cast<comb::AndOp>(*it++); // ???????? & 00000000 = 00000000
  auto and2 = cast<comb::AndOp>(*it++); // ???????? & 10101010 = ?0?0?0?0
  auto and3 = cast<comb::AndOp>(*it++); // 01010101 & 10101010 = 00000000
  ASSERT_TRUE(comb::computeKnownBits(and1).isZero());
  EXPECT_EQ(comb::computeKnownBits(and2).Zero, APInt(8, 85));
  ASSERT_TRUE(comb::computeKnownBits(and2).One.isZero());
  ASSERT_TRUE(comb::computeKnownBits(and3).isZero());

  // === OR TESTS ===
  auto or1 = cast<comb::OrOp>(*it++); // ???????? | 11111111 = 11111111
  auto or2 = cast<comb::OrOp>(*it++); // ???????? | 01010101 = ?1?1?1?1
  auto or3 = cast<comb::OrOp>(*it++); // 00001111 | 11110000 = 11111111
  ASSERT_TRUE(comb::computeKnownBits(or1).isAllOnes());
  EXPECT_EQ(comb::computeKnownBits(or2).One, APInt(8, 85));
  ASSERT_TRUE(comb::computeKnownBits(or2).Zero.isZero());
  ASSERT_TRUE(comb::computeKnownBits(or3).isAllOnes());

  // === XOR TESTS ===
  auto xor1 = cast<comb::XorOp>(*it++); // ???????? ^ 00000000 = ????????
  auto xor2 = cast<comb::XorOp>(*it++); // ???????? ^ 10101010 = ????????
  auto xor3 = cast<comb::XorOp>(*it++); // 01010101 ^ 10101010 = 11111111
  ASSERT_TRUE(comb::computeKnownBits(xor1).isUnknown());
  ASSERT_TRUE(comb::computeKnownBits(xor2).isUnknown());
  ASSERT_TRUE(comb::computeKnownBits(xor3).isAllOnes());
}

} // namespace
