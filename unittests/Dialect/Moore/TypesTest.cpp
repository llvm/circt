//===- TypesTest.cpp - Moore type unit tests ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace moore;

namespace {

TEST(TypesTest, UnitTypes) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  auto voidType = VoidType::get(&context);
  auto stringType = StringType::get(&context);
  auto chandleType = ChandleType::get(&context);
  auto eventType = EventType::get(&context);

  ASSERT_EQ(voidType.getBitSize(), 0u);
  ASSERT_EQ(stringType.getBitSize(), std::nullopt);
  ASSERT_EQ(chandleType.getBitSize(), std::nullopt);
  ASSERT_EQ(eventType.getBitSize(), std::nullopt);

  ASSERT_EQ(voidType.getDomain(), Domain::TwoValued);
  ASSERT_EQ(stringType.getDomain(), Domain::TwoValued);
  ASSERT_EQ(chandleType.getDomain(), Domain::TwoValued);
  ASSERT_EQ(eventType.getDomain(), Domain::TwoValued);
}

TEST(TypesTest, Ranges) {
  Range a(42);
  Range b(32, RangeDir::Down, -5);
  Range c(16, RangeDir::Up, -3);

  ASSERT_EQ(a.toString(), "41:0");
  ASSERT_EQ(b.toString(), "26:-5");
  ASSERT_EQ(c.toString(), "-3:12");

  ASSERT_EQ(a.left(), 41);
  ASSERT_EQ(a.right(), 0);
  ASSERT_EQ(a.low(), 0);
  ASSERT_EQ(a.high(), 41);
  ASSERT_EQ(a.increment(), -1);

  ASSERT_EQ(b.left(), 26);
  ASSERT_EQ(b.right(), -5);
  ASSERT_EQ(b.low(), -5);
  ASSERT_EQ(b.high(), 26);
  ASSERT_EQ(b.increment(), -1);

  ASSERT_EQ(c.left(), -3);
  ASSERT_EQ(c.right(), 12);
  ASSERT_EQ(c.low(), -3);
  ASSERT_EQ(c.high(), 12);
  ASSERT_EQ(c.increment(), 1);
}

TEST(TypesTest, PackedInt) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  std::tuple<IntType::Kind, StringRef, Domain, Sign> pairs[] = {
      {IntType::Bit, "bit", Domain::TwoValued, Sign::Unsigned},
      {IntType::Logic, "logic", Domain::FourValued, Sign::Unsigned},
      {IntType::Reg, "reg", Domain::FourValued, Sign::Unsigned},
      {IntType::Byte, "byte", Domain::TwoValued, Sign::Signed},
      {IntType::ShortInt, "shortint", Domain::TwoValued, Sign::Signed},
      {IntType::Int, "int", Domain::TwoValued, Sign::Signed},
      {IntType::LongInt, "longint", Domain::TwoValued, Sign::Signed},
      {IntType::Integer, "integer", Domain::FourValued, Sign::Signed},
      {IntType::Time, "time", Domain::TwoValued, Sign::Unsigned},
  };

  for (auto pair : pairs) {
    auto kind = std::get<0>(pair);
    auto type = IntType::get(&context, kind);
    ASSERT_EQ(type.getDomain(), std::get<2>(pair));
  }
}

TEST(TypesTest, Reals) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  auto t0 = RealType::get(&context, RealType::ShortReal);
  auto t1 = RealType::get(&context, RealType::Real);
  auto t2 = RealType::get(&context, RealType::RealTime);

  ASSERT_EQ(t0.getDomain(), Domain::TwoValued);
  ASSERT_EQ(t1.getDomain(), Domain::TwoValued);
  ASSERT_EQ(t2.getDomain(), Domain::TwoValued);

  ASSERT_EQ(t0.getBitSize(), 32u);
  ASSERT_EQ(t1.getBitSize(), 64u);
  ASSERT_EQ(t2.getBitSize(), 64u);
}

TEST(TypesTest, PackedDim) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  auto bitType = IntType::get(&context, IntType::Bit);
  auto arrayType1 = PackedRangeDim::get(bitType, 3);
  auto arrayType2 = PackedRangeDim::get(arrayType1, 2);
  auto arrayType3 = PackedUnsizedDim::get(arrayType2);

  ASSERT_EQ(arrayType1.getRange(), Range(3));
  ASSERT_EQ(arrayType2.getRange(), Range(2));
  ASSERT_EQ(arrayType3.getRange(), std::nullopt);
  ASSERT_EQ(arrayType1.getSize(), 3u);
  ASSERT_EQ(arrayType3.getSize(), std::nullopt);
}

TEST(TypesTest, UnpackedDim) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  auto stringType = StringType::get(&context);
  auto arrayType1 = UnpackedUnsizedDim::get(stringType);
  auto arrayType2 = UnpackedArrayDim::get(arrayType1, 42);
  auto arrayType3 = UnpackedRangeDim::get(arrayType2, 2);
  auto arrayType4 = UnpackedAssocDim::get(arrayType3);
  auto arrayType5 = UnpackedAssocDim::get(arrayType4, stringType);
  auto arrayType6 = UnpackedQueueDim::get(arrayType5);
  auto arrayType7 = UnpackedQueueDim::get(arrayType6, 9);

  ASSERT_EQ(arrayType2.getSize(), 42u);
  ASSERT_EQ(arrayType3.getRange(), Range(2));
  ASSERT_EQ(arrayType4.getIndexType(), UnpackedType{});
  ASSERT_EQ(arrayType5.getIndexType(), stringType);
  ASSERT_EQ(arrayType6.getBound(), std::nullopt);
  ASSERT_EQ(arrayType7.getBound(), 9u);
}

TEST(TypesTest, Structs) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();
  auto foo = StringAttr::get(&context, "foo");
  auto bar = StringAttr::get(&context, "bar");

  auto bitType = IntType::get(&context, IntType::Bit);
  auto logicType = IntType::get(&context, IntType::Logic);
  auto bit8Type = PackedRangeDim::get(bitType, 8);
  auto bitDynArrayType = PackedUnsizedDim::get(bitType);

  auto s0 = UnpackedStructType::get(&context, StructKind::Struct,
                                    {StructMember{foo, bitType}});
  auto s1 = UnpackedStructType::get(
      &context, StructKind::Struct,
      {StructMember{foo, bitType}, StructMember{bar, bit8Type}});
  auto s2 = UnpackedStructType::get(
      &context, StructKind::Struct,
      {StructMember{foo, bitType}, StructMember{bar, logicType}});
  auto s3 = UnpackedStructType::get(
      &context, StructKind::Struct,
      {StructMember{foo, bitType}, StructMember{bar, bitDynArrayType}});

  // Value domain
  ASSERT_EQ(s1.getDomain(), Domain::TwoValued);
  ASSERT_EQ(s2.getDomain(), Domain::FourValued);

  // Bit size
  ASSERT_EQ(s0.getBitSize(), 1u);
  ASSERT_EQ(s1.getBitSize(), 9u);
  ASSERT_EQ(s2.getBitSize(), 2u);
  ASSERT_EQ(s3.getBitSize(), std::nullopt);
}

TEST(TypesTest, SimpleBitVectorTypes) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  // Unpacked types have no SBV equivalent.
  auto stringType = StringType::get(&context);
  ASSERT_FALSE(stringType.isSimpleBitVector());
  ASSERT_FALSE(stringType.isCastableToSimpleBitVector());

  // Void is packed but cannot be cast to an SBV.
  auto voidType = VoidType::get(&context);
  ASSERT_FALSE(voidType.isSimpleBitVector());
  ASSERT_FALSE(voidType.isCastableToSimpleBitVector());

  // Integer atoms with a dimension are no SBVT, but can be cast to one.
  auto intTy = IntType::get(&context, IntType::Int);
  auto intArray = PackedRangeDim::get(intTy, 8);
  ASSERT_FALSE(intArray.isSimpleBitVector());
  ASSERT_TRUE(intArray.isCastableToSimpleBitVector());
  ASSERT_EQ(intArray.castToSimpleBitVector(),
            SimpleBitVectorType(Domain::TwoValued, 256));
}

} // namespace
