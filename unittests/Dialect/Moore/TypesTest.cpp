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

  auto i42 = IntType::getInt(&context, 42);
  auto l42 = IntType::getLogic(&context, 42);

  ASSERT_EQ(i42.getBitSize(), 42u);
  ASSERT_EQ(l42.getBitSize(), 42u);

  ASSERT_EQ(i42.getDomain(), Domain::TwoValued);
  ASSERT_EQ(l42.getDomain(), Domain::FourValued);
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

  auto arrayType1 = IntType::getInt(&context, 3);
  auto arrayType2 = PackedRangeDim::get(arrayType1, 2);
  auto arrayType3 = PackedUnsizedDim::get(arrayType2);

  ASSERT_EQ(arrayType2.getRange(), Range(2));
  ASSERT_EQ(arrayType3.getRange(), std::nullopt);
  ASSERT_EQ(arrayType2.getSize(), 2u);
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

  auto bitType = IntType::getInt(&context, 1);
  auto logicType = IntType::getLogic(&context, 1);
  auto bit8Type = IntType::getInt(&context, 8);
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
  ASSERT_EQ(s0.getDomain(), Domain::TwoValued);
  ASSERT_EQ(s1.getDomain(), Domain::TwoValued);
  ASSERT_EQ(s2.getDomain(), Domain::FourValued);
  ASSERT_EQ(s3.getDomain(), Domain::TwoValued);

  // Bit size
  ASSERT_EQ(s0.getBitSize(), 1u);
  ASSERT_EQ(s1.getBitSize(), 9u);
  ASSERT_EQ(s2.getBitSize(), 2u);
  ASSERT_EQ(s3.getBitSize(), std::nullopt);
}

} // namespace
