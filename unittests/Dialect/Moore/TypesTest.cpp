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

TEST(TypesTest, Ints) {
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

  auto t0 = RealType::get(&context, RealWidth::f64);
  ASSERT_EQ(t0.getDomain(), Domain::TwoValued);
  ASSERT_EQ(t0.getBitSize(), 64u);
}

TEST(TypesTest, PackedArrays) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  auto i3 = IntType::getInt(&context, 3);
  auto l3 = IntType::getLogic(&context, 3);
  auto arrayType1 = ArrayType::get(2, i3);
  auto arrayType2 = OpenArrayType::get(l3);

  // Value domain
  ASSERT_EQ(arrayType1.getDomain(), Domain::TwoValued);
  ASSERT_EQ(arrayType2.getDomain(), Domain::FourValued);

  // Bit size
  ASSERT_EQ(arrayType1.getBitSize(), 6u);
  ASSERT_EQ(arrayType2.getBitSize(), std::nullopt);

  // Element types
  ASSERT_EQ(arrayType1.getElementType(), i3);
  ASSERT_EQ(arrayType2.getElementType(), l3);

  // Other attributes
  ASSERT_EQ(arrayType1.getSize(), 2u);
}

TEST(TypesTest, UnpackedArrays) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  auto logicType = IntType::getLogic(&context, 1);
  auto stringType = StringType::get(&context);
  auto eventType = EventType::get(&context);
  auto arrayType1 = UnpackedArrayType::get(2, logicType);
  auto arrayType2 = OpenUnpackedArrayType::get(stringType);
  auto arrayType3 = AssocArrayType::get(stringType, eventType);
  auto arrayType4 = QueueType::get(stringType, 9);

  // Value domain
  ASSERT_EQ(arrayType1.getDomain(), Domain::FourValued);
  ASSERT_EQ(arrayType2.getDomain(), Domain::TwoValued);
  ASSERT_EQ(arrayType3.getDomain(), Domain::TwoValued);
  ASSERT_EQ(arrayType4.getDomain(), Domain::TwoValued);

  // Bit size
  ASSERT_EQ(arrayType1.getBitSize(), 2u);
  ASSERT_EQ(arrayType2.getBitSize(), std::nullopt);
  ASSERT_EQ(arrayType3.getBitSize(), std::nullopt);
  ASSERT_EQ(arrayType4.getBitSize(), std::nullopt);

  // Element types
  ASSERT_EQ(arrayType1.getElementType(), logicType);
  ASSERT_EQ(arrayType2.getElementType(), stringType);
  ASSERT_EQ(arrayType3.getElementType(), stringType);
  ASSERT_EQ(arrayType4.getElementType(), stringType);

  // Other attributes
  ASSERT_EQ(arrayType1.getSize(), 2u);
  ASSERT_EQ(arrayType3.getIndexType(), eventType);
  ASSERT_EQ(arrayType4.getBound(), 9u);
}

TEST(TypesTest, Structs) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();
  auto foo = StringAttr::get(&context, "foo");
  auto bar = StringAttr::get(&context, "bar");

  auto bitType = IntType::getInt(&context, 1);
  auto logicType = IntType::getLogic(&context, 1);
  auto bit8Type = IntType::getInt(&context, 8);
  auto bitDynArrayType = OpenUnpackedArrayType::get(bitType);

  auto s0 = StructType::get(&context, {StructLikeMember{foo, bitType}});
  auto s1 = StructType::get(&context, {StructLikeMember{foo, bitType},
                                       StructLikeMember{bar, bit8Type}});
  auto s2 = StructType::get(&context, {StructLikeMember{foo, bitType},
                                       StructLikeMember{bar, logicType}});
  auto s3 = UnpackedStructType::get(
      &context,
      {StructLikeMember{foo, bitType}, StructLikeMember{bar, bitDynArrayType}});

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

  auto u0 = UnionType::get(&context, {StructLikeMember{foo, bitType}});
  auto u1 = UnionType::get(&context, {StructLikeMember{foo, bitType},
                                      StructLikeMember{bar, bit8Type}});
  auto u2 = UnionType::get(&context, {StructLikeMember{foo, bitType},
                                      StructLikeMember{bar, logicType}});
  auto u3 = UnpackedUnionType::get(
      &context,
      {StructLikeMember{foo, bitType}, StructLikeMember{bar, bitDynArrayType}});

  // Value domain
  ASSERT_EQ(u0.getDomain(), Domain::TwoValued);
  ASSERT_EQ(u1.getDomain(), Domain::TwoValued);
  ASSERT_EQ(u2.getDomain(), Domain::FourValued);
  ASSERT_EQ(u3.getDomain(), Domain::TwoValued);

  // Bit size
  ASSERT_EQ(u0.getBitSize(), 1u);
  ASSERT_EQ(u1.getBitSize(), 8u);
  ASSERT_EQ(u2.getBitSize(), 1u);
  ASSERT_EQ(u3.getBitSize(), std::nullopt);
}

TEST(TypesTest, Refs) {
  MLIRContext context;
  context.loadDialect<MooreDialect>();

  auto bitType = IntType::getInt(&context, 1);
  auto logicType = IntType::getLogic(&context, 8);
  auto bitRefType = RefType::get(&context, bitType);
  auto logicRefType = RefType::get(&context, logicType);

  // Value domain
  ASSERT_EQ(bitRefType.getDomain(), Domain::TwoValued);
  ASSERT_EQ(logicRefType.getDomain(), Domain::FourValued);

  // Bit size
  ASSERT_EQ(bitRefType.getBitSize(), 1u);
  ASSERT_EQ(logicRefType.getBitSize(), 8u);

  // Nested type
  ASSERT_EQ(bitRefType.getNestedType(), bitType);
  ASSERT_EQ(logicRefType.getNestedType(), logicType);
}

} // namespace
