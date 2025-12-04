//===- TypesTest.cpp - FIRRTL type unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {

TEST(TypesTest, AnalogContainsAnalog) {
  MLIRContext context;
  context.loadDialect<FIRRTLDialect>();
  ASSERT_TRUE(AnalogType::get(&context).containsAnalog());
}

TEST(TypesTest, TypeAliasCast) {
  MLIRContext context;
  context.loadDialect<FIRRTLDialect>();
  // Check ContainAliasableTypes.
  static_assert(!ContainAliasableTypes<FIRRTLType>::value);
  // Return false for FIRRTLBaseType.
  static_assert(!ContainAliasableTypes<FIRRTLBaseType>::value);
  static_assert(!ContainAliasableTypes<StringType>::value);
  static_assert(ContainAliasableTypes<FVectorType>::value);
  static_assert(ContainAliasableTypes<UIntType, StringType>::value);
  static_assert(ContainAliasableTypes<hw::FieldIDTypeInterface>::value);
  AnalogType analog = AnalogType::get(&context);
  BaseTypeAliasType alias1 =
      BaseTypeAliasType::get(StringAttr::get(&context, "foo"), analog);
  BaseTypeAliasType alias2 =
      BaseTypeAliasType::get(StringAttr::get(&context, "bar"), alias1);
  ASSERT_TRUE(!type_isa<FVectorType>(analog));
  ASSERT_TRUE(type_isa<AnalogType>(analog));
  ASSERT_TRUE(type_isa<AnalogType>(alias1));
  ASSERT_TRUE(type_isa<AnalogType>(alias2));
  ASSERT_TRUE(!type_isa<FVectorType>(alias2));
  ASSERT_TRUE((type_isa<AnalogType, StringType>(alias2)));
}

//===----------------------------------------------------------------------===//
// Type Walking Tests
//===----------------------------------------------------------------------===//

class TypeWalkTest : public ::testing::Test {
protected:
  void SetUp() override { context.loadDialect<FIRRTLDialect>(); }

  MLIRContext context;
  Builder builder{&context};
};

TEST_F(TypeWalkTest, BundleTypeWalk) {
  // Create a nested bundle type with various element types
  auto uint8Type = UIntType::get(&context, 8);
  auto sint16Type = SIntType::get(&context, 16);
  auto clockType = ClockType::get(&context);

  auto aAttr = StringAttr::get(&context, "a");
  auto bAttr = StringAttr::get(&context, "b");
  auto clkAttr = StringAttr::get(&context, "clk");

  SmallVector<BundleType::BundleElement> elements = {
      {aAttr, false, uint8Type},
      {bAttr, true, sint16Type},
      {clkAttr, false, clockType}};

  auto bundleType = BundleType::get(&context, elements);

  // Walk the bundle type and collect visited types and attributes
  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  bundleType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({uint8Type, sint16Type, clockType, bundleType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({aAttr, bAttr, clkAttr}));
}

TEST_F(TypeWalkTest, FVectorTypeWalk) {
  // Create a vector type with nested element type
  auto elementType = UIntType::get(&context, 32);
  auto vectorType = FVectorType::get(elementType, 4);

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  vectorType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Verify both vector type and element type are visited
  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({elementType, vectorType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs), ArrayRef<Attribute>({}));
}

TEST_F(TypeWalkTest, NestedBundleTypeWalk) {
  // Create a bundle containing another bundle (nested structure)
  auto uint8Type = UIntType::get(&context, 8);
  auto innerAttr = StringAttr::get(&context, "inner");
  auto nestedAttr = StringAttr::get(&context, "nested");
  auto simpleAttr = StringAttr::get(&context, "simple");

  auto innerBundle = BundleType::get(&context, {{innerAttr, false, uint8Type}});

  auto outerBundle =
      BundleType::get(&context, {{nestedAttr, false, innerBundle},
                                 {simpleAttr, false, uint8Type}});

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  outerBundle.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({uint8Type, innerBundle, outerBundle}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({nestedAttr, innerAttr, simpleAttr}));
}

TEST_F(TypeWalkTest, RefTypeWalk) {
  // Create a ref type with nested base type
  auto baseType = UIntType::get(&context, 16);
  auto refType = RefType::get(baseType, false);

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  refType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Verify both ref type and base type are visited
  EXPECT_EQ(llvm::ArrayRef(visitedTypes), ArrayRef<Type>({baseType, refType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs), ArrayRef<Attribute>({}));
}

TEST_F(TypeWalkTest, BaseTypeAliasWalk) {
  // Create a type alias with nested inner type
  auto innerType = SIntType::get(&context, 32);
  auto nameAttr = StringAttr::get(&context, "MyAlias");
  auto aliasType = BaseTypeAliasType::get(nameAttr, innerType);

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  aliasType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Verify both alias type and inner type are visited
  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({innerType, aliasType}));

  // Verify the name attribute is visited
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs), ArrayRef<Attribute>({nameAttr}));
}

//===----------------------------------------------------------------------===//
// Edge Case Tests
//===----------------------------------------------------------------------===//

TEST_F(TypeWalkTest, EmptyBundleWalk) {
  // Test walking an empty bundle type
  auto emptyBundle = BundleType::get(&context, {});

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  emptyBundle.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Should still visit the bundle type itself
  EXPECT_EQ(llvm::ArrayRef(visitedTypes), ArrayRef<Type>({emptyBundle}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs), ArrayRef<Attribute>({}));
}

TEST_F(TypeWalkTest, WalkOrderTest) {
  // Test that walking respects order and visits elements correctly
  auto uint8Type = UIntType::get(&context, 8);
  auto vectorType = FVectorType::get(uint8Type, 2);

  SmallVector<Type> preOrderTypes;
  SmallVector<Type> postOrderTypes;
  SmallVector<Attribute> preOrderAttrs;
  SmallVector<Attribute> postOrderAttrs;

  // Pre-order walk
  vectorType.walk<WalkOrder::PreOrder>(
      [&](Type type) {
        preOrderTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        preOrderAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Post-order walk
  vectorType.walk<WalkOrder::PostOrder>(
      [&](Type type) {
        postOrderTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        postOrderAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Both should visit the same types, but in different orders
  EXPECT_EQ(llvm::ArrayRef(preOrderTypes),
            ArrayRef<Type>({vectorType, uint8Type}));
  EXPECT_EQ(llvm::ArrayRef(postOrderTypes),
            ArrayRef<Type>({uint8Type, vectorType}));
  EXPECT_EQ(llvm::ArrayRef(preOrderAttrs), ArrayRef<Attribute>({}));
  EXPECT_EQ(llvm::ArrayRef(postOrderAttrs), ArrayRef<Attribute>({}));
}

TEST_F(TypeWalkTest, FEnumTypeWalk) {
  // Create an enum type with various element types
  auto uint8Type = UIntType::get(&context, 8);
  auto sint16Type = SIntType::get(&context, 16);

  auto intType = mlir::IntegerType::get(&context, 2);
  auto strAttrA = StringAttr::get(&context, "A");
  auto strAttrB = StringAttr::get(&context, "B");
  auto intAttrA = IntegerAttr::get(intType, 0);
  auto intAttrB = IntegerAttr::get(intType, 1);

  SmallVector<FEnumType::EnumElement> elements = {
      {strAttrA, intAttrA, uint8Type}, {strAttrB, intAttrB, sint16Type}};

  auto enumType = FEnumType::get(&context, elements);

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  enumType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({intType, uint8Type, sint16Type, enumType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({strAttrA, intAttrA, strAttrB, intAttrB}));
}

TEST_F(TypeWalkTest, OpenBundleTypeWalk) {
  // Create an open bundle type (similar to bundle but allows different types)
  auto uint32Type = UIntType::get(&context, 32);
  auto stringType = StringType::get(&context);

  auto dataAttr = StringAttr::get(&context, "data");
  auto nameAttr = StringAttr::get(&context, "name");

  SmallVector<OpenBundleType::BundleElement> elements = {
      {dataAttr, false, uint32Type}, {nameAttr, false, stringType}};

  auto openBundleType = OpenBundleType::get(&context, elements);

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  openBundleType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({uint32Type, stringType, openBundleType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({dataAttr, nameAttr}));
}

TEST_F(TypeWalkTest, OpenVectorTypeWalk) {
  // Create an open vector type
  auto elementType = StringType::get(&context);
  auto openVectorType = OpenVectorType::get(elementType, 3);

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  openVectorType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Verify both vector type and element type are visited
  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({elementType, openVectorType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs), ArrayRef<Attribute>({}));
}

TEST_F(TypeWalkTest, LHSTypeWalk) {
  // Create an LHS type wrapping another type
  auto wrappedType = UIntType::get(&context, 64);
  auto lhsType = LHSType::get(wrappedType);

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  lhsType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Verify both LHS type and wrapped type are visited
  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({wrappedType, lhsType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs), ArrayRef<Attribute>({}));
}

TEST_F(TypeWalkTest, ListTypeWalk) {
  // Create a list type with element type (ListType requires PropertyType)
  auto elementType = StringType::get(&context);
  auto listType = ListType::get(&context, cast<PropertyType>(elementType));

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  listType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  // Verify both list type and element type are visited
  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({elementType, listType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs), ArrayRef<Attribute>({}));
}

TEST_F(TypeWalkTest, ComplexNestedStructureWalk) {
  // Create a complex nested structure to test deep walking
  auto uint8Type = UIntType::get(&context, 8);
  auto uint16Type = UIntType::get(&context, 16);

  auto innerFieldAttr = StringAttr::get(&context, "inner_field");
  auto vectorFieldAttr = StringAttr::get(&context, "vector_field");
  auto simpleFieldAttr = StringAttr::get(&context, "simple_field");
  auto complexTypeNameAttr = StringAttr::get(&context, "ComplexType");

  // Create inner bundle
  auto innerBundle =
      BundleType::get(&context, {{innerFieldAttr, false, uint8Type}});

  // Create vector of the inner bundle
  auto vectorOfBundles = FVectorType::get(innerBundle, 2);

  // Create outer bundle containing the vector
  auto outerBundle =
      BundleType::get(&context, {{vectorFieldAttr, false, vectorOfBundles},
                                 {simpleFieldAttr, false, uint16Type}});

  // Create alias for the outer bundle
  auto aliasType = BaseTypeAliasType::get(complexTypeNameAttr, outerBundle);

  SmallVector<Type> visitedTypes;
  SmallVector<Attribute> visitedAttrs;
  aliasType.walk(
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      },
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      });

  EXPECT_EQ(llvm::ArrayRef(visitedTypes),
            ArrayRef<Type>({uint8Type, innerBundle, vectorOfBundles, uint16Type,
                            outerBundle, aliasType}));
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({complexTypeNameAttr, vectorFieldAttr,
                                 innerFieldAttr, simpleFieldAttr}));
}

} // namespace
