//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains unit tests for verifying that FIRRTL dialect attributes
// properly implement walking functionality to visit all nested types and
// attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {

//===----------------------------------------------------------------------===//
// Attribute Walking Tests
//===----------------------------------------------------------------------===//

class AttributeWalkTest : public ::testing::Test {
protected:
  void SetUp() override { context.loadDialect<FIRRTLDialect>(); }

  MLIRContext context;
  Builder builder{&context};
};

TEST_F(AttributeWalkTest, ParamDeclAttrWalk) {
  // Create a parameter declaration attribute with nested type and value
  auto intType = mlir::IntegerType::get(&context, 32);
  auto paramValue = IntegerAttr::get(intType, 42);
  auto nameAttr = StringAttr::get(&context, "testParam");
  auto paramDecl = ParamDeclAttr::get(nameAttr, paramValue);

  SmallVector<Attribute> visitedAttrs;
  SmallVector<Type> visitedTypes;
  paramDecl.walk(
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      },
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      });

  // Verify the parameter declaration and its components are visited
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({nameAttr, paramValue, paramDecl}));
  EXPECT_EQ(llvm::ArrayRef(visitedTypes), ArrayRef<Type>({intType}));
}

TEST_F(AttributeWalkTest, MemoryInitAttrWalk) {
  // Create a memory initialization attribute
  auto filenameAttr = StringAttr::get(&context, "memory.hex");
  auto memInit = MemoryInitAttr::get(&context, filenameAttr, true, false);

  SmallVector<Attribute> visitedAttrs;
  SmallVector<Type> visitedTypes;
  memInit.walk(
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      },
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      });

  // Verify the memory init attribute and filename are visited
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({filenameAttr, memInit}));
  EXPECT_EQ(llvm::ArrayRef(visitedTypes), ArrayRef<Type>({}));
}

TEST_F(AttributeWalkTest, InternalPathAttrWalk) {
  // Create an internal path attribute with a path
  auto pathStringAttr = StringAttr::get(&context, "internal.path");
  auto pathAttr = InternalPathAttr::get(&context, pathStringAttr);

  SmallVector<Attribute> visitedAttrs;
  SmallVector<Type> visitedTypes;
  pathAttr.walk(
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      },
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      });

  // Verify the internal path attribute and path string are visited
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({pathStringAttr, pathAttr}));
  EXPECT_EQ(llvm::ArrayRef(visitedTypes), ArrayRef<Type>({}));
}

TEST_F(AttributeWalkTest, InternalPathAttrEmptyWalk) {
  // Create an internal path attribute without a path
  auto pathAttr = InternalPathAttr::get(&context);

  SmallVector<Attribute> visitedAttrs;
  SmallVector<Type> visitedTypes;
  pathAttr.walk(
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      },
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      });

  // Should still visit the attribute itself
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs), ArrayRef<Attribute>({pathAttr}));
  EXPECT_EQ(llvm::ArrayRef(visitedTypes), ArrayRef<Type>({}));
}

TEST_F(AttributeWalkTest, ParamDeclAttrWithoutValueWalk) {
  // Create a parameter declaration attribute without a value (type only)
  auto paramType = SIntType::get(&context, 32);
  auto nameAttr = StringAttr::get(&context, "typeOnlyParam");
  auto paramDecl = ParamDeclAttr::get(nameAttr, paramType);

  SmallVector<Attribute> visitedAttrs;
  SmallVector<Type> visitedTypes;
  paramDecl.walk(
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      },
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      });

  // Verify the parameter declaration and its components are visited
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({nameAttr, paramDecl}));
  EXPECT_EQ(llvm::ArrayRef(visitedTypes), ArrayRef<Type>({paramType}));
}

TEST_F(AttributeWalkTest, NestedAttributeWalk) {
  // Create a complex nested attribute structure
  auto innerType = mlir::IntegerType::get(&context, 8);
  auto innerValue = IntegerAttr::get(innerType, 123);
  auto nameAttr = StringAttr::get(&context, "innerParam");
  auto innerParam = ParamDeclAttr::get(nameAttr, innerValue);

  // Create an array containing the parameter
  auto arrayAttr = ArrayAttr::get(&context, {innerParam});

  SmallVector<Attribute> visitedAttrs;
  SmallVector<Type> visitedTypes;
  arrayAttr.walk(
      [&](Attribute attr) {
        visitedAttrs.push_back(attr);
        return WalkResult::advance();
      },
      [&](Type type) {
        visitedTypes.push_back(type);
        return WalkResult::advance();
      });

  // Verify all nested components are visited
  EXPECT_EQ(llvm::ArrayRef(visitedAttrs),
            ArrayRef<Attribute>({nameAttr, innerValue, innerParam, arrayAttr}));
  EXPECT_EQ(llvm::ArrayRef(visitedTypes), ArrayRef<Type>({innerType}));
}

} // namespace
