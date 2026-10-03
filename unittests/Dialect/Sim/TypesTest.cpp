//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace sim;

namespace {

// Check that the generic attribute and type walkers see into the alternatives
// of variant types.
TEST(TypesTest, WalkVariantAlternatives) {
  MLIRContext context;
  context.loadDialect<SimDialect>();
  Builder builder(&context);

  auto i8Type = builder.getIntegerType(8);
  auto i16Type = builder.getIntegerType(16);
  auto innerType =
      VariantType::get(&context, {{builder.getStringAttr("x"), i8Type},
                                  {builder.getStringAttr("y"), i16Type}});
  auto outerType =
      VariantType::get(&context, {{builder.getStringAttr("a"), innerType},
                                  {builder.getStringAttr("b"), i8Type}});

  SmallVector<Type> visitedTypes;
  outerType.walk([&](Type type) { visitedTypes.push_back(type); });
  EXPECT_TRUE(llvm::is_contained(visitedTypes, Type(innerType)));
  EXPECT_TRUE(llvm::is_contained(visitedTypes, Type(i8Type)));
  EXPECT_TRUE(llvm::is_contained(visitedTypes, Type(i16Type)));

  // The alternative names are walked as well.
  SmallVector<StringAttr> visitedNames;
  outerType.walk([&](StringAttr name) { visitedNames.push_back(name); });
  for (auto name : {"a", "b", "x", "y"})
    EXPECT_TRUE(llvm::is_contained(visitedNames, builder.getStringAttr(name)));
}

// Check that the generic type replacers can replace types nested within the
// alternatives of variant types.
TEST(TypesTest, ReplaceVariantAlternativeTypes) {
  MLIRContext context;
  context.loadDialect<SimDialect>();
  Builder builder(&context);

  auto i8Type = builder.getIntegerType(8);
  auto i16Type = builder.getIntegerType(16);
  auto i32Type = builder.getIntegerType(32);
  auto innerType =
      VariantType::get(&context, {{builder.getStringAttr("x"), i8Type},
                                  {builder.getStringAttr("y"), i16Type}});
  auto outerType =
      VariantType::get(&context, {{builder.getStringAttr("a"), innerType},
                                  {builder.getStringAttr("b"), i8Type}});

  // Replace i8 with i32 everywhere.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](IntegerType type) -> std::optional<Type> {
    if (type == i8Type)
      return i32Type;
    return std::nullopt;
  });
  auto replaced = cast<VariantType>(replacer.replace(Type(outerType)));

  auto alternatives = replaced.getAlternatives();
  ASSERT_EQ(alternatives.size(), 2u);
  EXPECT_EQ(alternatives[0].name, builder.getStringAttr("a"));
  EXPECT_EQ(alternatives[1].name, builder.getStringAttr("b"));
  EXPECT_EQ(alternatives[1].type, i32Type);

  // The nested variant is rebuilt with its names intact.
  auto replacedInner = cast<VariantType>(alternatives[0].type);
  auto innerAlternatives = replacedInner.getAlternatives();
  ASSERT_EQ(innerAlternatives.size(), 2u);
  EXPECT_EQ(innerAlternatives[0].name, builder.getStringAttr("x"));
  EXPECT_EQ(innerAlternatives[0].type, i32Type);
  EXPECT_EQ(innerAlternatives[1].name, builder.getStringAttr("y"));
  EXPECT_EQ(innerAlternatives[1].type, i16Type);
}

} // namespace
