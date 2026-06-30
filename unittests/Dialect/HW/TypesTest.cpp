//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace hw;

namespace {

// Check that the generic attribute and type walkers see into the fields of
// struct and union types.
TEST(TypesTest, WalkStructAndUnionFields) {
  MLIRContext context;
  context.loadDialect<HWDialect>();
  Builder builder(&context);

  auto i8Type = builder.getIntegerType(8);
  auto i16Type = builder.getIntegerType(16);
  auto unionType =
      UnionType::get(&context, {{builder.getStringAttr("x"), i8Type, 0},
                                {builder.getStringAttr("y"), i16Type, 0}});
  auto structType =
      StructType::get(&context, {{builder.getStringAttr("a"), unionType},
                                 {builder.getStringAttr("b"), i8Type}});

  SmallVector<Type> visited;
  structType.walk([&](Type type) { visited.push_back(type); });
  EXPECT_TRUE(llvm::is_contained(visited, Type(unionType)));
  EXPECT_TRUE(llvm::is_contained(visited, Type(i8Type)));
  EXPECT_TRUE(llvm::is_contained(visited, Type(i16Type)));
}

// Check that the generic type replacers can replace types nested within the
// fields of struct and union types.
TEST(TypesTest, ReplaceStructAndUnionFieldTypes) {
  MLIRContext context;
  context.loadDialect<HWDialect>();
  Builder builder(&context);

  auto i8Type = builder.getIntegerType(8);
  auto i16Type = builder.getIntegerType(16);
  auto i32Type = builder.getIntegerType(32);
  auto unionType =
      UnionType::get(&context, {{builder.getStringAttr("x"), i8Type, 0},
                                {builder.getStringAttr("y"), i16Type, 42}});
  auto structType =
      StructType::get(&context, {{builder.getStringAttr("a"), unionType},
                                 {builder.getStringAttr("b"), i8Type}});

  // Replace i8 with i32 everywhere.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](IntegerType type) -> std::optional<Type> {
    if (type == i8Type)
      return i32Type;
    return std::nullopt;
  });
  auto replaced = cast<StructType>(replacer.replace(Type(structType)));

  auto fields = replaced.getElements();
  ASSERT_EQ(fields.size(), 2u);
  EXPECT_EQ(fields[0].name, builder.getStringAttr("a"));
  EXPECT_EQ(fields[1].name, builder.getStringAttr("b"));
  EXPECT_EQ(fields[1].type, i32Type);

  // The union field offsets must survive the replacement.
  auto replacedUnion = cast<UnionType>(fields[0].type);
  auto unionFields = replacedUnion.getElements();
  ASSERT_EQ(unionFields.size(), 2u);
  EXPECT_EQ(unionFields[0].name, builder.getStringAttr("x"));
  EXPECT_EQ(unionFields[0].type, i32Type);
  EXPECT_EQ(unionFields[0].offset, 0u);
  EXPECT_EQ(unionFields[1].type, i16Type);
  EXPECT_EQ(unionFields[1].offset, 42u);
}

} // namespace
