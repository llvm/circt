//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace hw;

namespace {

TEST(MaterializerTest, ImmediateAttr) {
  MLIRContext context;
  context.loadDialect<HWDialect>();
  Location loc(UnknownLoc::get(&context));
  OpBuilder builder(&context);

  // Check that we don't crash on non-sensical materializations.
  auto attr = builder.getI8IntegerAttr(42);
  auto type = builder.getF64Type();
  context.getLoadedDialect<HWDialect>()->materializeConstant(builder, attr,
                                                             type, loc);
}

TEST(MaterializerTest, ParamAttr) {
  MLIRContext context;
  context.loadDialect<HWDialect>();
  Location loc(UnknownLoc::get(&context));
  OpBuilder builder(&context);
  // Set up a block without a parent op.
  Block block;
  builder.setInsertionPointToStart(&block);

  // Check that we don't crash on parameter materializations.
  auto attr = hw::ParamVerbatimAttr::get(builder.getStringAttr("123"));
  auto type = builder.getI16Type();
  context.getLoadedDialect<HWDialect>()->materializeConstant(builder, attr,
                                                             type, loc);
}

TEST(MaterializerTest, IntegerConstant) {
  MLIRContext context;
  context.loadDialect<HWDialect>();
  Location loc(UnknownLoc::get(&context));
  OpBuilder builder(&context);
  Block block;
  builder.setInsertionPointToStart(&block);
  auto *hwdialect = context.getLoadedDialect<HWDialect>();
  auto i5 = builder.getIntegerType(5);
  auto ui5 = builder.getIntegerType(5, /*isSigned=*/false);

  // Values which differ from the result type only in signedness are given the
  // result type.
  auto *op = hwdialect->materializeConstant(
      builder, builder.getIntegerAttr(ui5, 3), i5, loc);
  auto constOp = dyn_cast_or_null<ConstantOp>(op);
  ASSERT_TRUE(constOp);
  EXPECT_EQ(constOp.getValueAttr().getType(), i5);
  EXPECT_EQ(constOp.getType(), i5);
  EXPECT_TRUE(succeeded(constOp.verify()));

  // Aliases of integer types are materialized, with the value having the
  // canonical type.
  auto alias = TypeAliasType::get(
      SymbolRefAttr::get(builder.getStringAttr("ns"),
                         {FlatSymbolRefAttr::get(builder.getStringAttr("t"))}),
      i5);
  op = hwdialect->materializeConstant(builder, builder.getIntegerAttr(i5, 3),
                                      alias, loc);
  constOp = dyn_cast_or_null<ConstantOp>(op);
  ASSERT_TRUE(constOp);
  EXPECT_EQ(constOp.getValueAttr().getType(), i5);
  EXPECT_EQ(constOp.getType(), alias);
  EXPECT_TRUE(succeeded(constOp.verify()));

  // Width mismatches and non-signless result types can't be materialized.
  EXPECT_EQ(
      hwdialect->materializeConstant(
          builder, builder.getIntegerAttr(builder.getI8Type(), 3), i5, loc),
      nullptr);
  EXPECT_EQ(hwdialect->materializeConstant(
                builder, builder.getIntegerAttr(ui5, 3), ui5, loc),
            nullptr);
}

} // namespace
