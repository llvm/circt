//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
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

} // namespace
