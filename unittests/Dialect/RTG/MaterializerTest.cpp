//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace rtg;

namespace {

TEST(MaterializerTest, ImmediateAttr) {
  MLIRContext context;
  context.loadDialect<RTGDialect>();
  Location loc(UnknownLoc::get(&context));
  auto moduleOp = ModuleOp::create(loc);
  OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

  auto attr = ImmediateAttr::get(&context, APInt(12, 0));
  auto *op0 = context.getLoadedDialect<RTGDialect>()->materializeConstant(
      builder, attr, attr.getType(), loc);
  auto *op1 = context.getLoadedDialect<RTGDialect>()->materializeConstant(
      builder, attr, ImmediateType::get(&context, 2), loc);

  ASSERT_TRUE(op0 && isa<ConstantOp>(op0));
  ASSERT_EQ(op1, nullptr);
}

} // namespace
