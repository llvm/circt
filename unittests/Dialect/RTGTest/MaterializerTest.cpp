//===- MaterializerTest.cpp - RTGTest Dialect Materializer unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "circt/Dialect/RTGTest/IR/RTGTestOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace rtgtest;

namespace {

TEST(MaterializerTest, CPUAttr) {
  MLIRContext context;
  context.loadDialect<RTGTestDialect>();
  Location loc(UnknownLoc::get(&context));
  auto moduleOp = ModuleOp::create(loc);
  OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

  auto attr = CPUAttr::get(&context, 0);
  auto *op = context.getLoadedDialect<RTGTestDialect>()->materializeConstant(
      builder, attr, attr.getType(), loc);
  ASSERT_TRUE(op && isa<rtg::ConstantOp>(op));
}

TEST(MaterializerTest, RegisterAttr) {
  MLIRContext context;
  context.loadDialect<RTGTestDialect>();
  Location loc(UnknownLoc::get(&context));
  auto moduleOp = ModuleOp::create(loc);
  OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

  auto attr = RegZeroAttr::get(&context);
  auto *op = context.getLoadedDialect<RTGTestDialect>()->materializeConstant(
      builder, attr, attr.getType(), loc);
  ASSERT_TRUE(op && isa<rtg::ConstantOp>(op));
}

} // namespace
