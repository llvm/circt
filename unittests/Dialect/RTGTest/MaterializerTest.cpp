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
  ASSERT_TRUE(op && isa<CPUDeclOp>(op));
}

TEST(MaterializerTest, ImmediateAttr) {
  MLIRContext context;
  context.loadDialect<RTGTestDialect>();
  Location loc(UnknownLoc::get(&context));
  auto moduleOp = ModuleOp::create(loc);
  OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

  auto attr12 = Imm12Attr::get(&context, 0);
  auto attr21 = Imm32Attr::get(&context, 0);
  auto attr32 = Imm21Attr::get(&context, 0);

  auto *op12 = context.getLoadedDialect<RTGTestDialect>()->materializeConstant(
      builder, attr12, attr12.getType(), loc);
  auto *op21 = context.getLoadedDialect<RTGTestDialect>()->materializeConstant(
      builder, attr21, attr21.getType(), loc);
  auto *op32 = context.getLoadedDialect<RTGTestDialect>()->materializeConstant(
      builder, attr32, attr32.getType(), loc);

  ASSERT_TRUE(op12 && isa<ImmediateOp>(op12));
  ASSERT_TRUE(op21 && isa<ImmediateOp>(op21));
  ASSERT_TRUE(op32 && isa<ImmediateOp>(op32));
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
  ASSERT_TRUE(op && isa<rtg::FixedRegisterOp>(op));
}

} // namespace
