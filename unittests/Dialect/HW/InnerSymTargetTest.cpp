//===- InnerSymbolTableTest.cpp - HW inner symbol table tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

constexpr StringLiteral testModuleString = R"mlir(
%c0 = hw.constant 0 : i1
%wire0 = hw.wire %c0 : i1
)mlir";

TEST(InnerSymTargetTest, Equality) {
  MLIRContext context;
  context.loadDialect<HWDialect>();

  Block block;
  LogicalResult parseResult =
      parseSourceString(testModuleString, &block, &context);

  ASSERT_TRUE(succeeded(parseResult));

  InnerSymTarget target0(&block.front());
  InnerSymTarget target1(&block.back());

  ASSERT_TRUE(target0 == target0);
  ASSERT_TRUE(target1 == target1);
  ASSERT_TRUE(target0 != target1);
}

} // namespace
