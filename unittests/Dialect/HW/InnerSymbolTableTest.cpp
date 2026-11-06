//===- InnerSymbolTableTest.cpp - HW inner symbol table tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

constexpr StringLiteral testModuleString = R"mlir(
hw.module @foo(in %in : i1 {hw.exportPort = #hw<innerSym@port0>}) {
  %wire0 = hw.wire %in sym @wire0 : i1
  %wire1 = hw.wire %in : i1
  hw.output
}
)mlir";

TEST(InnerSymbolTableTest, Create) {
  MLIRContext context;
  context.loadDialect<HWDialect>();

  Block block;
  LogicalResult parseResult =
      parseSourceString(testModuleString, &block, &context);

  ASSERT_TRUE(succeeded(parseResult));

  Operation *testOp = &block.front();

  InnerSymbolTable innerSymbolTable(testOp);

  ASSERT_TRUE(innerSymbolTable.lookup("port0"));
  ASSERT_TRUE(innerSymbolTable.lookup("wire0"));
}

TEST(InnerSymbolTableTest, Add) {
  MLIRContext context;
  context.loadDialect<HWDialect>();

  Block block;
  LogicalResult parseResult =
      parseSourceString(testModuleString, &block, &context);

  ASSERT_TRUE(succeeded(parseResult));

  Operation *testOp = &block.front();

  InnerSymbolTable innerSymbolTable(testOp);

  auto name = StringAttr::get(&context, "wire1");

  ASSERT_FALSE(innerSymbolTable.lookup(name));

  Operation *wire1 = testOp->getRegions().front().front().front().getNextNode();

  auto innerSymTarget = InnerSymTarget{wire1};

  LogicalResult result1 = innerSymbolTable.add(name, innerSymTarget);

  ASSERT_TRUE(succeeded(result1));
  ASSERT_TRUE(innerSymbolTable.lookup(name));

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.getSeverity(), DiagnosticSeverity::Error);
    ASSERT_EQ(diag.str(), "redefinition of inner symbol named 'wire1'");
  });

  LogicalResult result2 = innerSymbolTable.add(name, innerSymTarget);

  ASSERT_FALSE(succeeded(result2));
}

} // namespace
