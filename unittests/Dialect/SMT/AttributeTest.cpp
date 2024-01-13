//===- AttributeTest.cpp - SMT attribute unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SMT/SMTAttributes.h"
#include "circt/Dialect/SMT/SMTDialect.h"
#include "circt/Dialect/SMT/SMTTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace smt;

namespace {

TEST(AttributeTest, BitVectorAttr) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  auto attr = BitVectorAttr::getChecked(loc, &context, 0U, 0U);
  ASSERT_EQ(attr, BitVectorAttr());
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "bit-width must be at least 1, but got 0");
  });

  attr = BitVectorAttr::get(&context, "#b1010");
  ASSERT_EQ(attr.getValue(), APInt(4, 10));
  ASSERT_EQ(attr.getType(), BitVectorType::get(&context, 4));
}

} // namespace
