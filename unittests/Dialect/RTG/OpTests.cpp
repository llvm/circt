//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace rtg;

namespace {

TEST(SequenceTests, Visibility) {
  MLIRContext context;
  context.loadDialect<RTGDialect>();
  Location loc(UnknownLoc::get(&context));
  auto moduleOp = ModuleOp::create(loc);
  OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  auto sequence =
      SequenceOp::create(builder, loc, "seq", SequenceType::get(&context, {}));

  ASSERT_TRUE(sequence.isPrivate());
  ASSERT_TRUE(sequence.canDiscardOnUseEmpty());
}

} // namespace
