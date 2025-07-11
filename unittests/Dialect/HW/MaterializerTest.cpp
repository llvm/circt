//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/IR/Builders.h"
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

} // namespace
