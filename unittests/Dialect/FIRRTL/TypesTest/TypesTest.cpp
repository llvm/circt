//===- TypesTest.cpp - FIRRTL type unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {

TEST(TypesTest, AnalogContainsAnalog) {
  MLIRContext context;
  context.loadDialect<FIRRTLDialect>();
  ASSERT_TRUE(AnalogType::get(&context).containsAnalog());
}

} // namespace
