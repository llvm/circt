//===- InstanceGraphTest.cpp - HW instance graph tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GraphFixture.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

TEST(InstanceGraphTest, PostOrderTraversal) {
  MLIRContext context;
  InstanceGraph graph(fixtures::createModule(&context));

  auto range = llvm::post_order(&graph);

  auto it = range.begin();
  ASSERT_EQ("Cat", it->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Bear", it->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Alligator", it->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Top", it->getModule().getModuleName());
  ++it;
  ASSERT_EQ(graph.getTopLevelNode(), *it);
  ++it;
  ASSERT_EQ(range.end(), it);
}

} // namespace
