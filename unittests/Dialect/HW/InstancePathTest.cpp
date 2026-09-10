//===- InstancePathTest.cpp - HW instance path tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GraphFixture.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Support/InstanceGraph.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

TEST(InstancePathTest, Enumerate) {
  MLIRContext context;
  ModuleOp circuit = fixtures::createModule(&context);
  hw::InstanceGraph graph(circuit);
  igraph::InstancePathCache pathCache(graph);

  auto cat = cast<HWModuleOp>(*circuit.getBody()->rbegin());
  auto catPaths = pathCache.getAbsolutePaths(cat);
  ASSERT_EQ(2ull, catPaths.size());

  ASSERT_EQ(3ull, catPaths[0].size());
  EXPECT_EQ("alligator", catPaths[0][0].getInstanceName());
  EXPECT_EQ("bear", catPaths[0][1].getInstanceName());
  EXPECT_EQ("cat", catPaths[0][2].getInstanceName());

  ASSERT_EQ(1ull, catPaths[1].size());
  EXPECT_EQ("cat", catPaths[1][0].getInstanceName());

  auto top = cast<HWModuleOp>(*circuit.getBody()->begin());
  auto topPaths = pathCache.getAbsolutePaths(top);
  ASSERT_EQ(1ull, topPaths.size());
}

TEST(InstancePathTest, RelativePath) {
  MLIRContext context;
  ModuleOp circuit = fixtures::createModule(&context);
  hw::InstanceGraph graph(circuit);
  igraph::InstancePathCache pathCache(graph);

  auto cat = cast<HWModuleOp>(*circuit.getBody()->rbegin());
  auto alligator = cast<HWModuleOp>(*std::next(circuit.getBody()->begin()));
  auto catToAlligatorPaths = pathCache.getRelativePaths(cat, graph[alligator]);

  ASSERT_EQ(1ull, catToAlligatorPaths.size());

  ASSERT_EQ(2ull, catToAlligatorPaths[0].size());
  EXPECT_EQ("bear", catToAlligatorPaths[0][0].getInstanceName());
  EXPECT_EQ("cat", catToAlligatorPaths[0][1].getInstanceName());
}

TEST(InstancePathTest, AppendPrependConcatInstance) {
  MLIRContext context;
  ModuleOp circuit = fixtures::createModule(&context);
  hw::InstanceGraph graph(circuit);
  igraph::InstancePathCache pathCache(graph);

  igraph::InstancePath empty;

  auto top = cast<HWModuleOp>(*circuit.getBody()->begin());
  auto cat = cast<HWModuleOp>(*circuit.getBody()->rbegin());

  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(circuit.getLoc(), circuit.getBody());
  auto dragon = HWModuleOp::create(builder, StringAttr::get(&context, "Dragon"),
                                   ArrayRef<PortInfo>{});

  builder.setInsertionPointToStart(dragon.getBodyBlock());
  auto breakfast =
      InstanceOp::create(builder, cat, "breakfast", ArrayRef<Value>{});

  builder.setInsertionPointToStart(top.getBodyBlock());
  auto kitty = InstanceOp::create(builder, cat, "kitty", ArrayRef<Value>{});

  auto prepended = pathCache.prependInstance(
      breakfast, pathCache.prependInstance(kitty, empty));
  ASSERT_EQ(2ull, prepended.size());
  EXPECT_EQ(breakfast, prepended[0]);
  EXPECT_EQ(kitty, prepended[1]);

  auto appended = pathCache.appendInstance(
      pathCache.appendInstance(empty, breakfast), kitty);
  ASSERT_EQ(2ull, appended.size());
  EXPECT_EQ(breakfast, appended[0]);
  EXPECT_EQ(kitty, appended[1]);

  auto concatenated =
      pathCache.concatPath(pathCache.appendInstance(empty, breakfast),
                           pathCache.appendInstance(empty, kitty));
  ASSERT_EQ(2ull, concatenated.size());
  EXPECT_EQ(breakfast, concatenated[0]);
  EXPECT_EQ(kitty, concatenated[1]);
}

TEST(InstancePathTest, Hash) {
  MLIRContext context;
  ModuleOp circuit = fixtures::createModule(&context);
  hw::InstanceGraph graph(circuit);
  igraph::InstancePathCache pathCache(graph);
  llvm::DenseSet<igraph::InstancePath> tests;
  // An empty path is *valid*, not the same as an empty key.
  ASSERT_TRUE(tests.insert({}).second);

  auto top = cast<HWModuleOp>(*circuit.getBody()->begin());
  auto cat = cast<HWModuleOp>(*circuit.getBody()->rbegin());
  auto path = pathCache.getAbsolutePaths(cat);
  // Make sure all paths are inserted.
  for (auto p : path)
    ASSERT_TRUE(tests.insert(p).second);

  hw::InstanceOp topCat;
  top.walk([&](hw::InstanceOp op) {
    if (op.getReferencedModuleNameAttr() == cat.getModuleNameAttr()) {
      topCat = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  ASSERT_TRUE(topCat);
  ASSERT_TRUE(tests.contains(pathCache.appendInstance({}, topCat)));
}

} // namespace
