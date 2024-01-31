//===- InstanceGraphTest.cpp - HW instance graph tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GraphFixture.h"
#include "circt-c/Dialect/HW.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
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

TEST(InstanceGraphCAPITest, PostOrderTraversal) {
  MLIRContext context;

  HWInstanceGraph instanceGraph =
      hwInstanceGraphGet(wrap(fixtures::createModule(&context).getOperation()));

  struct Context {
    size_t i;
    HWInstanceGraphNode topLevelNode;
  };
  auto ctx = Context{0, hwInstanceGraphGetTopLevelNode(instanceGraph)};

  hwInstanceGraphForEachNode(
      instanceGraph,
      [](HWInstanceGraphNode node, void *userData) {
        Context *ctx = reinterpret_cast<Context *>(userData);
        ctx->i++;

        if (ctx->i == 5) {
          ASSERT_EQ(hwInstanceGraphNodeEqual(ctx->topLevelNode, node), true);
          return;
        }

        MlirOperation moduleOp = hwInstanceGraphNodeGetModuleOp(node);
        MlirAttribute moduleNameAttr = mlirOperationGetAttributeByName(
            moduleOp, mlirStringRefCreateFromCString("sym_name"));
        StringRef moduleName = unwrap(mlirStringAttrGetValue(moduleNameAttr));

        switch (ctx->i) {
        case 1:
          ASSERT_EQ("Cat", moduleName);
          break;
        case 2:
          ASSERT_EQ("Bear", moduleName);
          break;
        case 3:
          ASSERT_EQ("Alligator", moduleName);
          break;
        case 4:
          ASSERT_EQ("Top", moduleName);
          break;
        default:
          llvm_unreachable("unexpected i value");
          break;
        }
      },
      &ctx);

  ASSERT_EQ(ctx.i, 5UL);
}

} // namespace
