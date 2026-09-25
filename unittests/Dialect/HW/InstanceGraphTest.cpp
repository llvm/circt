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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"
#include <atomic>
#include <mutex>

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

TEST(InstanceGraphTest, PostOrderTraversal) {
  MLIRContext context;
  InstanceGraph graph(fixtures::createModule(&context));

  auto range = llvm::post_order(&graph);

  auto it = range.begin();
  ASSERT_EQ("Cat", (*it)->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Bear", (*it)->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Alligator", (*it)->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Top", (*it)->getModule().getModuleName());
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

// Verify that walkParallelPostOrder visits every node exactly once.
TEST(InstanceGraphTest, ParallelPostOrderVisitsAll) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(fixtures::createModule(&context));

  std::mutex mu;
  llvm::SmallVector<std::string> visited;
  graph.walkParallelPostOrder([&](igraph::InstanceGraphNode &node) {
    std::lock_guard<std::mutex> lock(mu);
    visited.push_back(node.getModule().getModuleName().str());
  });

  // Every module must appear exactly once.
  ASSERT_EQ(4u, visited.size());
  for (const char *name : {"Top", "Alligator", "Bear", "Cat"}) {
    EXPECT_EQ(1, llvm::count(visited, name)) << name << " not visited once";
  }
}

// Verify that walkParallelPostOrder respects the post-order constraint:
// children are always visited before their parents.
TEST(InstanceGraphTest, ParallelPostOrderChildBeforeParent) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(fixtures::createModule(&context));

  // Record the visit index of each module.
  std::mutex mu;
  llvm::DenseMap<llvm::StringRef, unsigned> visitOrder;
  unsigned counter = 0;
  graph.walkParallelPostOrder([&](igraph::InstanceGraphNode &node) {
    std::lock_guard<std::mutex> lock(mu);
    visitOrder[node.getModule().getModuleName()] = counter++;
  });

  // The fixture graph:
  //   Top  -> Alligator, Cat
  //   Alligator -> Bear
  //   Bear -> Cat
  //   Cat  (leaf)
  //
  // We only assert ancestor/descendant pairs here.  Those relationships have
  // a guaranteed serial ordering regardless of how many threads run.  Sibling
  // pairs (e.g., Alligator vs Cat as direct children of Top) have no
  // guaranteed relative order and are intentionally not compared.
  EXPECT_LT(visitOrder["Cat"], visitOrder["Top"]);
  EXPECT_LT(visitOrder["Alligator"], visitOrder["Top"]);
  EXPECT_LT(visitOrder["Bear"], visitOrder["Alligator"]);
  EXPECT_LT(visitOrder["Cat"], visitOrder["Bear"]);
}

// Verify that a failing callback propagates the failure return value.
TEST(InstanceGraphTest, ParallelPostOrderPropagatesFailure) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(fixtures::createModule(&context));

  // Always return failure from the callback.
  LogicalResult result = graph.walkParallelPostOrder(
      [&](igraph::InstanceGraphNode &) -> LogicalResult { return failure(); });
  EXPECT_TRUE(failed(result));
}

// Verify that walkParallelPostOrder falls back correctly when multithreading is
// disabled on the context.
TEST(InstanceGraphTest, ParallelPostOrderFallbackToSerial) {
  MLIRContext context;
  context.disableMultithreading();
  InstanceGraph graph(fixtures::createModule(&context));

  llvm::SmallVector<std::string> visited;
  graph.walkParallelPostOrder([&](igraph::InstanceGraphNode &node) {
    visited.push_back(node.getModule().getModuleName().str());
  });

  ASSERT_EQ(4u, visited.size());
  for (const char *name : {"Top", "Alligator", "Bear", "Cat"}) {
    EXPECT_EQ(1, llvm::count(visited, name)) << name << " not visited once";
  }
}

// Verify that walkParallelInversePostOrder visits every node exactly once.
TEST(InstanceGraphTest, ParallelInversePostOrderVisitsAll) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(fixtures::createModule(&context));

  std::mutex mu;
  llvm::SmallVector<std::string> visited;
  graph.walkParallelInversePostOrder([&](igraph::InstanceGraphNode &node) {
    std::lock_guard<std::mutex> lock(mu);
    visited.push_back(node.getModule().getModuleName().str());
  });

  ASSERT_EQ(4u, visited.size());
  for (const char *name : {"Top", "Alligator", "Bear", "Cat"}) {
    EXPECT_EQ(1, llvm::count(visited, name)) << name << " not visited once";
  }
}

// Verify that walkParallelInversePostOrder respects the ordering constraint:
// parents are always visited before their children.
TEST(InstanceGraphTest, ParallelInversePostOrderParentBeforeChild) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(fixtures::createModule(&context));

  std::mutex mu;
  llvm::DenseMap<llvm::StringRef, unsigned> visitOrder;
  unsigned counter = 0;
  graph.walkParallelInversePostOrder([&](igraph::InstanceGraphNode &node) {
    std::lock_guard<std::mutex> lock(mu);
    visitOrder[node.getModule().getModuleName()] = counter++;
  });

  // The fixture graph:
  //   Top  -> Alligator, Cat
  //   Alligator -> Bear
  //   Bear -> Cat
  //   Cat  (leaf)
  //
  // We only assert ancestor/descendant pairs here.  Those relationships have
  // a guaranteed serial ordering regardless of how many threads run.  Sibling
  // pairs (e.g., Alligator vs Cat as direct children of Top) have no
  // guaranteed relative order and are intentionally not compared.
  EXPECT_LT(visitOrder["Top"], visitOrder["Cat"]);
  EXPECT_LT(visitOrder["Top"], visitOrder["Alligator"]);
  EXPECT_LT(visitOrder["Alligator"], visitOrder["Bear"]);
  EXPECT_LT(visitOrder["Bear"], visitOrder["Cat"]);
}

// Verify that a failing callback propagates the failure return value.
TEST(InstanceGraphTest, ParallelInversePostOrderPropagatesFailure) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(fixtures::createModule(&context));

  LogicalResult result = graph.walkParallelInversePostOrder(
      [&](igraph::InstanceGraphNode &) -> LogicalResult { return failure(); });
  EXPECT_TRUE(failed(result));
}

// Verify that walkParallelInversePostOrder falls back correctly when
// multithreading is disabled on the context.  The serial fallback must visit
// exactly the same set of managed nodes as the parallel path (i.e., only the
// nodes in the `nodes` list) and must not visit sentinel nodes such as the
// virtual top-level entry introduced by hw::InstanceGraph.
TEST(InstanceGraphTest, ParallelInversePostOrderFallbackToSerial) {
  MLIRContext context;
  context.disableMultithreading();
  InstanceGraph graph(fixtures::createModule(&context));

  llvm::SmallVector<std::string> visited;
  graph.walkParallelInversePostOrder([&](igraph::InstanceGraphNode &node) {
    visited.push_back(node.getModule().getModuleName().str());
  });

  ASSERT_EQ(4u, visited.size());
  for (const char *name : {"Top", "Alligator", "Bear", "Cat"}) {
    EXPECT_EQ(1, llvm::count(visited, name)) << name << " not visited once";
  }
}

// Multi-seed tests.
// A fixture with two independent chains exercises concurrent starts.  Both
// chain heads are seeds and can run simultaneously, which is the main scenario
// the parallel walk is designed for.
//
//   hw.module @A { hw.instance "b" @B }   chain 1: A instantiates B
//   hw.module @B {}
//   hw.module @C { hw.instance "d" @D }   chain 2: C instantiates D
//   hw.module @D {}
//
// In post-order the only constraints are B before A and D before C.
// In inverse post-order the constraints are A before B and C before D.

static mlir::ModuleOp createTwoChainModule(mlir::MLIRContext *context) {
  context->loadDialect<HWDialect>();
  mlir::LocationAttr loc = mlir::UnknownLoc::get(context);
  auto circuit = mlir::ModuleOp::create(loc);
  auto builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, circuit.getBody());

  auto modA = HWModuleOp::create(builder, mlir::StringAttr::get(context, "A"),
                                 llvm::ArrayRef<PortInfo>{});
  auto modB = HWModuleOp::create(builder, mlir::StringAttr::get(context, "B"),
                                 llvm::ArrayRef<PortInfo>{});
  auto modC = HWModuleOp::create(builder, mlir::StringAttr::get(context, "C"),
                                 llvm::ArrayRef<PortInfo>{});
  auto modD = HWModuleOp::create(builder, mlir::StringAttr::get(context, "D"),
                                 llvm::ArrayRef<PortInfo>{});

  modB.setVisibility(mlir::SymbolTable::Visibility::Private);
  modD.setVisibility(mlir::SymbolTable::Visibility::Private);

  builder.setInsertionPointToStart(modA.getBodyBlock());
  InstanceOp::create(builder, modB, "b", llvm::ArrayRef<mlir::Value>{});
  builder.setInsertionPointToStart(modC.getBodyBlock());
  InstanceOp::create(builder, modD, "d", llvm::ArrayRef<mlir::Value>{});

  return circuit;
}

TEST(InstanceGraphTest, ParallelPostOrderMultiSeedVisitsAll) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(createTwoChainModule(&context));

  std::mutex mu;
  llvm::SmallVector<std::string> visited;
  graph.walkParallelPostOrder([&](igraph::InstanceGraphNode &node) {
    std::lock_guard<std::mutex> lock(mu);
    visited.push_back(node.getModule().getModuleName().str());
  });

  ASSERT_EQ(4u, visited.size());
  for (const char *name : {"A", "B", "C", "D"}) {
    EXPECT_EQ(1, llvm::count(visited, name)) << name << " not visited once";
  }
}

TEST(InstanceGraphTest, ParallelPostOrderMultiSeedChildBeforeParent) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(createTwoChainModule(&context));

  std::mutex mu;
  llvm::DenseMap<llvm::StringRef, unsigned> order;
  unsigned counter = 0;
  graph.walkParallelPostOrder([&](igraph::InstanceGraphNode &node) {
    std::lock_guard<std::mutex> lock(mu);
    order[node.getModule().getModuleName()] = counter++;
  });

  EXPECT_LT(order["B"], order["A"]);
  EXPECT_LT(order["D"], order["C"]);
}

TEST(InstanceGraphTest, ParallelInversePostOrderMultiSeedParentBeforeChild) {
  MLIRContext context;
  context.enableMultithreading();
  InstanceGraph graph(createTwoChainModule(&context));

  std::mutex mu;
  llvm::DenseMap<llvm::StringRef, unsigned> order;
  unsigned counter = 0;
  graph.walkParallelInversePostOrder([&](igraph::InstanceGraphNode &node) {
    std::lock_guard<std::mutex> lock(mu);
    order[node.getModule().getModuleName()] = counter++;
  });

  EXPECT_LT(order["A"], order["B"]);
  EXPECT_LT(order["C"], order["D"]);
}

// Wide DAG stress test.
// Build a two-level DAG: one root module that instantiates N leaves.  All N
// leaves are seeds and can execute truly concurrently.  Run repeatedly to
// stress the thread safety of the walk bookkeeping itself.

static mlir::ModuleOp createWideDAGModule(mlir::MLIRContext *context,
                                          unsigned numLeaves) {
  context->loadDialect<HWDialect>();
  mlir::LocationAttr loc = mlir::UnknownLoc::get(context);
  auto circuit = mlir::ModuleOp::create(loc);
  auto builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, circuit.getBody());

  // Create the root module first so it appears first in the module list.
  auto root =
      HWModuleOp::create(builder, mlir::StringAttr::get(context, "Root"),
                         llvm::ArrayRef<PortInfo>{});

  llvm::SmallVector<HWModuleOp> leaves;
  for (unsigned i = 0; i < numLeaves; ++i) {
    auto name = mlir::StringAttr::get(context, "Leaf" + std::to_string(i));
    auto leaf = HWModuleOp::create(builder, name, llvm::ArrayRef<PortInfo>{});
    leaf.setVisibility(mlir::SymbolTable::Visibility::Private);
    leaves.push_back(leaf);
  }

  builder.setInsertionPointToStart(root.getBodyBlock());
  for (unsigned i = 0; i < numLeaves; ++i) {
    auto instName = "leaf" + std::to_string(i);
    InstanceOp::create(builder, leaves[i], instName,
                       llvm::ArrayRef<mlir::Value>{});
  }

  return circuit;
}

TEST(InstanceGraphTest, ParallelPostOrderWideDAGStress) {
  constexpr unsigned kNumLeaves = 100;
  constexpr unsigned kNumRounds = 20;

  MLIRContext context;
  context.enableMultithreading();

  for (unsigned round = 0; round < kNumRounds; ++round) {
    InstanceGraph graph(createWideDAGModule(&context, kNumLeaves));

    std::atomic<unsigned> leafCount(0);
    std::atomic<unsigned> rootCount(0);
    graph.walkParallelPostOrder([&](igraph::InstanceGraphNode &node) {
      if (node.getModule().getModuleName() == "Root")
        rootCount.fetch_add(1, std::memory_order_relaxed);
      else
        leafCount.fetch_add(1, std::memory_order_relaxed);
    });

    ASSERT_EQ(kNumLeaves, leafCount.load());
    ASSERT_EQ(1u, rootCount.load());
  }
}

} // namespace
