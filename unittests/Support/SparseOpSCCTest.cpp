//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SparseOpSCC.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

// Graph region with a simple chain:  and -> or -> output
// (each op's result is consumed by the next)
const char *ir = R"MLIR(
  hw.module private @test(in %a: i1, in %b: i1, out x: i1) {
    %and = comb.and %a, %b : i1
    %or  = comb.or  %and, %a : i1
    hw.output %or : i1
  }
)MLIR";

TEST(SparseOpSCCsTest, SimpleChain) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *andOp = &*it++;
  Operation *orOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  // Forward reachability from andOp.
  // Reverse topological order: leaves-first -> [outputOp, orOp, andOp].
  {
    SparseOpSCC<OpSCCDirection::Reachable> opScc;
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumVisited(), 3u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto topoIt = opScc.topological_begin();
    EXPECT_EQ(andOp, cast<Operation *>(*(topoIt++)));
    EXPECT_EQ(orOp, cast<Operation *>(*(topoIt++)));
    EXPECT_EQ(outputOp, cast<Operation *>(*(topoIt++)));
    EXPECT_EQ(topoIt, opScc.topological_end());

    auto revTopoIt = opScc.reverseTopological_begin();
    EXPECT_EQ(outputOp, cast<Operation *>(*(revTopoIt++)));
    EXPECT_EQ(orOp, cast<Operation *>(*(revTopoIt++)));
    EXPECT_EQ(andOp, cast<Operation *>(*(revTopoIt++)));
    EXPECT_EQ(revTopoIt, opScc.reverseTopological_end());
  }

  // Inverse reachability from outputOp.
  // Follows operands backward; reverse topo of inverse = forward topo:
  // [andOp, orOp, outputOp].
  {
    SparseOpSCC<OpSCCDirection::Reaching> opScc;
    opScc.visit(outputOp);

    EXPECT_EQ(opScc.getNumVisited(), 3u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto topoIt = opScc.topological_begin();
    EXPECT_EQ(andOp, cast<Operation *>(*(topoIt++)));
    EXPECT_EQ(orOp, cast<Operation *>(*(topoIt++)));
    EXPECT_EQ(outputOp, cast<Operation *>(*(topoIt++)));
    EXPECT_EQ(topoIt, opScc.topological_end());

    auto revTopoIt = opScc.reverseTopological_begin();
    EXPECT_EQ(outputOp, cast<Operation *>(*(revTopoIt++)));
    EXPECT_EQ(orOp, cast<Operation *>(*(revTopoIt++)));
    EXPECT_EQ(andOp, cast<Operation *>(*(revTopoIt++)));
    EXPECT_EQ(revTopoIt, opScc.reverseTopological_end());
  }
}

// Passing an empty seed list leaves all counts at zero.
TEST(SparseOpSCCsTest, EmptySeeds) {
  SparseOpSCC<OpSCCDirection::Reachable> fwd;
  fwd.visit(ArrayRef<Operation *>{});
  EXPECT_EQ(fwd.getNumVisited(), 0u);
  EXPECT_EQ(fwd.getNumSCCs(), 0u);
  EXPECT_EQ(fwd.getNumCyclicSCCs(), 0u);

  SparseOpSCC<OpSCCDirection::Reaching> inv;
  inv.visit(ArrayRef<Operation *>{});
  EXPECT_EQ(inv.getNumVisited(), 0u);
  EXPECT_EQ(inv.getNumSCCs(), 0u);
  EXPECT_EQ(inv.getNumCyclicSCCs(), 0u);
}

// A seed that is a boundary node in the traversal direction is returned as a
// single trivial SCC with no further expansion.
//   - Forward from outputOp: hw.output produces no results, so it has no
//     forward edges; only outputOp is emitted.
//   - Inverse from andOp: both operands are block arguments (no defining op),
//     so there are no inverse edges; only andOp is emitted.
TEST(SparseOpSCCsTest, BoundarySeed) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *andOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  {
    SparseOpSCC<OpSCCDirection::Reachable> opScc;
    opScc.visit(outputOp);

    EXPECT_EQ(opScc.getNumVisited(), 1u);
    EXPECT_EQ(opScc.getNumSCCs(), 1u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto topoIt = opScc.topological_begin();
    EXPECT_EQ(cast<Operation *>(*(topoIt++)), outputOp);
    EXPECT_EQ(topoIt, opScc.topological_end());
  }

  {
    SparseOpSCC<OpSCCDirection::Reaching> opScc;
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumVisited(), 1u);
    EXPECT_EQ(opScc.getNumSCCs(), 1u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto topoIt = opScc.topological_begin();
    EXPECT_EQ(cast<Operation *>(*(topoIt++)), andOp);
    EXPECT_EQ(topoIt, opScc.topological_end());
  }
}

// When one seed is already reachable from another, it is visited during the
// first seed's DFS and silently skipped when the loop encounters it as a seed.
// The result is identical to seeding with just the upstream op.
TEST(SparseOpSCCsTest, OverlappingSeeds) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *andOp = &*it++;
  Operation *orOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  SparseOpSCC<OpSCCDirection::Reachable> opScc;
  opScc.visit(andOp);
  opScc.visit(orOp); // already visited via andOp — skipped

  EXPECT_EQ(opScc.getNumSCCs(), 3u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

  auto topoIt = opScc.topological_begin();
  EXPECT_EQ(cast<Operation *>(*(topoIt++)), andOp);
  EXPECT_EQ(cast<Operation *>(*(topoIt++)), orOp);
  EXPECT_EQ(cast<Operation *>(*(topoIt++)), outputOp);
  EXPECT_EQ(topoIt, opScc.topological_end());
}

// A filter that excludes the seed op prevents it from being visited at all.
TEST(SparseOpSCCsTest, FilterExcludesSeed) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  Operation *andOp = &*hwModule.getBodyBlock()->begin();

  auto filter = [&](Operation *op) { return op != andOp; };
  SparseOpSCC<OpSCCDirection::Reachable> opScc(filter);
  opScc.visit(andOp);

  EXPECT_EQ(opScc.getNumVisited(), 0u);
  EXPECT_EQ(opScc.getNumSCCs(), 0u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);
}

// Diamond: %and splits into two branches that join at %merge; no cycle.
//
//   %and   = comb.and %a, %b : i1     split node
//   %or    = comb.or  %and, %a : i1   left branch
//   %xor   = comb.xor %and, %b : i1   right branch
//   %merge = comb.or  %or, %xor : i1  join node
//   hw.output %merge : i1
//
// Forward edges: and->{or,xor}, or->merge, xor->merge, merge->output.
// Reverse-topo order: outputOp, mergeOp, {orOp,xorOp} (order unspecified),
// andOp.
const char *diamondIr = R"MLIR(
  hw.module private @diamond(in %a: i1, in %b: i1, out x: i1) {
    %and   = comb.and %a, %b : i1
    %or    = comb.or  %and, %a : i1
    %xor   = comb.xor %and, %b : i1
    %merge = comb.or  %or, %xor : i1
    hw.output %merge : i1
  }
)MLIR";

TEST(SparseOpSCCsTest, DiamondNoCycle) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(diamondIr, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("diamond");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *andOp = &*it++;
  Operation *orOp = &*it++;
  Operation *xorOp = &*it++;
  Operation *mergeOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  SparseOpSCC<OpSCCDirection::Reachable> opScc;
  opScc.visit(andOp);

  EXPECT_EQ(opScc.getNumSCCs(), 5u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

  // outputOp first, mergeOp second, andOp last; orOp/xorOp order is
  // DFS-dependent.
  auto revIt = opScc.reverseTopological_begin();
  EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
  EXPECT_EQ(cast<Operation *>(*(revIt++)), mergeOp);
  std::array<Operation *, 2> middle = {cast<Operation *>(*(revIt++)),
                                       cast<Operation *>(*(revIt++))};
  EXPECT_TRUE(llvm::is_contained(middle, orOp));
  EXPECT_TRUE(llvm::is_contained(middle, xorOp));
  EXPECT_EQ(cast<Operation *>(*(revIt++)), andOp);
  EXPECT_EQ(revIt, opScc.reverseTopological_end());
}

// Graph with a cycle: and uses reg's result, reg uses and's result.
//
//   %reg = seq.firreg %and clock %clock : i1
//   %and = comb.and %reg, %a : i1
//   hw.output %and : i1
//
// Block order: regOp, andOp, outputOp (terminator).
// Forward edges: reg->and, and->{reg, output}.
const char *cycleIr = R"MLIR(
  hw.module private @cycle(in %clock: !seq.clock, in %a: i1, out x: i1) {
    %reg = seq.firreg %and clock %clock : i1
    %and = comb.and %reg, %a : i1
    hw.output %and : i1
  }
)MLIR";

TEST(SparseOpSCCsTest, CycleWithRegister) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<seq::SeqDialect>();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(cycleIr, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("cycle");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *regOp = &*it++;
  Operation *andOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  // Without filter: reg and and form a cycle -> one CyclicOpSCC.
  {
    SparseOpSCC<OpSCCDirection::Reachable> opScc;
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);

    CyclicOpSCC scc = cast<CyclicOpSCC>(*revIt++);
    EXPECT_EQ(scc.size(), 2u);
    EXPECT_TRUE(llvm::is_contained(scc, regOp));
    EXPECT_TRUE(llvm::is_contained(scc, andOp));
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }

  // With filter that excludes regOp: no cycle, both ops are trivial.
  {
    auto filter = [&](Operation *op) { return op != regOp; };
    SparseOpSCC<OpSCCDirection::Reachable> opScc(filter);
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto topoIt = opScc.topological_begin();
    EXPECT_EQ(cast<Operation *>(*(topoIt++)), andOp);
    EXPECT_EQ(cast<Operation *>(*(topoIt++)), outputOp);
    EXPECT_EQ(topoIt, opScc.topological_end());
  }
}

// Two disjoint register cycles sharing a common downstream merge node.
//
//   %reg1 = seq.firreg %and1 clock %clock : i1   -. cycle 1
//   %and1 = comb.and %reg1, %a : i1              -'
//   %reg2 = seq.firreg %or1  clock %clock : i1   -. cycle 2
//   %or1  = comb.or  %reg2, %a : i1              -'
//   %xor  = comb.xor %and1, %or1 : i1              merge node
//   hw.output %xor : i1
//
// Seeds: {and1Op, or1Op}.
// Expected reverse-topo: [outputOp, xorOp, SCC{reg1,and1}, SCC{reg2,or1}]
const char *twoCyclesIr = R"MLIR(
  hw.module private @twocycles(in %clock: !seq.clock, in %a: i1, out x: i1) {
    %reg1 = seq.firreg %and1 clock %clock : i1
    %and1 = comb.and %reg1, %a : i1
    %reg2 = seq.firreg %or1  clock %clock : i1
    %or1  = comb.or  %reg2, %a : i1
    %xor  = comb.xor %and1, %or1 : i1
    hw.output %xor : i1
  }
)MLIR";

TEST(SparseOpSCCsTest, TwoCycles) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<seq::SeqDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(twoCyclesIr, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("twocycles");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *reg1Op = &*it++;
  Operation *and1Op = &*it++;
  Operation *reg2Op = &*it++;
  Operation *or1Op = &*it++;
  Operation *xorOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  SparseOpSCC<OpSCCDirection::Reachable> opScc;
  opScc.visit(and1Op);
  opScc.visit(or1Op);

  EXPECT_EQ(opScc.getNumSCCs(), 4u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 2u);

  auto revIt = opScc.reverseTopological_begin();
  EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
  EXPECT_EQ(cast<Operation *>(*(revIt++)), xorOp);

  // First cyclic SCC: reg1 <-> and1.
  CyclicOpSCC scc0 = cast<CyclicOpSCC>(*revIt++);
  EXPECT_EQ(scc0.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(scc0, reg1Op));
  EXPECT_TRUE(llvm::is_contained(scc0, and1Op));

  // Second cyclic SCC: reg2 <-> or1.
  CyclicOpSCC scc1 = cast<CyclicOpSCC>(*revIt++);
  EXPECT_EQ(scc1.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(scc1, reg2Op));
  EXPECT_TRUE(llvm::is_contained(scc1, or1Op));

  EXPECT_EQ(revIt, opScc.reverseTopological_end());
}

// One large SCC containing two internal cycles and five operations.
//
//   %r0   = seq.firreg %and1 ...  -. cycle A: r0 <-> and1
//   %and1 = comb.and %r0, %r2     -'   (and1 also drives xor4 and output)
//   %r2   = seq.firreg %or3  ...  -. cycle B: r2 <-> or3
//   %or3  = comb.or  %r2, %xor4   -'
//   %xor4 = comb.xor %and1, %a      bridge: and1 -> xor4 -> or3 -> r2 -> and1
//
// All five are mutually reachable -> single CyclicOpSCC of size 5.
const char *twoInternalCyclesIr = R"MLIR(
  hw.module private @largeSCC(in %clock: !seq.clock, in %a: i1, out x: i1) {
    %r0   = seq.firreg %and1 clock %clock : i1
    %and1 = comb.and %r0, %r2 : i1
    %r2   = seq.firreg %or3  clock %clock : i1
    %or3  = comb.or  %r2, %xor4 : i1
    %xor4 = comb.xor %and1, %a : i1
    hw.output %and1 : i1
  }
)MLIR";

TEST(SparseOpSCCsTest, TwoInternalCycles) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<seq::SeqDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(twoInternalCyclesIr, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("largeSCC");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *r0Op = &*it++;
  Operation *and1Op = &*it++;
  Operation *r2Op = &*it++;
  Operation *or3Op = &*it++;
  Operation *xor4Op = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  auto checkFiveOpSCC = [&](CyclicOpSCC scc) {
    EXPECT_EQ(scc.size(), 5u);
    EXPECT_TRUE(llvm::is_contained(scc, r0Op));
    EXPECT_TRUE(llvm::is_contained(scc, and1Op));
    EXPECT_TRUE(llvm::is_contained(scc, r2Op));
    EXPECT_TRUE(llvm::is_contained(scc, or3Op));
    EXPECT_TRUE(llvm::is_contained(scc, xor4Op));
  };

  // Forward direction from and1Op: outputOp is the only leaf; the five-op SCC
  // follows.
  {
    SparseOpSCC<OpSCCDirection::Reachable> opScc;
    opScc.visit(and1Op);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    checkFiveOpSCC(cast<CyclicOpSCC>(*revIt++));
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }

  // Inverse direction from outputOp: the five-op SCC precedes outputOp
  // (topological order: predecessors before successors).
  {
    SparseOpSCC<OpSCCDirection::Reaching> opScc;
    opScc.visit(outputOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto topoIt = opScc.topological_begin();
    checkFiveOpSCC(cast<CyclicOpSCC>(*topoIt++));
    EXPECT_EQ(cast<Operation *>(*(topoIt++)), outputOp);
    EXPECT_EQ(topoIt, opScc.topological_end());
  }
}

// Self-loop: comb.and that uses its own result as one operand.
const char *selfLoopIr = R"MLIR(
  hw.module private @selfloop(in %a: i1, out x: i1) {
    %and = comb.and %and, %a : i1
    hw.output %and : i1
  }
)MLIR";

TEST(SparseOpSCCsTest, SelfLoop) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(selfLoopIr, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("selfloop");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *andOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  SparseOpSCC<OpSCCDirection::Reachable> opScc;
  opScc.visit(andOp);

  // outputOp is a trivial leaf; andOp is a size-1 CyclicOpSCC due to self-loop.
  EXPECT_EQ(opScc.getNumSCCs(), 2u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

  auto revIt = opScc.reverseTopological_begin();
  EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
  // Second entry is a CyclicOpSCC, not a plain Operation *.
  EXPECT_EQ(dyn_cast<Operation *>(*revIt), nullptr);
  CyclicOpSCC scc = cast<CyclicOpSCC>(*revIt++);
  EXPECT_EQ(scc.size(), 1u);
  EXPECT_EQ(scc[0], andOp);
  EXPECT_EQ(revIt, opScc.reverseTopological_end());
}

} // namespace
