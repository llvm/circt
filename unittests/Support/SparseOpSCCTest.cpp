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
#include "circt/Dialect/Seq/SeqOps.h"
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

// Graph with a register-and cycle: reg uses and's result as next/resetValue,
// and uses reg's result as an operand.
//   Forward edges: reg->and, and->{reg(next), reg(resetValue), output}
const char *cycleIr = R"MLIR(
  hw.module private @cycle(in %clock: !seq.clock, in %reset: i1, in %a: i1, out x: i1) {
    %reg = seq.firreg %and clock %clock reset sync %reset, %and : i1
    %and = comb.and %reg, %a : i1
    hw.output %and : i1
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
  {
    SparseOpSCC<OpSCCDirection::Forward> opScc;
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumDiscovered(), 3u);
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

    // getSCC maps each visited op to its own trivial SCC entry.
    EXPECT_EQ(cast<Operation *>(opScc.getSCC(andOp)), andOp);
    EXPECT_EQ(cast<Operation *>(opScc.getSCC(orOp)), orOp);
    EXPECT_EQ(cast<Operation *>(opScc.getSCC(outputOp)), outputOp);
  }

  // Inverse reachability from outputOp.
  // Follows operands backward; reverse topo of inverse = forward topo:
  // [andOp, orOp, outputOp].
  {
    SparseOpSCC<OpSCCDirection::Backward> opScc;
    opScc.visit(outputOp);

    EXPECT_EQ(opScc.getNumDiscovered(), 3u);
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

// reset() clears all accumulated state including cyclic SCC storage;
// re-visiting produces fresh results.
TEST(SparseOpSCCsTest, Reset) {
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

  SparseOpSCC<OpSCCDirection::Forward> opScc;
  opScc.visit(andOp); // discovers CyclicOpSCC{reg,and} + trivial outputOp
  EXPECT_EQ(opScc.getNumDiscovered(), 3u);
  EXPECT_EQ(opScc.getNumSCCs(), 2u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

  opScc.reset();

  // All state cleared, including cyclic SCC storage.
  EXPECT_EQ(opScc.getNumDiscovered(), 0u);
  EXPECT_EQ(opScc.getNumSCCs(), 0u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);
  EXPECT_FALSE(opScc.hasDiscovered(regOp));
  EXPECT_FALSE(opScc.hasDiscovered(andOp));
  EXPECT_FALSE(opScc.hasDiscovered(outputOp));
  EXPECT_EQ(opScc.topological_begin(), opScc.topological_end());
  EXPECT_EQ(opScc.reverseTopological_begin(), opScc.reverseTopological_end());

  // Re-visiting with a boundary seed produces the correct fresh result.
  opScc.visit(outputOp); // hw.output has no results: only outputOp discovered.
  EXPECT_EQ(opScc.getNumDiscovered(), 1u);
  EXPECT_EQ(opScc.getNumSCCs(), 1u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);
  EXPECT_TRUE(opScc.hasDiscovered(outputOp));
  EXPECT_FALSE(opScc.hasDiscovered(andOp));
}

// Passing an empty seed list leaves all counts at zero.
TEST(SparseOpSCCsTest, EmptySeeds) {
  SparseOpSCC<OpSCCDirection::Forward> fwd;
  fwd.visit(ArrayRef<Operation *>{});
  EXPECT_EQ(fwd.getNumDiscovered(), 0u);
  EXPECT_EQ(fwd.getNumSCCs(), 0u);
  EXPECT_EQ(fwd.getNumCyclicSCCs(), 0u);

  SparseOpSCC<OpSCCDirection::Backward> inv;
  inv.visit(ArrayRef<Operation *>{});
  EXPECT_EQ(inv.getNumDiscovered(), 0u);
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
    SparseOpSCC<OpSCCDirection::Forward> opScc;
    opScc.visit(outputOp);

    EXPECT_EQ(opScc.getNumDiscovered(), 1u);
    EXPECT_EQ(opScc.getNumSCCs(), 1u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto topoIt = opScc.topological_begin();
    EXPECT_EQ(cast<Operation *>(*(topoIt++)), outputOp);
    EXPECT_EQ(topoIt, opScc.topological_end());

    // getSCC returns a valid entry for the visited op and null for
    // unvisited.
    EXPECT_EQ(cast<Operation *>(opScc.getSCC(outputOp)), outputOp);
    EXPECT_FALSE(static_cast<bool>(opScc.getSCC(andOp)));
  }

  {
    SparseOpSCC<OpSCCDirection::Backward> opScc;
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumDiscovered(), 1u);
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

  SparseOpSCC<OpSCCDirection::Forward> opScc;
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

// Incremental visit: two independent roots feeding a shared merge node.
//
//   %av    = comb.and %a, %a : i1   seed 1 (independent)
//   %bv    = comb.or  %b, %b : i1   seed 2 (independent)
//   %merge = comb.xor %av, %bv : i1 shared successor
//   hw.output %merge : i1
//
// visit(avOp) discovers {avOp, mergeOp, outputOp}.
// visit(bvOp) discovers only bvOp: mergeOp is already discovered and is
// treated as a cross-edge, not re-entered.
const char *incrementalIr = R"MLIR(
  hw.module private @incremental(in %a: i1, in %b: i1, out x: i1) {
    %av    = comb.and %a, %a : i1
    %bv    = comb.or  %b, %b : i1
    %merge = comb.xor %av, %bv : i1
    hw.output %merge : i1
  }
)MLIR";

TEST(SparseOpSCCsTest, IncrementalVisit) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(incrementalIr, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("incremental");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *avOp = &*it++;
  Operation *bvOp = &*it++;
  Operation *mergeOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  SparseOpSCC<OpSCCDirection::Forward> opScc;

  // First visit: avOp → mergeOp → outputOp.
  opScc.visit(avOp);
  EXPECT_EQ(opScc.getNumDiscovered(), 3u);
  EXPECT_TRUE(opScc.hasDiscovered(avOp));
  EXPECT_TRUE(opScc.hasDiscovered(mergeOp));
  EXPECT_TRUE(opScc.hasDiscovered(outputOp));
  EXPECT_FALSE(opScc.hasDiscovered(bvOp));

  // Second visit: bvOp is new; mergeOp/outputOp are cross-edges — not
  // re-entered, so only bvOp is added.
  opScc.visit(bvOp);
  EXPECT_EQ(opScc.getNumDiscovered(), 4u);
  EXPECT_TRUE(opScc.hasDiscovered(bvOp));
  EXPECT_EQ(opScc.getNumSCCs(), 4u);
  EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

  // mergeOp's SCC was assigned by the first visit and is unchanged.
  EXPECT_EQ(cast<Operation *>(opScc.getSCC(mergeOp)), mergeOp);

  // Reverse-topological (leaves first): outputOp, mergeOp, avOp, bvOp.
  // bvOp trails because it was emitted by the second DFS.
  auto revIt = opScc.reverseTopological_begin();
  EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
  EXPECT_EQ(cast<Operation *>(*(revIt++)), mergeOp);
  EXPECT_EQ(cast<Operation *>(*(revIt++)), avOp);
  EXPECT_EQ(cast<Operation *>(*(revIt++)), bvOp);
  EXPECT_EQ(revIt, opScc.reverseTopological_end());
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

  // Without filter: all five ops, orOp/xorOp order is DFS-dependent.
  {
    SparseOpSCC<OpSCCDirection::Forward> opScc;
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumSCCs(), 5u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

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

  // Forward direction with edge filter blocking xorOp as destination: only
  // the left branch (andOp->orOp->mergeOp->outputOp) is visited. xorOp is
  // never reachable, so 4 trivial SCCs in deterministic reverseTopological
  // order: outputOp, mergeOp, orOp, andOp.
  {
    auto filter = [&](Operation *op, OpOperand &) { return op != xorOp; };
    SparseOpSCC<OpSCCDirection::Forward> opScc(filter);
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumSCCs(), 4u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    EXPECT_EQ(cast<Operation *>(*(revIt++)), mergeOp);
    EXPECT_EQ(cast<Operation *>(*(revIt++)), orOp);
    EXPECT_EQ(cast<Operation *>(*(revIt++)), andOp);
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }

  // Backward direction from outputOp with edge filter blocking xorOp as
  // source: the backward path through xorOp is cut, so only the left branch
  // (orOp) is visited (outputOp<-mergeOp<-orOp<-andOp). 4 trivial SCCs in
  // reverseTopological order (sinks-first): outputOp, mergeOp, orOp, andOp.
  {
    auto filter = [&](Operation *src, OpOperand &) { return src != xorOp; };
    SparseOpSCC<OpSCCDirection::Backward> opScc(filter);
    opScc.visit(outputOp);

    EXPECT_EQ(opScc.getNumSCCs(), 4u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    EXPECT_EQ(cast<Operation *>(*(revIt++)), mergeOp);
    EXPECT_EQ(cast<Operation *>(*(revIt++)), orOp);
    EXPECT_EQ(cast<Operation *>(*(revIt++)), andOp);
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }
}

// Graph with a cycle: "and" uses "reg"'s result, "reg" uses "and"'s result.
//
//   %reg = seq.firreg %and clock %clock reset sync %reset, %and : i1
//   %and = comb.and %reg, %a : i1
//   hw.output %and : i1
//
// %and feeds reg.next (the cycle edge) and reg.resetValue.
// Block order: regOp, andOp, outputOp (terminator).
// Forward edges: reg->and, and->{reg(next), reg(resetValue), output}.

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

  // Without filter: "reg" and "and" form a cycle -> one CyclicOpSCC.
  {
    SparseOpSCC<OpSCCDirection::Forward> opScc;
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

    // getSCC: both cycle members map to the same CyclicOpSCC entry.
    EXPECT_EQ(cast<CyclicOpSCC>(opScc.getSCC(regOp)), scc);
    EXPECT_EQ(cast<CyclicOpSCC>(opScc.getSCC(andOp)), scc);
    EXPECT_EQ(cast<Operation *>(opScc.getSCC(outputOp)), outputOp);
  }

  // With filter that excludes regOp: no cycle, both ops are trivial.
  {
    auto filter = [&](Operation *op, OpOperand &) { return op != regOp; };
    SparseOpSCC<OpSCCDirection::Forward> opScc(filter);
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto topoIt = opScc.topological_begin();
    EXPECT_EQ(cast<Operation *>(*(topoIt++)), andOp);
    EXPECT_EQ(cast<Operation *>(*(topoIt++)), outputOp);
    EXPECT_EQ(topoIt, opScc.topological_end());
  }

  // With edge filter blocking only reg's 'next' operand: %and also drives
  // reg.resetValue, so the cycle persists through the reset edge.
  {
    auto regEdgeFilter = [](Operation *, OpOperand &operand) -> bool {
      if (auto firReg = dyn_cast<seq::FirRegOp>(operand.getOwner()))
        return operand != firReg.getNextMutable();
      return true;
    };
    SparseOpSCC<OpSCCDirection::Forward> opScc(regEdgeFilter);
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);

    CyclicOpSCC scc = cast<CyclicOpSCC>(*revIt++);
    ASSERT_EQ(scc.size(), 2u);
    EXPECT_TRUE(llvm::is_contained(scc, regOp));
    EXPECT_TRUE(llvm::is_contained(scc, andOp));
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }

  // Backward direction from outputOp with regEdgeFilter: the backward path
  // through reg.resetValue keeps the cycle alive even though reg.next is
  // blocked. reverseTopological order (sinks-first): outputOp, then
  // CyclicOpSCC{regOp,andOp}.
  {
    auto regEdgeFilter = [](Operation *, OpOperand &operand) -> bool {
      if (auto firReg = dyn_cast<seq::FirRegOp>(operand.getOwner()))
        return operand != firReg.getNextMutable();
      return true;
    };
    SparseOpSCC<OpSCCDirection::Backward> opScc(regEdgeFilter);
    opScc.visit(outputOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    CyclicOpSCC scc = cast<CyclicOpSCC>(*revIt++);
    ASSERT_EQ(scc.size(), 2u);
    EXPECT_TRUE(llvm::is_contained(scc, regOp));
    EXPECT_TRUE(llvm::is_contained(scc, andOp));
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
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

  SparseOpSCC<OpSCCDirection::Forward> opScc;
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

  EXPECT_EQ(opScc.getSCC(reg1Op), opScc.getSCC(and1Op));
  EXPECT_NE(opScc.getSCC(reg1Op), opScc.getSCC(or1Op));
  EXPECT_NE(opScc.getSCC(reg1Op), opScc.getSCC(xorOp));
  EXPECT_NE(opScc.getSCC(reg1Op), opScc.getSCC(hwModule));

  // Second cyclic SCC: reg2 <-> or1.
  CyclicOpSCC scc1 = cast<CyclicOpSCC>(*revIt++);
  EXPECT_EQ(scc1.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(scc1, reg2Op));
  EXPECT_TRUE(llvm::is_contained(scc1, or1Op));

  EXPECT_EQ(opScc.getSCC(reg2Op), opScc.getSCC(or1Op));
  EXPECT_NE(opScc.getSCC(or1Op), opScc.getSCC(and1Op));
  EXPECT_NE(opScc.getSCC(or1Op), opScc.getSCC(xorOp));
  EXPECT_NE(opScc.getSCC(or1Op), opScc.getSCC(hwModule));

  EXPECT_EQ(revIt, opScc.reverseTopological_end());

  // Backward from xorOp: discovers both cycles plus xorOp itself.
  // outputOp is downstream of xorOp in the forward graph and is not reached.
  // reverseTopological: xorOp first (forward-graph leaf), then the two cyclic
  // SCCs in unspecified order.
  {
    SparseOpSCC<OpSCCDirection::Backward> opScc;
    opScc.visit(xorOp);

    EXPECT_EQ(opScc.getNumDiscovered(), 5u); // reg1, and1, reg2, or1, xor
    EXPECT_EQ(opScc.getNumSCCs(), 3u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 2u);
    EXPECT_FALSE(opScc.hasDiscovered(outputOp));

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), xorOp);

    std::array<CyclicOpSCC, 2> cycles = {cast<CyclicOpSCC>(*revIt++),
                                         cast<CyclicOpSCC>(*revIt++)};
    EXPECT_TRUE(llvm::any_of(cycles, [&](CyclicOpSCC scc) {
      return scc.size() == 2 && llvm::is_contained(scc, reg1Op) &&
             llvm::is_contained(scc, and1Op);
    }));
    EXPECT_TRUE(llvm::any_of(cycles, [&](CyclicOpSCC scc) {
      return scc.size() == 2 && llvm::is_contained(scc, reg2Op) &&
             llvm::is_contained(scc, or1Op);
    }));

    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }
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
    SparseOpSCC<OpSCCDirection::Forward> opScc;
    opScc.visit(and1Op);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    auto cyclicSCC = cast<CyclicOpSCC>(*revIt++);
    checkFiveOpSCC(cyclicSCC);
    EXPECT_TRUE(llvm::all_equal(llvm::map_range(
        cyclicSCC, [&](Operation *op) -> OpSCC { return opScc.getSCC(op); })));
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }

  // Inverse direction from outputOp: the five-op SCC precedes outputOp
  // (topological order: predecessors before successors).
  {
    SparseOpSCC<OpSCCDirection::Backward> opScc;
    opScc.visit(outputOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto topoIt = opScc.topological_begin();
    auto cyclicSCC = cast<CyclicOpSCC>(*topoIt++);
    checkFiveOpSCC(cyclicSCC);
    EXPECT_TRUE(llvm::all_equal(llvm::map_range(
        cyclicSCC, [&](Operation *op) -> OpSCC { return opScc.getSCC(op); })));
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

  // Without filter: andOp is a size-1 CyclicOpSCC due to self-loop.
  {
    SparseOpSCC<OpSCCDirection::Forward> opScc;
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    EXPECT_EQ(dyn_cast<Operation *>(*revIt), nullptr);
    CyclicOpSCC scc = cast<CyclicOpSCC>(*revIt++);
    EXPECT_EQ(scc.size(), 1u);
    EXPECT_EQ(scc[0], andOp);
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }

  // With edge filter blocking the self-loop edge: andOp becomes trivial.
  {
    auto filter = [&](Operation *dest, OpOperand &) { return dest != andOp; };
    SparseOpSCC<OpSCCDirection::Forward> opScc(filter);
    opScc.visit(andOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    EXPECT_EQ(cast<Operation *>(*(revIt++)), andOp);
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }
}

// Self-loop via register: both 'next' and 'resetValue' feed back into the
// register itself, creating two independent self-loop edges.
//
//   %reg = seq.firreg %reg clock %clock reset sync %reset, %reg : i1
//
// Operand 0 (next) and operand 3 (resetValue) are both self-loop edges.
const char *selfLoopRegIr = R"MLIR(
  hw.module private @selfloopReg(in %clock: !seq.clock, in %reset: i1, out x: i1) {
    %reg = seq.firreg %reg clock %clock reset sync %reset, %reg : i1
    hw.output %reg : i1
  }
)MLIR";

TEST(SparseOpSCCsTest, SelfLoopRegWithEdgeFilter) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<seq::SeqDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(selfLoopRegIr, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("selfloopReg");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *regOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  // Without filter: both self-loop edges are visible -> CyclicOpSCC of size 1.
  {
    SparseOpSCC<OpSCCDirection::Forward> opScc;
    opScc.visit(regOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    CyclicOpSCC scc = cast<CyclicOpSCC>(*revIt++);
    ASSERT_EQ(scc.size(), 1u);
    EXPECT_EQ(scc[0], regOp);
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }

  // Filter blocking only 'next': resetValue self-loop edge is still
  // traversable -> still classified as CyclicOpSCC.
  {
    auto regEdgeFilter = [](Operation *, OpOperand &operand) -> bool {
      if (auto firReg = dyn_cast<seq::FirRegOp>(operand.getOwner()))
        return operand != firReg.getNextMutable();
      return true;
    };
    SparseOpSCC<OpSCCDirection::Forward> opScc(regEdgeFilter);
    opScc.visit(regOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 1u);
  }

  // Filter blocking all edges into regOp: both self-loop edges blocked ->
  // regOp becomes trivial.
  {
    auto filter = [&](Operation *dest, OpOperand &) { return dest != regOp; };
    SparseOpSCC<OpSCCDirection::Forward> opScc(filter);
    opScc.visit(regOp);

    EXPECT_EQ(opScc.getNumSCCs(), 2u);
    EXPECT_EQ(opScc.getNumCyclicSCCs(), 0u);

    auto revIt = opScc.reverseTopological_begin();
    EXPECT_EQ(cast<Operation *>(*(revIt++)), outputOp);
    EXPECT_EQ(cast<Operation *>(*(revIt++)), regOp);
    EXPECT_EQ(revIt, opScc.reverseTopological_end());
  }
}

// IR for casting test: a reg<->and cycle, a leaf output, and a disconnected
// or-op that is never reached by a forward DFS from andOp.
const char *castingIr = R"MLIR(
  hw.module private @casting(in %clock: !seq.clock, in %reset: i1, in %a: i1, out x: i1) {
    %reg     = seq.firreg %and clock %clock reset sync %reset, %and : i1
    %and     = comb.and %reg, %a : i1
    %unused  = comb.or  %a, %a : i1
    hw.output %and : i1
  }
)MLIR";

// Verify isa / dyn_cast behaviour across all three OpSCC variants.
TEST(SparseOpSCCsTest, OpSCCCasting) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<seq::SeqDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(castingIr, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("casting");
  ASSERT_TRUE(hwModule);

  auto it = hwModule.getBodyBlock()->begin();
  Operation *regOp = &*it++;
  Operation *andOp = &*it++;
  Operation *unusedOp = &*it++;
  Operation *outputOp = hwModule.getBodyBlock()->getTerminator();

  SparseOpSCC<OpSCCDirection::Forward> opScc;
  opScc.visit(andOp);

  // Obtain one entry of each kind via getSCC.
  OpSCC trivialEntry = opScc.getSCC(outputOp); // trivial: no cycle
  OpSCC cyclicEntry = opScc.getSCC(andOp);     // cyclic: reg <-> and
  OpSCC nullEntry = opScc.getSCC(unusedOp);    // not reachable from andOp

  // isa<> correctly identifies each variant.
  EXPECT_TRUE(isa<Operation *>(trivialEntry));
  EXPECT_FALSE(isa<CyclicOpSCC>(trivialEntry));

  EXPECT_FALSE(isa<Operation *>(cyclicEntry));
  EXPECT_TRUE(isa<CyclicOpSCC>(cyclicEntry));

  EXPECT_FALSE(isa<Operation *>(nullEntry));
  EXPECT_FALSE(isa<CyclicOpSCC>(nullEntry));

  // An undiscovered entry is bool-false; present entries are bool-true.
  EXPECT_FALSE(static_cast<bool>(nullEntry));
  EXPECT_TRUE(static_cast<bool>(trivialEntry));
  EXPECT_TRUE(static_cast<bool>(cyclicEntry));

  // dyn_cast on known-present values.
  EXPECT_EQ(dyn_cast<Operation *>(trivialEntry), outputOp);
  EXPECT_TRUE(static_cast<bool>(dyn_cast<Operation *>(trivialEntry)));
  EXPECT_FALSE(static_cast<bool>(dyn_cast<CyclicOpSCC>(trivialEntry)));

  EXPECT_EQ(dyn_cast<Operation *>(cyclicEntry), nullptr);
  EXPECT_FALSE(static_cast<bool>(dyn_cast<Operation *>(cyclicEntry)));
  EXPECT_TRUE(static_cast<bool>(dyn_cast<CyclicOpSCC>(cyclicEntry)));

  EXPECT_EQ(dyn_cast<CyclicOpSCC>(cyclicEntry), opScc.getSCC(regOp));
  EXPECT_NE(dyn_cast<CyclicOpSCC>(cyclicEntry), opScc.getSCC(outputOp));
  EXPECT_NE(dyn_cast<CyclicOpSCC>(cyclicEntry), opScc.getSCC(unusedOp));

  EXPECT_FALSE(static_cast<bool>(dyn_cast<CyclicOpSCC>(trivialEntry)));
  if (auto scc = dyn_cast<CyclicOpSCC>(cyclicEntry)) {
    EXPECT_EQ(scc.size(), 2u);
    EXPECT_TRUE(llvm::is_contained(scc, regOp));
    EXPECT_TRUE(llvm::is_contained(scc, andOp));
  } else {
    FAIL() << "expected CyclicOpSCC for cyclicEntry";
  }

  // dyn_cast_if_present additionally handles null entries gracefully.
  EXPECT_EQ(dyn_cast_if_present<Operation *>(nullEntry), nullptr);
  EXPECT_FALSE(static_cast<bool>(dyn_cast_if_present<Operation *>(nullEntry)));
  EXPECT_FALSE(static_cast<bool>(dyn_cast_if_present<CyclicOpSCC>(nullEntry)));
}

} // namespace
