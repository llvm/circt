//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

#define DEBUG_TYPE "arc-outline"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_OUTLINEARCSPASS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;
using llvm::BumpPtrAllocator;
using llvm::SmallDenseMap;
using llvm::SmallDenseSet;

namespace {
struct OutlineArcsPass
    : public arc::impl::OutlineArcsPassBase<OutlineArcsPass> {
  void runOnOperation() override;
  friend struct Colorer;
  friend struct Outliner;
};
} // namespace

static bool isArcBreaker(Operation *op) {
  return !mlir::isMemoryEffectFree(op) ||
         op->hasTrait<OpTrait::IsTerminator>() ||
         op->hasTrait<OpTrait::ConstantLike>() ||
         // `llhd.prb` changes memory effects based on its parent op. Yikes.
         isa<llhd::ProbeOp>(op);
}

//===----------------------------------------------------------------------===//
// Coloring
//===----------------------------------------------------------------------===//

namespace {

/// A color assigned to an operation or arc-breaking operand.
struct Color {
  /// A unique identifier for this color. Mainly for debugging and picking names
  /// for outlined arcs.
  unsigned index;
  /// A unique and sorted list of terminal colors this color feeds into. A
  /// terminal is an `OpOperand` on an arc-breaking operation.
  ArrayRef<unsigned> terminals;
};

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Color &color) {
  os << "#" << color.index << "[";
  llvm::interleaveComma(color.terminals, os);
  os << "]";
  return os;
}

/// Special handling for the set interning `Color` instances.
struct ColorInfo : DenseMapInfo<Color *> {
  static unsigned getHashValue(const Color *color) {
    return llvm::hash_combine_range(color->terminals);
  }
  static unsigned getHashValue(ArrayRef<unsigned> terminals) {
    return llvm::hash_combine_range(terminals);
  }
  static bool isEqual(const Color *lhs, const Color *rhs) {
    return lhs == rhs || lhs->terminals == rhs->terminals;
  }
  static bool isEqual(ArrayRef<unsigned> lhs, const Color *rhs) {
    return lhs == rhs->terminals;
  }
};

/// A table of interned colors.
struct ColorTable {
  DenseSet<Color *, ColorInfo> colors;
  BumpPtrAllocator alloc;

  Color *get(ArrayRef<unsigned> terminals) {
    // Check if we already have a color for these terminals.
    if (auto it = colors.find_as(terminals); it != colors.end())
      return *it;

    // Otherwise allocate a new color.
    auto *color = new (alloc) Color{colors.size(), terminals.copy(alloc)};
    colors.insert(color);
    return color;
  }
};

/// The result of coloring the ops in a module.
struct Coloring {
  /// The color allocator and interning table.
  ColorTable table;
  /// The colors assigned to each non-arc-breaking operation.
  DenseMap<Operation *, Color *> colors;
};

/// A helper struct that colors the ops in a module.
struct Colorer {
  Colorer(hw::HWModuleOp moduleOp, OutlineArcsPass &pass)
      : moduleOp(moduleOp), pass(pass) {}

  Coloring run();
  void colorOpFanout(Operation *op);
  bool updateOpColor(Operation *op);
  void outlineOps();
  void outlineOp(Operation *op);

  hw::HWModuleOp moduleOp;
  OutlineArcsPass &pass;

  /// The resulting coloring.
  Coloring coloring;
  /// The terminal IDs assigned to operands of arc-breaking ops.
  DenseMap<OpOperand *, unsigned> terminals;

  /// The worklist used to color operations.
  struct WorklistItem {
    Operation *op;
    Operation::use_iterator use, end;
  };
  SmallVector<WorklistItem> worklist;

  /// A worklist of operations that require a lattice-based fixup after the
  /// initial DFS color assignment. This allows us to properly color loops which
  /// the DFS part misses.
  SetVector<Operation *> dirtyOps;
};

} // namespace

/// Color the operations in the module.
Coloring Colorer::run() {
  // Perform a first depth-first traversal along the op user chain.
  for (auto &op : *moduleOp.getBodyBlock())
    if (!isArcBreaker(&op))
      colorOpFanout(&op);

  // To support cycles, perform a lattice-based color update pass until all op
  // colors stabilize.
  while (!dirtyOps.empty()) {
    auto *op = dirtyOps.pop_back_val();
    pass.numFixups++;
    if (updateOpColor(op)) {
      op->walk([&](Operation *op) {
        for (auto operand : op->getOperands())
          if (auto *defOp = operand.getDefiningOp())
            if (defOp->getBlock() == moduleOp.getBodyBlock())
              if (!isArcBreaker(defOp))
                dirtyOps.insert(defOp);
      });
    }
  }

  return std::move(coloring);
}

/// Color an operation and transitively all its users.
void Colorer::colorOpFanout(Operation *op) {
  // Assign a sentinel color to the op such that we can detect recursion. If the
  // insertion fails, we already have a color for the op.
  if (!coloring.colors.insert({op, nullptr}).second)
    return;

  // Perform a depth-first traversal of the op's users and color each. An op's
  // color is the union of the arc-breaking op operands in its fanout.
  worklist.push_back({op, op->use_begin(), op->use_end()});
  while (!worklist.empty()) {
    auto &item = worklist.back();
    if (item.use == item.end) {
      // All uses are colored. Time to color this op and pop it off the
      // worklist.
      updateOpColor(item.op);
      worklist.pop_back();
    } else {
      // Lookup the user op and advance the use iterator. If the user is in a
      // nested op, zip up to the parent op that sits directly in the module.
      auto *userOp = (item.use++)->getOwner();
      userOp = moduleOp.getBodyBlock()->findAncestorOpInBlock(*userOp);

      // Push the user op onto the worklist if it isn't an arc-breaking op and
      // the op hasn't already been colored (or has a sentinel null color to
      // break cycles).
      if (!isArcBreaker(userOp))
        if (coloring.colors.insert({userOp, nullptr}).second)
          worklist.push_back({userOp, userOp->use_begin(), userOp->use_end()});
    }
  }
}

/// Update the color of an operation based on the colors of its users. Returns
/// true if the color changed, false otherwise.
bool Colorer::updateOpColor(Operation *op) {
  // Compute the union of all terminals in the op's users.
  SmallDenseSet<unsigned, 8> fanoutTerminals;
  for (auto &use : op->getUses()) {
    if (isArcBreaker(use.getOwner())) {
      // The use is an operand of an arc-breaking op. Add that operand's unique
      // terminal ID to the set.
      auto terminal = terminals.insert({&use, terminals.size()}).first->second;
      fanoutTerminals.insert(terminal);
    } else if (auto *color = coloring.colors.at(use.getOwner())) {
      // Add the user op's terminals.
      fanoutTerminals.insert_range(color->terminals);
    } else {
      // If we get here we've hit a null color sentinel indicating that this op
      // is on a cycle. Mark it as dirty such that its color gets updated during
      // the lattice-based fixup after the initial DFS coloring.
      dirtyOps.insert(op);
    }
  }

  // Sort the terminals such that we can assign them a color.
  SmallVector<unsigned, 8> sortedTerminals;
  sortedTerminals.reserve(fanoutTerminals.size());
  sortedTerminals.append(fanoutTerminals.begin(), fanoutTerminals.end());
  llvm::sort(sortedTerminals);

  // Turn the list of terminals into a color and update the op's color.
  auto &color = coloring.colors[op];
  auto *oldColor = color;
  color = coloring.table.get(sortedTerminals);
  pass.numUpdates++;
  if (color != oldColor)
    return true;
  pass.numVacuousUpdates++;
  return false;
}

//===----------------------------------------------------------------------===//
// Outlining
//===----------------------------------------------------------------------===//

namespace {

/// An outlined set of operations.
struct OutlinedArc {
  unsigned index;
  std::unique_ptr<Block> block;
  SmallVector<OpOperand *> forwardRefs;
  OutputOp terminator;
  CallOp call;
};

/// A helper struct to outline the colored ops in a module.
struct Outliner {
  Outliner(hw::HWModuleOp moduleOp, SymbolTable &symbolTable,
           Coloring &coloring, OutlineArcsPass &pass)
      : moduleOp(moduleOp), symbolTable(symbolTable), coloring(coloring),
        pass(pass) {}

  void run();
  void outlineOp(Operation *op);
  void sortAndBreakCycles(OutlinedArc &arc);
  void createTerminator(OutlinedArc &arc);
  void createAndCallArc(OutlinedArc &arc, OpBuilder &arcBuilder,
                        OpBuilder &callBuilder);
  void useCallResults(OutlinedArc &arc);

  hw::HWModuleOp moduleOp;
  SymbolTable &symbolTable;
  Coloring &coloring;
  OutlineArcsPass &pass;

  /// The outlined block for each color.
  MapVector<Color *, OutlinedArc> arcs;
};

} // namespace

void Outliner::run() {
  // Move all colored ops into a dedicated block for each color.
  for (auto &op : llvm::make_early_inc_range(*moduleOp.getBodyBlock()))
    if (!isArcBreaker(&op))
      outlineOp(&op);
  pass.numArcs += arcs.size();

  // Sort the ops in outlined blocks and take note of any cycles.
  for (auto &[color, arc] : arcs)
    sortAndBreakCycles(arc);

  // Add a terminator to each block.
  for (auto &[color, arc] : arcs)
    createTerminator(arc);

  // Create a call in the module body for each outlined block.
  OpBuilder arcBuilder(moduleOp);
  OpBuilder callBuilder(moduleOp.getBodyBlock()->getTerminator());
  for (auto &[color, arc] : arcs)
    createAndCallArc(arc, arcBuilder, callBuilder);

  // Replace any direct uses of results in an arc with the corresponding call
  // op's result.
  for (auto &[color, arc] : arcs)
    useCallResults(arc);
}

/// Move the given operation into the outlined arc for its color.
void Outliner::outlineOp(Operation *op) {
  auto *color = coloring.colors.at(op);
  auto &arc = arcs[color];
  if (!arc.block) {
    arc.index = arcs.size();
    arc.block = std::make_unique<Block>();
    LLVM_DEBUG(llvm::dbgs() << "- Create arc " << arc.index << " for color "
                            << *color << "\n");
  }
  op->moveBefore(arc.block.get(), arc.block->end());
}

/// Sort the ops in an outlined block topologically and note any uses of a
/// result before the defining op. We'll later move the cycle outside the arc
/// such that the arc body can be an SSACFG region.
void Outliner::sortAndBreakCycles(OutlinedArc &arc) {
  // Topologically sort the arc body. If the function returns true there are no
  // cycles and we don't need to break any cycles.
  if (mlir::sortTopologically(arc.block.get()))
    return;

  // Otherwise take note of any uses-before-defs.
  arc.block->walk([&](Operation *op) {
    auto *blockOp = arc.block->findAncestorOpInBlock(*op);
    for (auto &operand : op->getOpOperands())
      if (auto *defOp = operand.get().getDefiningOp())
        if (defOp->getBlock() == arc.block.get() &&
            !defOp->isBeforeInBlock(blockOp))
          arc.forwardRefs.push_back(&operand);
  });
  pass.numForwardRefs += arc.forwardRefs.size();

  LLVM_DEBUG({
    if (!arc.forwardRefs.empty())
      llvm::dbgs() << "- Arc " << arc.index << " has " << arc.forwardRefs.size()
                   << " forward references\n";
  });
}

void Outliner::createTerminator(OutlinedArc &arc) {
  // Collect all op results that have uses outside the block.
  SmallSetVector<Value, 8> results;
  for (auto &op : *arc.block) {
    for (auto result : op.getResults()) {
      for (auto *userOp : result.getUsers()) {
        if (!arc.block->findAncestorOpInBlock(*userOp)) {
          results.insert(result);
          break;
        }
      }
    }
  }

  // Append values involved in a forward reference.
  for (auto *operand : arc.forwardRefs)
    results.insert(operand->get());

  // Create an `arc.output` terminator op with these results as operands.
  OpBuilder builder(moduleOp);
  builder.setInsertionPointToEnd(arc.block.get());
  arc.terminator =
      OutputOp::create(builder, moduleOp.getLoc(), results.getArrayRef());
}

void Outliner::createAndCallArc(OutlinedArc &arc, OpBuilder &arcBuilder,
                                OpBuilder &callBuilder) {
  // Collect all values used in the block that are defined outside the block,
  // and create a block argument for each.
  SmallVector<Value, 8> values;
  SmallMapVector<Value, Value, 8> args;

  auto addArg = [&](Value value) {
    auto &arg = args[value];
    if (!arg) {
      arg = arc.block->addArgument(value.getType(), value.getLoc());
      values.push_back(value);
    }
    return arg;
  };

  OpBuilder constBuilder(arc.terminator);
  constBuilder.setInsertionPointToStart(arc.block.get());

  arc.block->walk([&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      // Get the defining block and zip up to the root block. Since the arc
      // blocks are all detached at the moment, values defined inside an arc
      // will stop at the arc's block, and all others will zip all the way up to
      // the root module. We are only interested in values where this block is
      // not the arc's block.
      auto *defBlock = operand.get().getParentBlock();
      while (defBlock && defBlock->getParentOp())
        defBlock = defBlock->getParentOp()->getBlock();
      if (defBlock == arc.block.get())
        continue;

      // Clone constant-like ops into the arc.
      if (auto *defOp = operand.get().getDefiningOp();
          defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
        if (!args.count(operand.get())) {
          pass.numClonedConstants++;
          auto *clonedOp = constBuilder.clone(*defOp);
          for (auto [oldResult, newResult] :
               llvm::zip(defOp->getResults(), clonedOp->getResults()))
            args[oldResult] = newResult;
        }
        operand.set(args.at(operand.get()));
        continue;
      }

      // Create a block argument if none exists yet.
      operand.set(addArg(operand.get()));
    }
  });

  // Add arguments for all forward references.
  for (auto *operand : arc.forwardRefs)
    operand->set(addArg(operand->get()));

  // Create an arc definition.
  auto arcOp = DefineOp::create(
      arcBuilder, arc.terminator.getLoc(),
      arcBuilder.getStringAttr(moduleOp.getModuleName() + "_arc" +
                               Twine(arc.index)),
      arcBuilder.getFunctionType(arc.block->getArgumentTypes(),
                                 arc.terminator->getOperandTypes()));
  arcOp.getBody().push_back(arc.block.release());

  // Create a call to the arc.
  arc.call = CallOp::create(callBuilder, arcOp.getLoc(), arcOp, values);
}

/// Replace any external uses of an arc's internal results with the
/// corresponding call result.
void Outliner::useCallResults(OutlinedArc &arc) {
  auto *arcBlock = arc.terminator->getBlock();
  for (auto [valueInside, valueOutside] :
       llvm::zip(arc.terminator.getOperands(), arc.call.getResults())) {
    valueInside.replaceUsesWithIf(valueOutside, [&](OpOperand &use) {
      return !arcBlock->findAncestorOpInBlock(*use.getOwner());
    });
  }
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void OutlineArcsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto moduleOp : getOperation().getOps<hw::HWModuleOp>()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Outline arcs from @" << moduleOp.getModuleName() << "\n");
    auto coloring = Colorer(moduleOp, *this).run();
    Outliner(moduleOp, symbolTable, coloring, *this).run();
  }
}
