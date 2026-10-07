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
  friend struct Outliner;
};
} // namespace

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

} // namespace

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Color &color) {
  os << "#" << color.index << "[";
  llvm::interleaveComma(color.terminals, os);
  os << "]";
  return os;
}

//===----------------------------------------------------------------------===//
// Outliner
//===----------------------------------------------------------------------===//
namespace {

/// A helper struct to outline the ops in a module.
struct Outliner {
  Outliner(hw::HWModuleOp moduleOp, SymbolTable &symbolTable,
           OutlineArcsPass &pass)
      : moduleOp(moduleOp), symbolTable(symbolTable), pass(pass) {}
  void run();
  void colorOps();
  void colorOpFanout(Operation *op);
  bool updateOpColor(Operation *op);

  hw::HWModuleOp moduleOp;
  SymbolTable &symbolTable;
  OutlineArcsPass &pass;

  /// All interned colors.
  ColorTable colors;
  /// The terminal IDs assigned to operands of arc-breaking ops.
  DenseMap<OpOperand *, unsigned> terminals;
  /// The colors assigned to each non-arc-breaking operation.
  DenseMap<Operation *, Color *> opColors;

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

static bool isArcBreaker(Operation *op) {
  return !mlir::isMemoryEffectFree(op) || op->hasTrait<OpTrait::IsTerminator>();
}

void Outliner::run() {
  LLVM_DEBUG(llvm::dbgs() << "Outlining arcs in @" << moduleOp.getModuleName()
                          << "\n");
  colorOps();
}

/// Color the operations in the module.
void Outliner::colorOps() {
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
              dirtyOps.insert(defOp);
      });
    }
  }
}

/// Color an operation and transitively all its users.
void Outliner::colorOpFanout(Operation *op) {
  // Assign a sentinel color to the op such that we can detect recursion. If the
  // insertion fails, we already have a color for the op.
  if (!opColors.insert({op, nullptr}).second)
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
      // Lookup the user op and advance the use iterator.
      auto *userOp = (item.use++)->getOwner();

      // If this use is in a nested op, zip up to the parent op that sits
      // directly in the module.
      while (userOp->getBlock() != moduleOp.getBodyBlock())
        userOp = userOp->getParentOp();

      // Push the user op onto the worklist if it isn't an arc-breaking op and
      // the op hasn't already been colored (or has a sentinel null color to
      // break cycles).
      if (!isArcBreaker(userOp))
        if (opColors.insert({userOp, nullptr}).second)
          worklist.push_back({userOp, userOp->use_begin(), userOp->use_end()});
    }
  }
}

/// Update the color of an operation based on the colors of its users. Returns
/// true if the color changed, false otherwise.
bool Outliner::updateOpColor(Operation *op) {
  // Compute the union of all terminals in the op's users.
  SmallDenseSet<unsigned, 8> fanoutTerminals;
  for (auto &use : op->getUses()) {
    if (isArcBreaker(use.getOwner())) {
      // The use is an operand of an arc-breaking op. Add that operand's unique
      // terminal ID to the set.
      auto terminal = terminals.insert({&use, terminals.size()}).first->second;
      fanoutTerminals.insert(terminal);
    } else if (auto *color = opColors.at(use.getOwner())) {
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
  auto &color = opColors[op];
  auto *oldColor = color;
  color = colors.get(sortedTerminals);
  pass.numUpdates++;
  if (color != oldColor)
    return true;
  pass.numVacuousUpdates++;
  return false;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void OutlineArcsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto moduleOp : getOperation().getOps<hw::HWModuleOp>()) {
    Outliner outliner(moduleOp, symbolTable, *this);
    outliner.run();
  }
}
