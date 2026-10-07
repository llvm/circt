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
using llvm::SmallDenseMap;
using llvm::SmallDenseSet;

//===----------------------------------------------------------------------===//
// Outliner
//===----------------------------------------------------------------------===//

namespace {
/// A color assigned to an operation or arc-breaking operand.
struct Color {
  unsigned index;
  SmallDenseSet<OpOperand *, 4> sinks;
};

/// Hash slots in the interned allocator as if they were the pointed-to value
/// itself.
struct ColorDenseMapInfo : DenseMapInfo<Color *> {
  static unsigned getHashValue(const Color *color) {
    return mlir::hash_combine(color->sinks);
  }
  static bool isEqual(const Color *lhs, const Color *rhs) {
    if (!lhs || !rhs)
      return lhs == rhs;
    return lhs->sinks == rhs->sinks;
  }
};

struct ColorTable {
  DenseSet<Color *, ColorDenseMapInfo> interned;
  BumpPtrAllocator &allocator;
};

} // namespace

//===----------------------------------------------------------------------===//
// Outliner
//===----------------------------------------------------------------------===//
namespace {

/// A helper struct to outline the ops in a module.
struct Outliner {
  Outliner(hw::HWModuleOp moduleOp, SymbolTable &symbolTable)
      : moduleOp(moduleOp), symbolTable(symbolTable) {}
  void run();

  hw::HWModuleOp moduleOp;
  SymbolTable &symbolTable;

  /// All interned colors.
  ColorTable colors;
  /// The colors assigned to operands of arc-breaking ops.
  DenseMap<OpOperand *, Color> sinkColors;
};
} // namespace

static bool isArcBreaker(Operation *op) {
  return mlir::isMemoryEffectFree(op) && !op->hasTrait<OpTrait::IsTerminator>();
}

void Outliner::run() {
  LLVM_DEBUG(llvm::dbgs() << "Outlining arcs in @" << moduleOp.getModuleName()
                          << "\n");
  for (auto &op : *moduleOp.getBodyBlock()) {
    if (isArcBreaker(&op))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "- Side-effecting " << op << "\n");
  }
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct OutlineArcsPass
    : public arc::impl::OutlineArcsPassBase<OutlineArcsPass> {
  void runOnOperation() override;
};
} // namespace

void OutlineArcsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto moduleOp : getOperation().getOps<hw::HWModuleOp>()) {
    Outliner outliner(moduleOp, symbolTable);
    outliner.run();
  }
}
