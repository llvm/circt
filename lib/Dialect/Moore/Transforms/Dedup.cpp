//===- Dedup.cpp - Moore module deduping --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file  implements moore module deduplication.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_DEDUP
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;

namespace {
class DedupPass : public circt::moore::impl::DedupBase<DedupPass> {
  using SymbolTable = DenseSet<mlir::StringAttr>;
  using Symbol2Symbol = DenseMap<mlir::StringAttr, SymbolTable>;
  Symbol2Symbol replacTable;
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createDedupPass() {
  return std::make_unique<DedupPass>();
}

void DedupPass::runOnOperation() {
  getOperation()->walk([&](SVModuleOp Ops) {
    mlir::OpBuilder builder(&getContext());

    // Do equiplance and record in replacTable
    // Dedup already exist module op
    for (auto &Op : Ops) {
      if (isa<SVModuleOp>(Op)) {
        ;
      }
    }

    // replace instanceop symbol to new symbol
    for (auto &Op : Ops) {
      if (isa<InstanceOp>(Op)) {
        ;
      }
    }
    return WalkResult::advance();
  });
}
