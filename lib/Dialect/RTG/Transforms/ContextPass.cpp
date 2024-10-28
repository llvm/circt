//===- ContextPass.cpp - RTG ContextPass implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass elaborates the contexts of the RTG dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <random>

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_CONTEXTPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace rtg;

#define DEBUG_TYPE "rtg-context"

namespace {
class ContextPass : public rtg::impl::ContextPassBase<ContextPass> {
public:
  ContextPass() : ContextPassBase() {}
  ContextPass(const ContextPass &other) : ContextPassBase(other) {}
  void runOnOperation() override;
};
} // end namespace

template <typename T>
static size_t indexOf(T val, ArrayRef<T> arr) {
  auto it = std::find(arr.begin(), arr.end(), val);
  if (it == arr.end()) {
    return -1;
  }
  return std::distance(arr.begin(), it);
}

void ContextPass::runOnOperation() {
  auto moduleOp = getOperation();
  // TODO: Use a proper visitor
  moduleOp.walk([&](RenderedContextOp rc2) {
    if (auto rc1 = dyn_cast_or_null<RenderedContextOp>(rc2->getPrevNode())) {
      DenseMap<int64_t, std::pair<Block *, Block *>> newRegions;
      for (auto [sel, reg] : llvm::zip(rc1.getSelectors(), rc1.getRegions())) {
        auto &newReg = newRegions[sel];
        newReg.first = &reg->getBlocks().front();
      }
      for (auto [sel, reg] : llvm::zip(rc2.getSelectors(), rc2.getRegions())) {
        auto &newReg = newRegions[sel];
        newReg.second = &reg->getBlocks().front();
      }

      SmallVector<int64_t> newSel;
      for (auto [i, r] : newRegions)
        newSel.push_back(i);
      std::sort(newSel.begin(), newSel.end());
      OpBuilder builder(rc1);
      auto newOp = builder.create<RenderedContextOp>(rc1.getLoc(), newSel,
                                                     newSel.size());
      for (auto offset : llvm::seq(newRegions.size())) {
        auto [b1, b2] = newRegions[newSel[offset]];
        auto &newReg = newOp.getRegion(offset);
        auto *nb = new Block();
        newReg.push_back(nb);
        if (b1)
          nb->getOperations().splice(nb->getOperations().end(),
                                     b1->getOperations());
        if (b2)
          nb->getOperations().splice(nb->getOperations().end(),
                                     b2->getOperations());
      }
      rc1.erase();
      rc2.erase();
    }
  });
}
