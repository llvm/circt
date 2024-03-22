//===- PrintOperationCounts.cpp - Strip debug information selectively ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct PrintOperationCounts
    : public circt::PrintOperationCountsBase<PrintOperationCounts> {
  void runOnOperation() override {
    llvm::StringMap<size_t> counts;
    unsigned total = 0;
    getOperation().walk([&](Operation *op) {
      ++total;
      ++counts[op->getName().getStringRef()];
    });
    SmallVector<std::pair<StringRef, unsigned>> stats;
    for (auto op : counts.keys())
      stats.push_back({op, counts[op]});
    std::sort(stats.begin(), stats.end());
    llvm::outs() << total * sizeof(Operation) / 1000000 << " MB\n";
    for (auto [op, count] : stats)
      llvm::outs() << op << " " << count << " " << ((float)count / total) * 100
                   << " " << count * sizeof(Operation) / 1000000 << " MB\n";
  }
};
} // namespace

namespace circt {
/// Creates a pass to strip debug information from a function.
std::unique_ptr<Pass> createPrintOperationCountsPass() {
  return std::make_unique<PrintOperationCounts>();
}
} // namespace circt
