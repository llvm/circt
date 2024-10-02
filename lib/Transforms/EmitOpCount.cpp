//===- EmitOpCount.cpp - Operation Count Emission Pass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains a pass to emit operation count results
//
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/OpCountAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_EMITOPCOUNT
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

void printOpAndOperandCounts(analysis::OpCountAnalysis &opCount,
                             raw_ostream &os, bool alphabetical = false) {
  auto opNames = opCount.getFoundOpNames();
  // Sort to account for non-deterministic DenseMap ordering
  if (alphabetical)
    llvm::sort(opNames, [](OperationName name1, OperationName name2) {
      return name1.getStringRef() < name2.getStringRef();
    });
  for (auto opName : opNames) {
    os << opName << ": " << opCount.getOpCount(opName) << "\n";
    auto operandMap = opCount.getOperandCountMap(opName);
    // Sort for determinism again
    llvm::SmallVector<size_t> keys;
    for (auto pair : operandMap)
      keys.push_back(pair.first);
    if (alphabetical)
      llvm::sort(keys);
    for (auto num : keys)
      os << " with " << num << " operands: " << operandMap[num] << "\n";
  }
}

void printOpAndOperandJSON(analysis::OpCountAnalysis &opCount,
                           raw_ostream &os) {
  auto opNames = opCount.getFoundOpNames();
  // Sort to account for non-deterministic DenseMap ordering
  os << "{\n";
  for (auto [i, opName] : llvm::enumerate(opNames)) {
    os << " \"" << opName << "\": {\n";
    auto operandMap = opCount.getOperandCountMap(opName);
    for (auto [j, pair] : llvm::enumerate(operandMap)) {
      os << "  " << pair.first << ": " << pair.second;
      if (j != operandMap.size() - 1)
        os << ",";
      os << "\n";
    }
    os << " }";
    if (i != opNames.size() - 1)
      os << ",";
    os << "\n";
  }
  os << "}\n";
}

struct EmitOpCountPass : public circt::impl::EmitOpCountBase<EmitOpCountPass> {
public:
  EmitOpCountPass(raw_ostream &os) : os(os) {}

  void runOnOperation() override {
    auto &opCount = getAnalysis<circt::analysis::OpCountAnalysis>();
    switch (emissionFormat) {
    case OpCountEmissionFormat::Readable:
      printOpAndOperandCounts(opCount, os);
      break;
    case OpCountEmissionFormat::ReadableAlphabetical:
      printOpAndOperandCounts(opCount, os, /*alphabetical=*/true);
      break;
    case OpCountEmissionFormat::JSON:
      printOpAndOperandJSON(opCount, os);
      break;
    }
  }
  /// Output stream for emission
  raw_ostream &os;
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createEmitOpCountPass() {
  return std::make_unique<EmitOpCountPass>(llvm::outs());
}
} // namespace circt
