//===- PrintOpCount.cpp - Operation Count Emission Pass ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains a pass to emit operation count results
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/OpCountAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"

namespace circt {
#define GEN_PASS_DEF_PRINTOPCOUNT
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

void printOpAndOperandCounts(analysis::OpCountAnalysis &opCount,
                             raw_ostream &os, bool sorted = false) {
  auto opNames = opCount.getFoundOpNames();
  // Sort to account for non-deterministic DenseMap ordering
  if (sorted)
    llvm::sort(opNames, [](OperationName name1, OperationName name2) {
      return name1.getStringRef() < name2.getStringRef();
    });
  for (auto opName : opNames) {
    os << "- name: " << opName << "\n";
    os << "  count: " << opCount.getOpCount(opName) << "\n";
    auto operandMap = opCount.getOperandCountMap(opName);
    if (operandMap.size() <= 1)
      continue;
    llvm::SmallVector<size_t> keys;
    for (auto pair : operandMap)
      keys.push_back(pair.first);
    // Sort for determinism if required again
    if (sorted)
      llvm::sort(keys);
    for (auto num : keys) {
      os << "    - operands: " << num << "\n";
      os << "      count: " << operandMap[num] << "\n";
    }
  }
}

void printOpAndOperandJSON(analysis::OpCountAnalysis &opCount,
                           raw_ostream &os) {
  auto jos = llvm::json::OStream(os);
  jos.object([&] {
    for (auto opName : opCount.getFoundOpNames())
      jos.attributeObject(opName.getStringRef(), [&] {
        for (auto pair : opCount.getOperandCountMap(opName))
          jos.attribute(std::to_string(pair.first), pair.second);
      });
  });
}

struct PrintOpCountPass
    : public circt::impl::PrintOpCountBase<PrintOpCountPass> {
public:
  PrintOpCountPass(raw_ostream &os) : os(os) {}

  void runOnOperation() override {
    auto &opCount = getAnalysis<circt::analysis::OpCountAnalysis>();
    switch (emissionFormat) {
    case OpCountEmissionFormat::Readable:
      printOpAndOperandCounts(opCount, os);
      break;
    case OpCountEmissionFormat::ReadableSorted:
      printOpAndOperandCounts(opCount, os, /*sorted=*/true);
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
// Construct with alternative output stream where desired
std::unique_ptr<mlir::Pass> createPrintOpCountPass(llvm::raw_ostream &os) {
  return std::make_unique<PrintOpCountPass>(os);
}

// Print to outs by default
std::unique_ptr<mlir::Pass> createPrintOpCountPass() {
  return std::make_unique<PrintOpCountPass>(llvm::outs());
}
} // namespace circt
