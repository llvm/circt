//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Path.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_PRINTTESTNAMESPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Print Test Names Pass
//===----------------------------------------------------------------------===//

namespace {
struct PrintTestNamesPass
    : public rtg::impl::PrintTestNamesPassBase<PrintTestNamesPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void PrintTestNamesPass::runOnOperation() {
  auto output = createOutputFile(filename, "", [&]() {
    return mlir::emitError(UnknownLoc::get(&getContext()));
  });
  if (!output)
    return signalPassFailure();

  llvm::raw_ostream &os = output->os();

  for (auto testOp : getOperation().getOps<rtg::TestOp>())
    os << testOp.getSymName() << "," << testOp.getTemplateName() << "\n";

  output->keep();
  markAllAnalysesPreserved();
}
