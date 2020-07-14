//===- LLHDCanonicalization.cpp - Register LLHD Canonicalization Patterns -===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "circt/Dialect/LLHD/IR/LLHDCanonicalization.inc"
} // anonymous namespace

void llhd::XorOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<XorAllBitsSet>(context);
}
