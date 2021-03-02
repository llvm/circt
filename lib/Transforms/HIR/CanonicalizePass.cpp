//=========- CanonicalizationPass.cpp - Canonicalize varius instructions---===//
//
// This file implements the HIR canonicalization pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
using namespace mlir;
namespace {

class CanonicalizationPass
    : public hir::CanonicalizationBase<CanonicalizationPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

void CanonicalizationPass::runOnOperation() {}
namespace mlir {
namespace hir {
std::unique_ptr<OperationPass<hir::DefOp>> createCanonicalizationPass() {
  return std::make_unique<CanonicalizationPass>();
}
} // namespace hir
} // namespace mlir
