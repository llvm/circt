#include "HIROpVerifier.h"
namespace mlir {
namespace hir {
LogicalResult verifySelectOp(hir::SelectOp op) {
  for (auto idx : op.indices()) {
    if (idx.getType().isa<IndexType>() || idx.getType().isa<hir::ConstType>())
      continue;
    return op.emitError("Indices can only be constants or index types!");
  }
  return success();
}
} // namespace hir
} // namespace mlir
