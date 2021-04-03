#include "HIROpVerifier.h"
namespace mlir {
namespace hir {
LogicalResult verifySliceOp(hir::SliceOp op) {
  for (auto addr : op.varAddrs()) {
    if (addr.getType().isa<IndexType>() || addr.getType().isa<hir::ConstType>())
      continue;
    return op.emitError("Indices can only be constants or index types!");
  }
  return success();
}
} // namespace hir
} // namespace mlir
