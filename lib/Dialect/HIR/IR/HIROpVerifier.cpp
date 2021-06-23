#include "HIROpVerifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
namespace circt {
namespace hir {

LogicalResult verifyStandardOp(mlir::IndexCastOp op) {
  if (op.getType().isa<IndexType>())
    return op.emitError(
        "index_cast op can only cast from index type to an integer type.");
  return success();
}

LogicalResult verifyFuncOp(hir::FuncOp funcOp) {
  WalkResult result = funcOp.walk([](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<mlir::IndexCastOp>(operation))
      if (failed(verifyStandardOp(op)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return success();
}

} // namespace hir
} // namespace circt
