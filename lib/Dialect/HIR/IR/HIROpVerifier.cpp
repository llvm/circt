#include "HIROpVerifier.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
namespace circt {
namespace hir {

LogicalResult verifyFuncOp(hir::FuncOp op) {
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  if (!funcTy)
    return op.emitError("OpVerifier failed. hir::FuncOp::funcTy must be of "
                        "type hir::FuncType.");
  for (Type arg : funcTy.getInputTypes())
    if (arg.isa<IndexType>())
      return op.emitError(
          "hir.func op does not support index type in argument location.");

  return success();
}

LogicalResult verifyDelayOp(hir::DelayOp op) {
  if (!helper::isBuiltinSizedType(op.input().getType()))
    return op.emitError("hir.delay op only supports signless-integer, float "
                        "and tuple/tensor of these types.");
  return success();
}

LogicalResult verifyForOp(hir::ForOp op) {
  auto ivTy = op.getInductionVar().getType();
  if (op->getAttr("unroll"))
    if (!ivTy.isa<IndexType>())
      return op.emitError("Expected induction-var to be IndexType for loop "
                          "with 'unroll' attribute.");
  if (!ivTy.isIntOrIndex())
    return op.emitError(
        "Expected induction var to be IntegerType or IndexType.");
  if (op.lb().getType() != ivTy)
    return op.emitError("Expected lower bound to be of type ") << ivTy << ".";
  if (op.ub().getType() != ivTy)
    return op.emitError("Expected upper bound to be of type ") << ivTy << ".";
  if (op.step().getType() != ivTy)
    return op.emitError("Expected step size to be of type ") << ivTy << ".";
  if (op.getInductionVar().getType() != ivTy)
    return op.emitError("Expected induction var to be of type ") << ivTy << ".";
  if (!op.getIterTimeVar().getType().isa<hir::TimeType>())
    return op.emitError("Expected time var to be of !hir.time type.");

  return success();
}
} // namespace hir
} // namespace circt
