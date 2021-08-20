#include "HIROpVerifier.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
using namespace circt;
using namespace hir;
//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------
namespace {
LogicalResult verifyTimeAndOffset(Value time, llvm::Optional<uint64_t> offset) {
  if (time && !offset.hasValue())
    return failure();
  if (offset.hasValue() && offset.getValue() < 0)
    return failure();
  return success();
}

LogicalResult checkRegionCaptures(Region &region) {
  SmallVector<InFlightDiagnostic> errorMsgs;
  mlir::visitUsedValuesDefinedAbove(region, [&errorMsgs](OpOperand *operand) {
    Type ty = operand->get().getType();
    if (ty.isa<hir::MemrefType>() || ty.isa<hir::BusType>())
      return;
    if (ty.isa<hir::TimeType>())
      errorMsgs.push_back(
          operand->getOwner()->emitError()
          << "hir::TimeType can not be captured by this region. Only the "
             "region's "
             "own time-var (and any other time-var derived from it) is "
             "available inside the region.");
    return;
  });
  if (!errorMsgs.empty())
    return failure();
  return success();
}
} // namespace
//-----------------------------------------------------------------------------

namespace circt {
namespace hir {
//-----------------------------------------------------------------------------
// HIR Op verifiers.
//-----------------------------------------------------------------------------
LogicalResult verifyFuncOp(hir::FuncOp op) {
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  if (!funcTy)
    return op.emitError("OpVerifier failed. hir::FuncOp::funcTy must be of "
                        "type hir::FuncType.");
  for (Type arg : funcTy.getInputTypes()) {
    if (arg.isa<IndexType>())
      return op.emitError(
          "hir.func op does not support index type in argument location.");
  }
  for (uint64_t i = 0; i < op.getFuncBody().getNumArguments(); i++) {
    Value arg = op.getFuncBody().getArguments()[i];
    if (arg.getType().isa<hir::MemrefType>()) {
      for (auto &use : arg.getUses()) {
        if (auto loadOp = dyn_cast<hir::LoadOp>(use.getOwner())) {
          auto port = loadOp.port();
          if (!port)
            continue;
          auto ports = helper::extractMemrefPortsFromDict(
              op.getFuncType().getInputAttrs()[i]);
          auto loadOpPort = ports.getValue()[port.getValue()];
          if (!helper::isRead(loadOpPort))
            return use.getOwner()->emitError()
                   << "specified port is not a read port.";
        } else if (auto storeOp = dyn_cast<hir::StoreOp>(use.getOwner())) {
          auto port = storeOp.port();
          if (!port)
            continue;
          auto ports = helper::extractMemrefPortsFromDict(
              op.getFuncType().getInputAttrs()[i]);
          auto storeOpPort = ports.getValue()[port.getValue()];
          if (!helper::isWrite(storeOpPort))
            return use.getOwner()->emitError()
                   << "specified port is not a write port.";
        }
      }
    }
  }
  return success();
}

LogicalResult verifyAllocaOp(hir::AllocaOp op) {
  auto res = op.res();
  auto ports = op.ports();
  for (auto &use : res.getUses()) {
    if (auto loadOp = dyn_cast<hir::LoadOp>(use.getOwner())) {
      auto port = loadOp.port();
      if (!port)
        continue;
      auto loadOpPort = ports[port.getValue()];
      if (!helper::isRead(loadOpPort))
        return use.getOwner()->emitError()
               << "specified port is not a read port.";
    } else if (auto storeOp = dyn_cast<hir::StoreOp>(use.getOwner())) {
      auto port = storeOp.port();
      if (!port)
        continue;
      auto storeOpPort = ports.getValue()[port.getValue()];
      if (!helper::isWrite(storeOpPort))
        return use.getOwner()->emitError()
               << "specified port is not a write port.";
    }
  }
  return success();
}

LogicalResult verifyCallOp(hir::CallOp op) {
  auto inputTypes = op.getFuncType().getInputTypes();
  auto operands = op.getOperands();
  for (uint64_t i = 0; i < operands.size(); i++) {
    if (operands[i].getType() != inputTypes[i]) {
      op.emitError() << "input arg " << i
                     << " expects different type than prior uses: '"
                     << inputTypes[i] << "' vs '" << operands[i].getType();
      return failure();
    }
  }
  return success();
}

LogicalResult verifyDelayOp(hir::DelayOp op) {
  Type inputTy = op.input().getType();
  if (helper::isBuiltinSizedType(inputTy) || helper::isBusType(inputTy))
    return success();
  return op.emitError("hir.delay op only supports signless-integer, float "
                      "and tuple/tensor of these types.");
}

LogicalResult verifyLatchOp(hir::LatchOp op) {
  if (failed(verifyTimeAndOffset(op.tstart(), op.offset())))
    return op.emitError("Invalid offset.");
  return success();
}

LogicalResult verifyForOp(hir::ForOp op) {
  if (failed(verifyTimeAndOffset(op.tstart(), op.offset())))
    return op.emitError("Invalid offset.");
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

  for (Value capture : op.captures())
    if (capture.getType().isa<hir::TimeType>())
      return op.emitError(
          "hir::TimeType can not be captured in a 'latch' operand.");
  if (failed(checkRegionCaptures(op.getLoopBody())))
    return failure();

  return success();
}

LogicalResult verifyLoadOp(hir::LoadOp op) {
  if (failed(verifyTimeAndOffset(op.tstart(), op.offset())))
    return op.emitError("Invalid offset.");
  return success();
}
LogicalResult verifyStoreOp(hir::StoreOp op) {
  if (failed(verifyTimeAndOffset(op.tstart(), op.offset())))
    return op.emitError("Invalid offset.");
  return success();
}

LogicalResult verifyIfOp(hir::IfOp op) {
  if (failed(verifyTimeAndOffset(op.tstart(), op.offset())))
    return op.emitError("Invalid offset.");
  if (failed(checkRegionCaptures(op.if_region())))
    return failure();
  if (failed(checkRegionCaptures(op.else_region())))
    return failure();
  return success();
}
//-----------------------------------------------------------------------------
} // namespace hir
} // namespace circt
