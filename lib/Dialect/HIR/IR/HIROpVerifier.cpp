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
  auto inputTypes = funcTy.getInputTypes();
  auto resultTypes = funcTy.getResultTypes();
  if (!inputTypes.empty()) {
    auto argNames = op->getAttrOfType<ArrayAttr>("argNames");
    // argNames also contains the start time.
    if ((!argNames) || (argNames.size() - 1 != inputTypes.size()))
      return op.emitError("Mismatch in number of argument names.");
  }
  if (!resultTypes.empty()) {
    auto resultNames = op->getAttrOfType<ArrayAttr>("resultNames");
    if ((!resultNames) || (resultNames.size() != resultTypes.size()))
      return op.emitError("Mismatch in number of result names.");
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

LogicalResult verifyBusTensorMapOp(hir::BusTensorMapOp op) {
  if (op.getNumOperands() == 0)
    return op.emitError() << "Op must have atleast one operand.";
  if (op.getNumResults() == 0)
    return op.emitError() << "Op must have atleast one result.";
  auto busTensorTy =
      op.getResult(0).getType().dyn_cast_or_null<hir::BusTensorType>();
  if (!busTensorTy)
    return op.emitError(
        "Expected input and result types to be hir.bus_tensor type.");

  for (auto operand : op.operands()) {
    auto operandTy = operand.getType().dyn_cast_or_null<hir::BusTensorType>();
    if (!operandTy)
      return op.emitError(
          "Expected input and result types to be hir.bus_tensor type.");
    if (operandTy.getShape() != busTensorTy.getShape())
      return op.emitError(
          "Expected all input and result tensors to have same shape.");
  }

  for (auto result : op.results()) {
    auto resultTy = result.getType().dyn_cast_or_null<hir::BusTensorType>();
    if (!resultTy)
      return op.emitError(
          "Expected input and result types to be hir.bus_tensor type.");
    if (resultTy.getShape() != busTensorTy.getShape())
      return op.emitError(
          "Expected input and result tensors to have same shape.");
  }
  return success();
}

LogicalResult verifyYieldOp(hir::YieldOp op) {
  auto *operation = op->getParentRegion()->getParentOp();
  auto resultTypes = operation->getResultTypes();
  auto operands = op.operands();
  if (resultTypes.size() != operands.size())
    return op.emitError() << "Expected " << resultTypes.size() << " operands.";
  for (uint64_t i = 0; i < resultTypes.size(); i++) {
    if (!helper::isBuiltinSizedType(operands[i].getType()))
      return op.emitError() << "verifyYieldOp: Unsupported input type.";
  }
  return success();
}

LogicalResult verifyFuncExternOp(hir::FuncExternOp op) {
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  if (!funcTy)
    return op.emitError("OpVerifier failed. hir::FuncOp::funcTy must be of "
                        "type hir::FuncType.");
  for (Type arg : funcTy.getInputTypes()) {
    if (arg.isa<IndexType>())
      return op.emitError(
          "hir.func op does not support index type in argument location.");
  }

  auto inputTypes = funcTy.getInputTypes();
  auto resultTypes = funcTy.getResultTypes();
  if (!inputTypes.empty()) {
    auto argNames = op->getAttrOfType<ArrayAttr>("argNames");
    // argNames also contains the start time.
    if ((!argNames) || (argNames.size() - 1 != inputTypes.size()))
      return op.emitError("Mismatch in number of argument names.");
  }
  if (!resultTypes.empty()) {
    auto resultNames = op->getAttrOfType<ArrayAttr>("resultNames");
    if ((!resultNames) || (resultNames.size() != resultTypes.size()))
      return op.emitError("Mismatch in number of result names.");
  }
  return success();
}

llvm::Optional<std::string> typeMismatch(Location loc, hir::FuncType ta,
                                         hir::FuncType tb) {

  auto inputTypesA = ta.getInputTypes();
  auto inputTypesB = tb.getInputTypes();
  if (inputTypesA.size() != inputTypesB.size())
    return std::string("Mismatch with function declaration. Mismatched number "
                       "of input types.");
  for (size_t i = 0; i < inputTypesA.size(); i++) {
    if (inputTypesA[i] != inputTypesB[i])
      return std::string("Type Mismatch in input arg ") + std::to_string(i);
  }

  auto resultTypesA = ta.getResultTypes();
  auto resultTypesB = tb.getResultTypes();
  if (resultTypesA.size() != resultTypesB.size())
    return std::string("Mismatch with function declaration. Mismatched number "
                       "of result types.");
  for (size_t i = 0; i < resultTypesA.size(); i++) {
    if (resultTypesA[i] != resultTypesB[i])
      return std::string("Type Mismatch in result arg ") + std::to_string(i);
  }

  if (ta != tb)
    return std::string("Mismatch in function types.");
  return llvm::None;
}

bool isEqual(mlir::DictionaryAttr d1, mlir::DictionaryAttr d2) {

  if (!d1 && !d2)
    return true;
  if (!d1 || !d2)
    return false;

  for (auto param : d1) {
    auto value = d2.get(param.first);
    if (!value || (value.getType() != param.second.getType()))
      return false;
  }

  for (auto param : d2) {
    auto value = d1.get(param.first);
    if (!value || (value.getType() != param.second.getType()))
      return false;
  }

  return true;
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
  auto *calleeDeclOperation = op.getCalleeDecl();
  if (!calleeDeclOperation)
    return op.emitError() << "Could not find function declaration.";

  if (auto funcOp = dyn_cast_or_null<hir::FuncOp>(calleeDeclOperation)) {
    auto error =
        typeMismatch(op.getLoc(), funcOp.getFuncType(), op.getFuncType());
    if (error)
      return op.emitError("Mismatch with function definition.")
                 .attachNote(funcOp.getLoc())
             << error.getValue();
  } else if (hir::FuncExternOp funcExternOp =
                 dyn_cast_or_null<hir::FuncExternOp>(calleeDeclOperation)) {
    auto error =
        typeMismatch(op.getLoc(), funcExternOp.getFuncType(), op.getFuncType());
    if (error)
      return op.emitError("Mismatch with function declaration.")
                 .attachNote(funcExternOp.getLoc())
             << error.getValue();
  }
  auto callOpParams = op->getAttrOfType<mlir::DictionaryAttr>("params");
  auto declOpParams =
      calleeDeclOperation->getAttrOfType<mlir::DictionaryAttr>("params");
  if (!isEqual(callOpParams, declOpParams))
    return op.emitError(
        "Mismatch in params attribute between function declaration and use.");
  return success();
}

LogicalResult verifyCastOp(hir::CastOp op) {
  auto inputHWType = helper::convertToHWType(op.input().getType());
  auto resultHWType = helper::convertToHWType(op.res().getType());
  if (!inputHWType)
    return op.emitError()
           << "Input type should be convertible to a valid hardware type.";
  if (!resultHWType)
    return op.emitError()
           << "Result type should be convertible to a valid hardware type.";
  if (*inputHWType != *resultHWType)
    return op.emitError() << "Incompatible Input and Result types.";

  return success();
}

LogicalResult verifyDelayOp(hir::DelayOp op) {
  Type inputTy = op.input().getType();
  if (helper::isBuiltinSizedType(inputTy) || helper::isBusLikeType(inputTy))
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
LogicalResult verifyNextIterOp(hir::NextIterOp op) {
  if (op.condition() && isa<hir::ForOp>(op->getParentOp()))
    return op.emitError("condition is not supported in hir.next_iter when it "
                        "is inside a hir.for op.");
  return success();
}

LogicalResult verifyProbeOp(hir::ProbeOp op) {
  auto ty = op.input().getType();
  if (!(helper::isBuiltinSizedType(ty) || ty.isa<hir::TimeType>() ||
        helper::isBusLikeType(ty)))
    return op.emitError() << "Unsupported type for hir.probe.";
  if (op.verilog_name().size() == 0 || op.verilog_name().startswith("%") ||
      isdigit(op.verilog_name().data()[0]))
    return op.emitError() << "Invalid name.";
  return success();
}

LogicalResult verifyBusTensorInsertElementOp(hir::BusTensorInsertElementOp op) {
  auto busTensorTy = op.tensor().getType().dyn_cast<hir::BusTensorType>();
  if (!busTensorTy)
    return op.emitError() << "Expected BusTensorType, got "
                          << op.tensor().getType();
  auto busTy = op.element().getType().dyn_cast<hir::BusType>();
  if (!busTy)
    return op.emitError() << "Expected BusType, got " << op.element().getType();
  if (busTy.getElementType() != busTensorTy.getElementType())
    return op.emitError() << "Incompatible input types";
  return success();
}
//-----------------------------------------------------------------------------
} // namespace hir
} // namespace circt
