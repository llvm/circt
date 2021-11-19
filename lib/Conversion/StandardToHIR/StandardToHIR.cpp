//===----- StandardToHIR.cpp - Standard to HIR Lowering Pass-------*-C++-*-===//
//
// This pass converts standard + scf + memref to HIR dialect.
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/StandardToHIR/StandardToHIR.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include <iostream>
using namespace circt;

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------
namespace {
ArrayAttr getRegWritePort(OpBuilder &builder) {
  SmallVector<Attribute> ports;
  ports.push_back(helper::getDictionaryAttr(
      builder, "wr_latency",
      helper::getI64IntegerAttr(builder.getContext(), 1)));
  return ArrayAttr::get(builder.getContext(), ports);
}

ArrayAttr getDefaultMemrefPorts(OpBuilder &builder) {
  SmallVector<Attribute> ports;
  ports.push_back(helper::getDictionaryAttr(
      builder, "rd_latency",
      helper::getI64IntegerAttr(builder.getContext(), 1)));
  ports.push_back(helper::getDictionaryAttr(
      builder, "wr_latency",
      helper::getI64IntegerAttr(builder.getContext(), 1)));
  return ArrayAttr::get(builder.getContext(), ports);
}

LogicalResult getBankDims(MLIRContext *context, ArrayAttr &bankDims,
                          Attribute bankAttr) {
  if (!bankAttr)
    bankDims = ArrayAttr::get(context, SmallVector<Attribute>());
  else {
    bankDims = bankAttr.dyn_cast<ArrayAttr>();
    if (!bankDims)
      return failure();
    for (auto attr : bankDims)
      if (!attr.dyn_cast<IntegerAttr>())
        return failure();
  }
  return success();
}

SmallVector<hir::DimKind>
getDimKindsFromArrayOfBankDims(ArrayRef<int64_t> shape, ArrayAttr bankDims) {
  SmallVector<hir::DimKind> dimKinds;
  for (size_t i = 0; i < shape.size(); i++) {
    bool isBankDim = false;
    for (auto dim : bankDims) {
      if (dim.dyn_cast<IntegerAttr>().getInt() ==
          (int64_t)(shape.size() - i - 1)) {
        isBankDim = true;
        break;
      }
    }
    if (isBankDim)
      dimKinds.push_back(hir::DimKind::BANK);
    else
      dimKinds.push_back(hir::DimKind::ADDR);
  }
  return dimKinds;
}

unsigned int getLatencyFromMemrefAttrDict(StringRef latencyOf,
                                          DictionaryAttr dict) {
  unsigned int wrDelay = 0;
  ArrayAttr ports =
      dict.getNamed("hir.memref.ports").getValue().second.dyn_cast<ArrayAttr>();
  for (auto port : ports) {
    auto latencyAttr = port.dyn_cast<DictionaryAttr>().getNamed(latencyOf);
    if (latencyAttr.hasValue()) {
      wrDelay = latencyAttr.getValue().second.dyn_cast<IntegerAttr>().getInt();
    }
  }
  return wrDelay;
}
} // namespace
//------------------------------------------------------------------------------

namespace {
class StandardToHIRPass : public StandardToHIRBase<StandardToHIRPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (failed(convertRegion(moduleOp.body(), SmallVector<Type>(),
                             SmallVector<DictionaryAttr>()))) {
      signalPassFailure();
      return;
    }
    for (auto *op : opsToErase)
      op->erase();
  }

private:
  LogicalResult convertRegion(Region &region, ArrayRef<Type> blockArgumentTypes,
                              ArrayRef<DictionaryAttr> blockArgumentAttrs,
                              bool tstartRequired = false) {
    if (region.getBlocks().size() > 1)
      region.getParentOp()->emitError(
          "HIR only supports one block per region.");

    // Replace the old arguments in the body with new ones.
    auto &body = region.front();

    // hir::FuncOp can insert extra blockArguments for the mlir::funcOp results.
    assert(body.getNumArguments() <= blockArgumentTypes.size());
    for (size_t i = 0; i < blockArgumentTypes.size(); i++) {
      Value newArg;
      if (i < body.getNumArguments()) {
        newArg = body.insertArgument(i, blockArgumentTypes[i]);
        body.getArgument(i + 1).replaceAllUsesWith(newArg);
        body.eraseArgument(i + 1);
      } else {
        newArg = body.addArgument(blockArgumentTypes[i]);
      }
      if (auto memrefTy = blockArgumentTypes[i].dyn_cast<hir::MemrefType>()) {
        // delay;
        ;
        mapMemrefToWrDelay[newArg] =
            getLatencyFromMemrefAttrDict("wr_latency", blockArgumentAttrs[i]);
        mapMemrefToRdDelay[newArg] =
            getLatencyFromMemrefAttrDict("rd_latency", blockArgumentAttrs[i]);
      }
    }

    if (tstartRequired)
      body.addArgument(hir::TimeType::get(region.getContext()));
    // lower the ops in the region.
    for (Operation &operation : body) {
      if (auto op = dyn_cast<mlir::FuncOp>(operation)) {
        if (failed(convertOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::scf::ForOp>(operation)) {
        if (failed(convertOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::scf::YieldOp>(operation)) {
        if (failed(convertOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::memref::LoadOp>(operation)) {
        if (failed(convertOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::memref::StoreOp>(operation)) {
        if (failed(convertOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::ReturnOp>(operation)) {
        if (failed(convertOp(op)))
          return failure();
      }
    }
    return success();
  }

private: // Helper methods.
private:
  LogicalResult convertOp(mlir::FuncOp);
  LogicalResult convertOp(mlir::scf::ForOp);
  LogicalResult convertOp(mlir::scf::YieldOp);
  LogicalResult convertOp(mlir::memref::LoadOp);
  LogicalResult convertOp(mlir::memref::StoreOp);
  // LogicalResult convertOp(mlir::AddIOp);
  // LogicalResult convertOp(mlir::SubIOp);
  // LogicalResult convertOp(mlir::MulIOp);
  // LogicalResult convertOp(mlir::AddFOp);
  // LogicalResult convertOp(mlir::SubFOp);
  // LogicalResult convertOp(mlir::MulFOp);
  LogicalResult convertOp(mlir::ReturnOp);

  template <typename FromT, typename ToT>
  LogicalResult convertBinOp(FromT op) {
    OpBuilder builder(op);
    Attribute opDelayAttr = op->getAttr("hir.op.delay");
    IntegerAttr delayAttr;
    if (opDelayAttr)
      delayAttr = opDelayAttr.dyn_cast<IntegerAttr>();
    else
      delayAttr = builder.getI64IntegerAttr(0);
    Value res =
        builder.create<ToT>(op.getLoc(), op.getResult().getType(), op.lhs(),
                            op.rhs(), delayAttr, Value(), IntegerAttr());
    op.getResult().replaceAllUsesWith(res);
    opsToErase.push_back(op);
    return success();
  }

private: // State.
  SmallVector<Operation *> opsToErase;
  SmallVector<unsigned int> mapResultPosToArgumentPos;
  hir::FuncOp enclosingFuncOp;
  llvm::DenseMap<Value, unsigned int> mapMemrefToWrDelay;
  llvm::DenseMap<Value, unsigned int> mapMemrefToRdDelay;
};
} // namespace

//------------------------------------------------------------------------------
// Op conversions.
//------------------------------------------------------------------------------
LogicalResult StandardToHIRPass::convertOp(mlir::FuncOp op) {
  FunctionType originalFunctionTy = op.type().dyn_cast<FunctionType>();
  auto originalInputTypes = originalFunctionTy.getInputs();
  OpBuilder builder(op);
  auto *context = builder.getContext();
  ;
  SmallVector<Type, 4> inputTypes;
  SmallVector<DictionaryAttr, 4> inputAttrs, resultAttrs;

  // Insert FunctionOp's input types into inputTypes while converting
  // memref into hir.memref and populate inputAttrs array.
  for (int i = 0; i < (int)originalInputTypes.size(); i++) {
    Type ty = originalInputTypes[i];
    if (auto memrefTy = ty.dyn_cast<mlir::MemRefType>()) {
      ArrayAttr bankDims;
      if (failed(getBankDims(context, bankDims, op.getArgAttr(i, "hir.bank"))))
        return op.emitError(
            "'bank' attribute must be an Array of IntegerAttr.");

      auto dimKinds =
          getDimKindsFromArrayOfBankDims(memrefTy.getShape(), bankDims);

      inputTypes.push_back(hir::MemrefType::get(
          context, memrefTy.getShape(), memrefTy.getElementType(), dimKinds));
      inputAttrs.push_back(helper::getDictionaryAttr(
          builder, "hir.memref.ports", getDefaultMemrefPorts(builder)));
    } else if (helper::isBuiltinSizedType(ty)) {
      inputTypes.push_back(ty);
      inputAttrs.push_back(helper::getDictionaryAttr(
          builder, "hir.delay", builder.getI64IntegerAttr(0)));
    } else
      return op.emitError("Only mlir builtin types (except mlir::IndexType) "
                          "and memrefs are supported in input types.");
  }

  // Insert result types into inputs as a reg with write only port.
  for (Type ty : originalFunctionTy.getResults()) {
    if (helper::isBuiltinSizedType(ty)) {
      auto resultRegTy =
          hir::MemrefType::get(context, SmallVector<int64_t>({1}), ty,
                               SmallVector<hir::DimKind>({hir::DimKind::BANK}));

      // Note the arg location at which the reg corresponding to this result is
      // placed.
      mapResultPosToArgumentPos.push_back(inputTypes.size());
      inputTypes.push_back(resultRegTy);
      inputAttrs.push_back(helper::getDictionaryAttr(
          builder, "hir.memref.ports", getRegWritePort(builder)));
    } else
      return op.emitError("Only builtin types are supported in return types.");
  }

  auto funcTy =
      hir::FuncType::get(context, inputTypes, inputAttrs, SmallVector<Type>({}),
                         SmallVector<DictionaryAttr, 4>({}));
  auto funcOp = builder.create<hir::FuncOp>(
      op.getLoc(), funcTy.getFunctionType(), op.sym_name(), funcTy);
  BlockAndValueMapping mapper;

  op.getBody().cloneInto(&funcOp.getFuncBody(), mapper);

  // Replace the old arguments in the body with new ones.
  // for (size_t i = 0; i < inputTypes.size(); i++) {
  //  auto &body = funcOp.getBody().front();

  //  Value replacedArg = body.insertArgument(i, inputTypes[i]);
  //  body.getArgument(i + 1).replaceAllUsesWith(replacedArg);
  //  body.eraseArgument(i + 1);
  //}

  opsToErase.push_back(op);
  enclosingFuncOp = funcOp;
  return convertRegion(funcOp.getFuncBody(), inputTypes, inputAttrs,
                       /*tstartRequired*/ true);
}

LogicalResult StandardToHIRPass::convertOp(mlir::scf::ForOp op) {
  OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  Value lb = op.lowerBound();
  Value ub = op.upperBound();
  Value step = op.step();
  assert(lb);
  assert(ub);
  assert(step);
  // We do not support iter_args so scf.for should not have any results.
  assert(op.getNumResults() == 0);

  auto forOp = builder.create<hir::ForOp>(
      op.getLoc(), hir::TimeType::get(op.getContext()), lb, ub, step,
      SmallVector<Value>({}), Value(), IntegerAttr());
  BlockAndValueMapping mapper;
  op.getLoopBody().cloneInto(&forOp.getLoopBody(), mapper);
  if (Attribute unrollAttr = op->getAttr("hir.unroll"))
    forOp->setAttr("unroll", unrollAttr);

  auto regionOperandTypes =
      SmallVector<Type>({IndexType::get(builder.getContext())});

  opsToErase.push_back(op);
  return convertRegion(forOp.getLoopBody(),
                       {IndexType::get(builder.getContext())},
                       {helper::getDictionaryAttr(
                           builder, "hir.delay", builder.getI64IntegerAttr(0))},
                       /*tstartRequired*/ true);
}

LogicalResult StandardToHIRPass::convertOp(mlir::scf::YieldOp op) {
  OpBuilder builder(op);
  if (op.getNumOperands() > 0)
    return op->getParentOp()->emitError(
        "lower-to-hir does not support iter_args.");

  builder.create<hir::YieldOp>(op.getLoc());
  opsToErase.push_back(op);
  return success();
}

LogicalResult StandardToHIRPass::convertOp(mlir::memref::LoadOp op) {
  OpBuilder builder(op);
  auto memref = op.memref();
  if (op->getNumResults() > 1)
    return op.emitError("Only one result is supported.");
  auto res = op->getResult(0);

  auto *context = builder.getContext();

  // The memref's type is already fixed while converting the enclosing
  // FunctionOp to hir::FuncOp.
  auto hirMemrefTy = memref.getType().dyn_cast<hir::MemrefType>();
  assert(hirMemrefTy);
  auto dimKinds = hirMemrefTy.getDimKinds();
  auto shape = hirMemrefTy.getShape();
  SmallVector<Value, 6> castedIndices;
  if (op.indices().size() != shape.size()) {
    return op.emitError() << "op.indices().size() = " << op.indices().size()
                          << ", shape.size()= " << shape.size() << "\n";
  }
  for (int i = 0; i < (int)op.indices().size(); i++) {
    Value originalIdx = op.indices()[i];
    if (dimKinds[i] == hir::ADDR) {
      auto correctWidthIntTy =
          IntegerType::get(context, helper::clog2(shape[i]));
      castedIndices.push_back(
          builder
              .create<mlir::arith::IndexCastOp>(builder.getUnknownLoc(),
                                                correctWidthIntTy, originalIdx)
              .getResult());
    } else {
      castedIndices.push_back(originalIdx);
    }
  }

  auto loadOp = builder.create<hir::LoadOp>(
      op.getLoc(), res.getType(), memref, castedIndices, /*port*/ IntegerAttr(),
      builder.getI64IntegerAttr(mapMemrefToRdDelay[memref]), Value(),
      IntegerAttr());

  op->replaceAllUsesWith(loadOp);
  opsToErase.push_back(op);
  return success();
}

LogicalResult StandardToHIRPass::convertOp(mlir::memref::StoreOp op) {
  OpBuilder builder(op);
  auto memref = op.memref();
  if (op->getNumResults() > 1)
    return op.emitError("Only one result is supported.");
  auto *context = builder.getContext();

  // The memref's type is already fixed while converting the enclosing
  // FunctionOp to hir::FuncOp.
  auto hirMemrefTy = memref.getType().dyn_cast<hir::MemrefType>();
  assert(hirMemrefTy);
  auto dimKinds = hirMemrefTy.getDimKinds();
  auto shape = hirMemrefTy.getShape();
  SmallVector<Value, 6> castedIndices;
  if (op.indices().size() != shape.size()) {
    return op.emitError() << "op.indices().size() = " << op.indices().size()
                          << ", shape.size()= " << shape.size() << "\n";
  }
  for (int i = 0; i < (int)op.indices().size(); i++) {
    Value originalIdx = op.indices()[i];
    if (dimKinds[i] == hir::ADDR) {
      auto correctWidthIntTy =
          IntegerType::get(context, helper::clog2(shape[i]));
      castedIndices.push_back(
          builder
              .create<mlir::arith::IndexCastOp>(builder.getUnknownLoc(),
                                                correctWidthIntTy, originalIdx)
              .getResult());
    } else {
      castedIndices.push_back(originalIdx);
    }
  }

  auto storeOp = builder.create<hir::StoreOp>(
      op.getLoc(), op.value(), memref, castedIndices, /*port*/ IntegerAttr(),
      builder.getI64IntegerAttr(mapMemrefToWrDelay[memref]), Value(),
      IntegerAttr());
  op->replaceAllUsesWith(storeOp);
  opsToErase.push_back(op);
  return success();
}

// LogicalResult StandardToHIRPass::convertOp(mlir::AddIOp op) {
//  OpBuilder builder(op);
//  Value res =
//      builder
//          .create<hir::AddIOp>(op.getLoc(), op.getResult().getType(),
//          op.lhs(),
//                               op.rhs(), IntegerAttr(), Value(), Value())
//          .getResult();
//  op.getResult().replaceAllUsesWith(res);
//  opsToErase.push_back(op);
//  return success();
//}
//
// LogicalResult StandardToHIRPass::convertOp(mlir::SubIOp op) {
//  OpBuilder builder(op);
//  builder.create<hir::SubIOp>(op.getLoc(), op.getResult().getType(), op.lhs(),
//                              op.rhs(), IntegerAttr(), Value(), Value());
//  opsToErase.push_back(op);
//  return success();
//}
//
// LogicalResult StandardToHIRPass::convertOp(mlir::MulIOp op) {
//  OpBuilder builder(op);
//  builder.create<hir::MulIOp>(op.getLoc(), op.getResult().getType(), op.lhs(),
//                              op.rhs(), IntegerAttr(), Value(), Value());
//  opsToErase.push_back(op);
//  return success();
//}
//
// LogicalResult StandardToHIRPass::convertOp(mlir::AddFOp op) {
//  OpBuilder builder(op);
//  builder.create<hir::AddFOp>(op.getLoc(), op.getResult().getType(), op.lhs(),
//                              op.rhs(), IntegerAttr(), Value(), Value());
//  opsToErase.push_back(op);
//  return success();
//}
//
// LogicalResult StandardToHIRPass::convertOp(mlir::SubFOp op) {
//  OpBuilder builder(op);
//  builder.create<hir::SubFOp>(op.getLoc(), op.getResult().getType(), op.lhs(),
//                              op.rhs(), IntegerAttr(), Value(), Value());
//  opsToErase.push_back(op);
//  return success();
//}
//
// LogicalResult StandardToHIRPass::convertOp(mlir::MulFOp op) {
//  OpBuilder builder(op);
//  builder.create<hir::MulFOp>(op.getLoc(), op.getResult().getType(), op.lhs(),
//                              op.rhs(), IntegerAttr(), Value(), Value());
//  opsToErase.push_back(op);
//  return success();
//}

LogicalResult StandardToHIRPass::convertOp(mlir::ReturnOp op) {
  OpBuilder builder(op);
  Value c0 = builder
                 .create<mlir::arith::ConstantOp>(
                     op.getLoc(), IndexType::get(builder.getContext()),
                     builder.getIndexAttr(0))
                 .getResult();
  for (size_t i = 0; i < op.getNumOperands(); i++) {
    Value memref = enclosingFuncOp.getFuncBody().front().getArgument(
        mapResultPosToArgumentPos[i]);
    Value operand = op.getOperand(i);
    builder.create<hir::StoreOp>(
        op.getLoc(), operand, memref, c0,
        /*port*/ IntegerAttr(),
        builder.getI64IntegerAttr(mapMemrefToWrDelay[memref]), Value(),
        IntegerAttr());
  }

  builder.create<hir::ReturnOp>(op.getLoc());
  opsToErase.push_back(op);
  return success();
}

/// std-to-hir pass Constructor
std::unique_ptr<mlir::Pass> circt::createStandardToHIRPass() {
  return std::make_unique<StandardToHIRPass>();
}
