//===----- AffineToHIR.cpp - Affine to HIR Lowering Pass ------------------===//
//
// This pass converts affine+std+memref dialect to HIR.
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/StandardToHIR/StandardToHIR.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>
using namespace circt;

namespace {
class ConvertStandardToHIRPass
    : public ConvertStandardToHIRBase<ConvertStandardToHIRPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (failed(lowerRegion(moduleOp.body(), SmallVector<Type>()))) {
      signalPassFailure();
      return;
    }
    for (auto *op : opsToErase)
      op->erase();
  }

private:
  LogicalResult lowerRegion(Region &region, ArrayRef<Type> blockArgumentTypes) {
    if (region.getBlocks().size() > 1)
      region.getParentOp()->emitError(
          "Standard to HIR lowering only supports one block per region.");

    // Replace the old arguments in the body with new ones.
    auto &body = region.front();

    if (body.getNumArguments() != blockArgumentTypes.size()) {
      return region.getParentOp()->emitError()
             << "original numArguments(" << body.getNumArguments()
             << ") != blockArgumentTypes.size()(" << blockArgumentTypes.size()
             << ")";
    }
    for (size_t i = 0; i < body.getNumArguments(); i++) {
      Value replacedArg = body.insertArgument(i, blockArgumentTypes[i]);
      body.getArgument(i + 1).replaceAllUsesWith(replacedArg);
      body.eraseArgument(i + 1);
    }
    // body always has one extra argument for tstart.
    body.addArgument(hir::TimeType::get(region.getContext()));

    // lower the ops in the region.
    for (Operation &operation : body) {
      if (auto op = dyn_cast<mlir::FuncOp>(operation)) {
        if (failed(lowerOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::scf::ForOp>(operation)) {
        if (failed(lowerOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::memref::LoadOp>(operation)) {
        if (failed(lowerOp(op)))
          return failure();
      }
    }
    return success();
  }

private:
  LogicalResult lowerOp(mlir::FuncOp);
  LogicalResult lowerOp(mlir::scf::ForOp);
  LogicalResult lowerOp(mlir::memref::LoadOp);
  SmallVector<Operation *> opsToErase;
};
} // namespace

// Helper functions
//------------------------------------------------------------------------------

bool isConstantIndex(Value v) {
  auto constantIndexOp = dyn_cast<mlir::ConstantIndexOp>(v.getDefiningOp());
  if (!constantIndexOp)
    return false;
  return true;
}

int64_t getConstantIndex(Value v) {
  auto constantIndexOp = dyn_cast<mlir::ConstantIndexOp>(v.getDefiningOp());
  assert(constantIndexOp);
  return constantIndexOp.getValue();
}

ArrayAttr getRegWritePort(OpBuilder &builder) {
  SmallVector<Attribute> ports;
  ports.push_back(helper::getDictionaryAttr(
      builder, "wr_latency", helper::getIntegerAttr(builder.getContext(), 1)));
  return ArrayAttr::get(builder.getContext(), ports);
}

ArrayAttr getDefaultMemrefPorts(OpBuilder &builder) {
  SmallVector<Attribute> ports;
  ports.push_back(helper::getDictionaryAttr(
      builder, "rd_latency", helper::getIntegerAttr(builder.getContext(), 1)));
  ports.push_back(helper::getDictionaryAttr(
      builder, "wr_latency", helper::getIntegerAttr(builder.getContext(), 1)));
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

//------------------------------------------------------------------------------

// Op lowering functions.
//------------------------------------------------------------------------------

LogicalResult ConvertStandardToHIRPass::lowerOp(mlir::FuncOp op) {
  FunctionType originalFunctionTy = op.type().dyn_cast<FunctionType>();
  auto originalInputTypes = originalFunctionTy.getInputs();
  auto originalResultTypes = originalFunctionTy.getResults();
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
    } else if (helper::isBuiltinType(ty)) {
      inputTypes.push_back(ty);
      inputAttrs.push_back(helper::getDictionaryAttr(
          builder, "hir.delay", helper::getIntegerAttr(context, 0)));
    } else
      return op.emitError(
          "Only builtin types and memrefs are supported in input types.");
  }

  // Insert result types into inputs as a reg with write only port.
  for (int i = 0; i < (int)originalResultTypes.size(); i++) {
    Type ty = originalResultTypes[i];
    if (helper::isBuiltinType(ty)) {
      auto resultReg =
          hir::MemrefType::get(context, SmallVector<int64_t>({1}), ty,
                               SmallVector<hir::DimKind>({hir::DimKind::BANK}));

      inputTypes.push_back(resultReg);
      inputAttrs.push_back(helper::getDictionaryAttr(
          builder, "hir.memref.ports", getRegWritePort(builder)));
      op.getBody().front().insertArgument(i, inputTypes.back());
    } else
      return op.emitError("Only builtin types are supported in return types.");
  }
  // Insert tstart as the last argument to the loop region.

  auto funcTy =
      hir::FuncType::get(context, inputTypes, inputAttrs, originalResultTypes,
                         SmallVector<DictionaryAttr, 4>({}));
  auto funcOp = builder.create<hir::FuncOp>(
      op.getLoc(), funcTy.getFunctionType(), op.sym_name(), funcTy);
  BlockAndValueMapping mapper;

  op.getBody().cloneInto(&funcOp.getBody(), mapper);
  assert(inputTypes.size() == funcOp.getBody().front().getNumArguments());

  // Replace the old arguments in the body with new ones.
  for (size_t i = 0; i < inputTypes.size(); i++) {
    auto &body = funcOp.getBody().front();

    Value replacedArg = body.insertArgument(i, inputTypes[i]);
    body.getArgument(i + 1).replaceAllUsesWith(replacedArg);
    body.eraseArgument(i + 1);
  }

  opsToErase.push_back(op);
  return lowerRegion(funcOp.getBody(), inputTypes);
}

LogicalResult ConvertStandardToHIRPass::lowerOp(mlir::scf::ForOp op) {
  OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  Value lb = op.lowerBound();
  Value ub = op.upperBound();
  Value step = op.step();
  assert(lb);
  assert(ub);
  assert(step);
  auto forOp =
      builder.create<hir::ForOp>(op.getLoc(), lb, ub, step, Value(), Value(),
                                 IndexType::get(builder.getContext()));
  // forOp.getLoopBody().push_back(new Block);
  // forOp.getLoopBody().front().addArgument(IndexType::get(builder.getContext()));
  BlockAndValueMapping mapper;
  op.getLoopBody().cloneInto(&forOp.getLoopBody(), mapper);

  // tfinish is returned in addition to any other outputs.
  assert(forOp.getNumResults() == 1 + op.getNumResults());
  for (size_t i = 0; i < op.getNumResults(); i++) {
    op.getResult(i).replaceAllUsesWith(forOp.getResult(i + 1));
  }

  opsToErase.push_back(op);

  return lowerRegion(forOp.getLoopBody(),
                     SmallVector<Type>({IndexType::get(builder.getContext())}));
}

LogicalResult ConvertStandardToHIRPass::lowerOp(mlir::memref::LoadOp op) {
  OpBuilder builder(op);
  auto memref = op.memref();
  if (op->getNumResults() > 1)
    return op.emitError("Only one result is supported.");
  auto res = op->getResult(0);

  auto *context = builder.getContext();

  // The memref's type is already fixed while lowering the enclosing FunctionOp
  // to hir::FuncOp.
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
              .create<mlir::IndexCastOp>(builder.getUnknownLoc(),
                                         correctWidthIntTy, originalIdx)
              .getResult());
    } else {
      castedIndices.push_back(originalIdx);
    }
  }

  auto loadOp = builder.create<hir::LoadOp>(
      op.getLoc(), res.getType(), memref, castedIndices,
      builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1), Value(),
      Value());
  op->replaceAllUsesWith(loadOp);
  opsToErase.push_back(op);
  return success();
}

LogicalResult builderReturnOp(mlir::ReturnOp op, PatternRewriter &builder) {
  builder.eraseOp(op);
  return success();
}

/// lower-to-hir pass Constructor
std::unique_ptr<mlir::Pass> circt::createConvertStandardToHIRPass() {
  return std::make_unique<ConvertStandardToHIRPass>();
}
