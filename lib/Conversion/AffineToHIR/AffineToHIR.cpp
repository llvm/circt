//===----- AffineToHIR.cpp - Affine to HIR Lowering Pass ------------------===//
//
// This pass converts affine+std+memref dialect to HIR.
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/AffineToHIR/AffineToHIR.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;

class AffineFuncLowering : public OpConversionPattern<mlir::FuncOp> {
  using OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::FuncOp functionOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

namespace {
class AffineFuncLoweringImpl {
public:
  AffineFuncLoweringImpl(mlir::FuncOp functionOp,
                         ConversionPatternRewriter &rewriter)
      : originalFunctionOp(functionOp), rewriter(rewriter) {}
  LogicalResult lowerAffineFunction();

private:
  Value getHIRValue(Value);
  LogicalResult lowerOp(mlir::AffineLoadOp);
  LogicalResult lowerOp(mlir::AffineStoreOp);
  LogicalResult lowerOp(mlir::ReturnOp);
  mlir::FuncOp originalFunctionOp;

  ConversionPatternRewriter &rewriter;
  llvm::DenseMap<Value, std::pair<Value, Value>> mapValue2TimeAndOffset;
  llvm::DenseMap<Value, Value> mapValue2LoweredValue;
};
} // namespace

// Helper functions
//------------------------------------------------------------------------------
Value AffineFuncLoweringImpl::getHIRValue(Value original) {
  assert(mapValue2LoweredValue.find(original) != mapValue2LoweredValue.end());
  return mapValue2LoweredValue[original];
}

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

//------------------------------------------------------------------------------

// Op lowering functions.
//------------------------------------------------------------------------------

LogicalResult AffineFuncLoweringImpl::lowerAffineFunction() {
  printf("lowerAffineFunction called.\n");
  FunctionType originalFunctionTy =
      originalFunctionOp.type().dyn_cast<FunctionType>();
  auto originalInputTypes = originalFunctionTy.getInputs();
  auto originalResultTypes = originalFunctionTy.getResults();
  auto *context = rewriter.getContext();
  SmallVector<Type, 4> inputTypes;
  SmallVector<int64_t, 4> inputDelays, resultDelays;
  for (int i = 0; i < (int)originalInputTypes.size(); i++) {
    Type ty = originalInputTypes[i];
    if (auto memrefTy = ty.dyn_cast<mlir::MemRefType>()) {
      ArrayAttr bankDims;
      Attribute bankAttr = originalFunctionOp.getArgAttr(i, "bank");
      if (!bankAttr)
        bankDims = ArrayAttr::get(context, SmallVector<Attribute>());
      else {
        bankDims = bankAttr.dyn_cast<ArrayAttr>();
        if (!bankDims) {
          return originalFunctionOp.emitError(
              "'bank' attribute must be an array of integers.");
        }
        for (auto attr : bankDims) {
          if (!attr.dyn_cast<IntegerAttr>())
            return originalFunctionOp.emitError(
                "'bank' attribute must be an array of integers.");
        }
      }
      SmallVector<hir::DimKind> dimKinds;
      for (size_t i = 0; i < memrefTy.getShape().size(); i++) {
        bool isBankDim = false;
        for (auto dim : bankDims) {
          if (dim.dyn_cast<IntegerAttr>().getInt() ==
              (int64_t)(memrefTy.getShape().size() - i - 1)) {
            isBankDim = true;
            break;
          }
        }
        if (isBankDim)
          dimKinds.push_back(hir::DimKind::BANK);
        else
          dimKinds.push_back(hir::DimKind::ADDR);
      }
      inputTypes.push_back(hir::MemrefType::get(
          context, memrefTy.getShape(), memrefTy.getElementType(), dimKinds));
    } else {
      inputTypes.push_back(ty);
    }
    inputDelays.push_back(0);
  }
  resultDelays.append(originalResultTypes.size(), 0);
  mlir::FunctionType functionTy =
      FunctionType::get(context, inputTypes, originalResultTypes);
  Type funcTy = hir::FuncType::get(context, functionTy,
                                   rewriter.getI64ArrayAttr(inputDelays),
                                   rewriter.getI64ArrayAttr(resultDelays));
  auto funcOp =
      rewriter.create<hir::FuncOp>(originalFunctionOp.getLoc(), functionTy,
                                   originalFunctionOp.sym_name(), funcTy);
  rewriter.inlineRegionBefore(originalFunctionOp.getBody(), funcOp.getBody(),
                              funcOp.end());
  originalFunctionOp.sym_nameAttr(
      StringAttr::get(context, "old_" + originalFunctionOp.sym_name()));

  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    bool loweringFailed = false;
    if (auto op = dyn_cast<mlir::AffineLoadOp>(operation))
      loweringFailed = mlir::failed(lowerOp(op));
    else if (auto op = dyn_cast<mlir::AffineStoreOp>(operation))
      loweringFailed = mlir::failed(lowerOp(op));
    else if (auto op = dyn_cast<mlir::ReturnOp>(operation))
      loweringFailed = mlir::failed(lowerOp(op));

    if (loweringFailed)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  return success();
}

LogicalResult AffineFuncLoweringImpl::lowerOp(mlir::AffineLoadOp op) {
  // Value loweredMemref = getHIRValue(op.memref());
  // if (op->getNumResults() > 0)
  //  return op.emitError("Only one result is supported.");
  // auto res = op->getResult(0);

  // auto *context = rewriter.getContext();
  // auto hirMemrefTy = loweredMemref.getType().dyn_cast<hir::MemrefType>();
  // auto dimKinds = hirMemrefTy.getDimensionKinds();
  // auto shape = hirMemrefTy.getShape();
  // SmallVector<Value, 6> castedIndices;
  // assert(op.indices().size() == shape.size());
  // for (int i = 0; i < (int)op.indices().size(); i++) {
  //  Value originalIdx = op.indices()[i];
  //  if (dimKinds[i] == hir::MemrefType::ADDR) {
  //    auto correctWidthIntTy =
  //        IntegerType::get(context, helper::clog2(shape[i]));
  //    castedIndices.push_back(rewriter
  //                                .create<hir::CastOp>(rewriter.getUnknownLoc(),
  //                                                     correctWidthIntTy,
  //                                                     originalIdx)
  //                                .getResult());
  //  } else {
  //    if (isConstantIndex(originalIdx))
  //      return op.emitError(
  //          "Expected constant of IndexType at index location " +
  //          std::to_string(i) + ".");
  //    castedIndices.push_back(originalIdx);
  //  }
  //}

  // rewriter.create<hir::LoadOp>(op.getLoc(), res.getType(), loweredMemref);
  return success();
}

LogicalResult AffineFuncLoweringImpl::lowerOp(mlir::AffineStoreOp op) {
  return success();
}
LogicalResult AffineFuncLoweringImpl::lowerOp(mlir::ReturnOp op) {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult
AffineFuncLowering::matchAndRewrite(mlir::FuncOp functionOp,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
  AffineFuncLoweringImpl functionRewriter(functionOp, rewriter);
  rewriter.eraseOp(functionOp);
  return functionRewriter.lowerAffineFunction();
}

namespace {
class AffineToHIRLoweringPass
    : public LowerAffineToHIRBase<AffineToHIRLoweringPass> {
public:
  void runOnOperation() override {
    auto op = getOperation();
    ConversionTarget target(getContext());
    target.addLegalDialect<hir::HIRDialect>();
    RewritePatternSet patterns(op.getContext());
    patterns.insert<AffineFuncLowering>(op.getContext());
    if (failed(mlir::applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::createLowerAffineToHIRPass() {
  return std::make_unique<AffineToHIRLoweringPass>();
}
