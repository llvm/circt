//===----- HIRToHW.cpp - HIR To HW Conversion Pass-------*-C++-*-===//
//
// This pass converts HIR to HW, Comb and SV dialect.
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "HIRToHWUtils.h"
#include "circt/Conversion/HIRToHW/HIRToHW.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>
using namespace circt;

class HIRToHWPass : public HIRToHWBase<HIRToHWPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitRegion(mlir::Region &);
  LogicalResult visitOperation(Operation *);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(mlir::ConstantOp op);

private:
  OpBuilder *builder;
};

LogicalResult HIRToHWPass::visitOp(mlir::ConstantOp op) {

  if (op.getType().isa<mlir::IndexType>())
    return success();
  if (!op.getType().isa<mlir::IntegerType>())
    return op.emitError(
        "hir-to-hw pass only supports IntegerType/IndexType constants.");

  builder->create<hw::ConstantOp>(op.getLoc(),
                                  op.value().dyn_cast<IntegerAttr>());
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::FuncOp op) {
  builder = new OpBuilder(op);
  builder->setInsertionPoint(op);
  auto portInfoList = getHWModulePortInfoList(
      *builder, op.getFuncType(), op.getInputNames(), op.getResultNames());
  auto name = builder->getStringAttr("hw_" + op.getNameAttr().getValue().str());
  auto hwModuleOp =
      builder->create<hw::HWModuleOp>(op.getLoc(), name, portInfoList);
  delete (builder);
  builder->setInsertionPointToStart(hwModuleOp.getBodyBlock());
  return visitRegion(op.getFuncBody());
}

LogicalResult HIRToHWPass::visitRegion(mlir::Region &region) {
  Block &bb = *region.begin();
  for (Operation &operation : bb)
    if (failed(visitOperation(&operation)))
      return failure();
  return success();
}

LogicalResult HIRToHWPass::visitOperation(Operation *operation) {
  if (auto op = dyn_cast<hir::FuncOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::ConstantOp>(operation))
    return visitOp(op);

  // operation->emitRemark() << "Unsupported operation for hir-to-hw pass.";
  return success();
}
void HIRToHWPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  WalkResult result = moduleOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::FuncOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}
/// hir-to-hw pass Constructor
std::unique_ptr<mlir::Pass> circt::createHIRToHWPass() {
  return std::make_unique<HIRToHWPass>();
}
