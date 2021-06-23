//=========- OpLoweringPass.cpp - Lower all ops---===//
//
// This file implements lowering pass to lower ops for codegen.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/Analysis/FanoutAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;
namespace {

class OpLoweringPass : public hir::OpLoweringBase<OpLoweringPass> {
public:
  void runOnOperation() override;
  void dispatchOp(Operation *);
  void lowerOp(hir::FuncOp);
  void lowerOp(hir::AllocaOp);

private:
  llvm::DenseMap<Value, Value> mapMemrefRdAddrSend;
  llvm::DenseMap<Value, Value> mapMemrefRdDataRecv;
  llvm::DenseMap<Value, Value> mapMemrefWrSend;
  std::unique_ptr<BusFanoutInfo> busFanoutInfo;
};
} // end anonymous namespace

void OpLoweringPass::lowerOp(hir::AllocaOp op) {
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  auto results = op.getResults();
  if (op.moduleAttr().dyn_cast<StringAttr>().getValue().str() != "bus")
    return;
  for (auto res : results) {
    if (auto busTy = res.getType().dyn_cast<BusType>()) {
      SmallVector<Operation *, 4> uses = busFanoutInfo->mapBus2Uses[res];
      if (uses.size() <= 1)
        continue;
      auto mergeFunc =
          busTy.getProto().get("merge").dyn_cast<StringAttr>().getValue();
      auto resTy = RankedTensorType::get(
          SmallVector<int64_t>({(long)(uses.size())}), busTy);

      auto functionTy =
          FunctionType::get(context, SmallVector<Type, 2>({resTy, busTy}),
                            SmallVector<Type, 1>({}));
      auto zeroAttr = helper::getIntegerAttr(context, 64, 0);
      auto funcTy = hir::FuncType::get(
          context, functionTy,
          ArrayAttr::get(
              context, SmallVector<Attribute>({zeroAttr, zeroAttr, zeroAttr})),
          ArrayAttr::get(context, SmallVector<Attribute>({})));

      Value tstart = getOperation().getBody().front().getArguments().back();
      builder.create<hir::AllocaOp>(op.getLoc(), SmallVector<Type, 2>({}),
                                    StringAttr::get(context, "bus"));
      builder.create<hir::CallOp>(
          op.getLoc(), FlatSymbolRefAttr::get(context, mergeFunc), funcTy,
          Value(), SmallVector<Value, 2>({}), tstart, Value());
    }
  }
}

void OpLoweringPass::dispatchOp(Operation *operation) {
  if (auto op = dyn_cast<hir::FuncOp>(operation))
    lowerOp(op);
  if (auto op = dyn_cast<hir::AllocaOp>(operation))
    lowerOp(op);
}

void OpLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  busFanoutInfo = std::make_unique<BusFanoutInfo>(funcOp);
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    dispatchOp(operation);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createOpLoweringPass() {
  return std::make_unique<OpLoweringPass>();
}
} // namespace hir
} // namespace circt
