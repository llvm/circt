//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/helper.h"

using namespace mlir;
using namespace hir;
namespace {

class MemrefLoweringPass : public hir::MemrefLoweringBase<MemrefLoweringPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

void inspectOp(hir::FuncOp op) {}
void inspectOp(hir::ConstantOp op) {}
void inspectOp(hir::ForOp op) {}
void inspectOp(hir::UnrollForOp op) {}
void inspectOp(hir::AddOp op) {}
void inspectOp(hir::SubtractOp op) {}
void inspectOp(hir::StoreOp op) {}
void inspectOp(hir::ReturnOp op) {}
void inspectOp(hir::YieldOp op) {}
void inspectOp(hir::SendOp op) {}
void inspectOp(hir::RecvOp op) {}
void inspectOp(hir::DelayOp op) {}
void inspectOp(hir::CallOp op) {}

void inspectOp(hir::AllocaOp op) {
  std::string moduleAttr =
      op.moduleAttr().dyn_cast<StringAttr>().getValue().str();
  if (moduleAttr != "bram" && moduleAttr != "reg")
    return;

  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  auto results = op.res();

  Type tyI1 = helper::getIntegerType(context, 1);

  DictionaryAttr protoValid = DictionaryAttr::get(
      context, SmallVector<NamedAttribute>({builder.getNamedAttr(
                   "join", StringAttr::get(context, "join_valid"))}));
  DictionaryAttr protoEmpty =
      DictionaryAttr::get(context, SmallVector<NamedAttribute>({}));

  for (auto res : results) {
    hir::MemrefType ty = res.getType().dyn_cast<hir::MemrefType>();
    assert(ty);
    int dataWidth = helper::getBitWidth(ty.getElementType());
    int depth = ty.getDepth();
    int numBanks = ty.getNumBanks();
    auto port = ty.getPort();
    int addrWidth = helper::clog2(depth);
    Type tyIAddr = helper::getIntegerType(context, addrWidth);
    Type tyIData = helper::getIntegerType(context, dataWidth);
    Type tyIAddrAndData =
        TupleType::get(context, SmallVector<Type>({tyIAddr, tyIData}));

    Type tyRdAddrSend =
        hir::BusType::get(context, SmallVector<Type>({tyI1, tyIAddr}),
                          SmallVector<PortKind>({wr, wr}), protoValid);
    if (numBanks > 1) {
      tyRdAddrSend = RankedTensorType::get(ty.getBankShape(), tyRdAddrSend);
    }

    Type tyRdAddrRecv =
        hir::BusType::get(context, SmallVector<Type>({tyI1, tyIAddr}),
                          SmallVector<PortKind>({rd, rd}), protoEmpty);
    if (numBanks > 1) {
      tyRdAddrRecv = RankedTensorType::get(ty.getBankShape(), tyRdAddrRecv);
    }
    Type tyRdDataSend =
        hir::BusType::get(context, SmallVector<Type>({tyIData}),
                          SmallVector<PortKind>({wr}), protoEmpty);
    if (numBanks > 1) {
      tyRdDataSend = RankedTensorType::get(ty.getBankShape(), tyRdDataSend);
    }
    Type tyRdDataRecv =
        hir::BusType::get(context, SmallVector<Type>({tyIData}),
                          SmallVector<PortKind>({rd}), protoEmpty);
    if (numBanks > 1) {
      tyRdDataRecv = RankedTensorType::get(ty.getBankShape(), tyRdDataRecv);
    }
    Type tyWrSend =
        hir::BusType::get(context, SmallVector<Type>({tyI1, tyIAddrAndData}),
                          SmallVector<PortKind>({wr, wr}), protoValid);
    if (numBanks > 1) {
      tyWrSend = RankedTensorType::get(ty.getBankShape(), tyWrSend);
    }
    Type tyWrRecv =
        hir::BusType::get(context, SmallVector<Type>({tyI1, tyIAddrAndData}),
                          SmallVector<PortKind>({rd, rd}), protoEmpty);
    if (numBanks > 1) {
      tyWrRecv = RankedTensorType::get(ty.getBankShape(), tyWrRecv);
    }

    if (port == rd) {
      auto rdAddrBuses = builder.create<hir::AllocaOp>(
          op.getLoc(), SmallVector<Type>({tyRdAddrSend, tyRdAddrRecv}),
          builder.getStringAttr("bus"));
      auto rdDataBuses = builder.create<hir::AllocaOp>(
          op.getLoc(), SmallVector<Type>({tyRdDataSend, tyRdDataRecv}),
          builder.getStringAttr("bus"));
    } else if (port == wr) {
      auto wrBuses = builder.create<hir::AllocaOp>(
          op.getLoc(), SmallVector<Type>({tyWrSend, tyWrRecv}),
          builder.getStringAttr("bus"));
    } else {
      assert(false && "rw is not supported yet");
    }
  }
}

void processLoadOp(hir::LoadOp op) {
  // mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  // builder.setInsertionPoint(op);
  // hir::DelayOp newDelayOp = builder.create<hir::DelayOp>(
  //    op.getLoc(), op.tstart().getType(), op.tstart(), op.offset(),
  //    op.tstart(), mlir::Value());
  // hir::LoadOp newLoadOp =
  //    builder.create<hir::LoadOp>(op.getLoc(), op.res().getType(), op.mem(),
  //                                op.addr(), newDelayOp, mlir::Value());
  // op.replaceAllUsesWith(newLoadOp.getOperation());
  // op.getOperation()->dropAllReferences();
  // op.getOperation()->dropAllUses();
  // op.getOperation()->erase();
}

void MemrefLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult result = funcOp.walk([](Operation *operation) -> WalkResult {
    if (hir::AllocaOp op = dyn_cast<hir::AllocaOp>(operation))
      inspectOp(op);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

namespace mlir {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createMemrefLoweringPass() {
  return std::make_unique<MemrefLoweringPass>();
}
} // namespace hir
} // namespace mlir
