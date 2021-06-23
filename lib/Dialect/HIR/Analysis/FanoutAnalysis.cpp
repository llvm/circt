//=========- -------BusFanoutInfo.cpp - Analysis pass for bus fanout-------===//
//
// Calculates fanout info of each bus in a FuncOp.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/Analysis/FanoutAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace mlir;
using namespace hir;

BusFanoutInfo::BusFanoutInfo(Operation *op) {
  auto funcOp = dyn_cast<hir::FuncOp>(op);
  assert(funcOp);

  visitOp(funcOp);

  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::AllocaOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::TensorExtractOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::BusUnpackOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::CallOp>(operation))
      visitOp(op);
    return WalkResult::advance();
  });
  assert(!result.wasInterrupted());
}

void BusFanoutInfo::visitOp(hir::FuncOp op) {
  auto &entryBlock = op.getBody().front();
  auto arguments = entryBlock.getArguments();
  for (auto arg : arguments) {
    if (auto tensorTy = arg.getType().dyn_cast<TensorType>()) {
      if (auto busTy = arg.getType().dyn_cast<hir::BusType>()) {

        mapBusTensor2Uses[arg].append(
            helper::getSizeFromShape(tensorTy.getShape()),
            SmallVector<Operation *, 1>({}));
      }
    } else if (auto busTy = arg.getType().dyn_cast<hir::BusType>()) {
      mapBus2Uses[arg] = SmallVector<Operation *, 4>({});
    }
  }
}

void BusFanoutInfo::visitOp(hir::AllocaOp op) {
  std::string moduleAttr =
      op.moduleAttr().dyn_cast<StringAttr>().getValue().str();
  if (moduleAttr != "bus")
    return;

  // initialize the map with a vector of usage count filled with zeros.
  auto buses = op.getResults();
  for (Value bus : buses) {
    if (auto tensorTy = bus.getType().dyn_cast<TensorType>()) {
      Type busTy = tensorTy.getElementType().dyn_cast<hir::BusType>();
      assert(busTy);
      mapBusTensor2Uses[bus].append(
          helper::getSizeFromShape(tensorTy.getShape()),
          SmallVector<Operation *, 1>({}));
    } else if (auto busTy = bus.getType().dyn_cast<hir::BusType>()) {
      mapBus2Uses[bus] = SmallVector<Operation *, 4>({});
    } else {
      assert(false && "We only support bus and tensor of bus.");
    }
  }
}

void BusFanoutInfo::visitOp(hir::TensorExtractOp op) {
  Value bus = op.bus();
  auto tensorTy = bus.getType().dyn_cast<TensorType>();
  assert(mapBusTensor2Uses.find(bus) != mapBusTensor2Uses.end());
  int64_t linearIdx =
      helper::calcLinearIndex(op.indices(), tensorTy.getShape());
  mapBusTensor2Uses[bus][linearIdx].push_back(op);
}

void BusFanoutInfo::visitOp(hir::BusUnpackOp op) {
  Value bus = op.bus();
  assert(mapBus2Uses.find(bus) != mapBus2Uses.end());
  mapBus2Uses[bus].push_back(op);
}
