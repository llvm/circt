//=========- -------BusFanoutInfo.cpp - Analysis pass for bus fanout-------===//
//
// Calculates fanout info of each bus in a FuncOp.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace mlir;
using namespace hir;

class BusFanoutInfo {
public:
  BusFanoutInfo(Operation *);

private:
  void dispatchOp(Operation *);
  // visit ops that can define a bus.
  void visitOp(hir::FuncOp);
  void visitOp(hir::AllocaOp);
  // visit ops that use a bus.
  void visitOp(hir::SendOp);
  void visitOp(hir::RecvOp);
  void visitOp(hir::SplitOp);
  void visitOp(hir::CallOp);

public:
  llvm::DenseMap<Value, SmallVector<unsigned, 1>> mapBusTensor2UseCount;
  llvm::DenseMap<Value, SmallVector<unsigned, 1>> mapBus2UseCount;
};

BusFanoutInfo::BusFanoutInfo(Operation *op) {
  auto funcOp = dyn_cast<hir::FuncOp>(op);
  assert(funcOp);

  visitOp(funcOp);

  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::AllocaOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::SendOp>(operation))
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
        mapBusTensor2UseCount[arg].append(
            helper::getSizeFromShape(tensorTy.getShape()), 0);
      }
    } else if (auto busTy = arg.getType().dyn_cast<hir::BusType>()) {
      mapBus2UseCount[arg].append(busTy.getElementTypes().size(), 0);
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
      mapBusTensor2UseCount[bus].append(
          helper::getSizeFromShape(tensorTy.getShape()), 0);
    } else if (auto busTy = bus.getType().dyn_cast<hir::BusType>()) {
      mapBus2UseCount[bus].append(busTy.getElementTypes().size(), 0);
    } else {
      assert(false && "We only support bus and tensor of bus.");
    }
  }
}

void BusFanoutInfo::visitOp(hir::SendOp op) {
  Value bus = op.bus();
  auto index = helper::getConstantIntValue(op.index());
  assert(bus.getType().dyn_cast<hir::BusType>());
  assert(mapBus2UseCount.find(bus) != mapBus2UseCount.end());
  mapBus2UseCount[bus][index]++;
}

void BusFanoutInfo::visitOp(hir::RecvOp op) {
  Value bus = op.bus();
  auto index = helper::getConstantIntValue(op.index());
  assert(bus.getType().dyn_cast<hir::BusType>());
  assert(mapBus2UseCount.find(bus) != mapBus2UseCount.end());
  mapBus2UseCount[bus][index]++;
}

void BusFanoutInfo::visitOp(hir::SplitOp op) {
  Value bus = op.bus();
  if (auto busTy = bus.getType().dyn_cast<hir::BusType>()) {
    assert(mapBus2UseCount.find(bus) != mapBus2UseCount.end());
    for (int i = 0; i < (int)mapBus2UseCount[bus].size(); i++) {
      mapBus2UseCount[bus][i]++;
    }
  } else if (auto tensorTy = bus.getType().dyn_cast<TensorType>()) {
    assert(mapBusTensor2UseCount.find(bus) != mapBusTensor2UseCount.end());
    int64_t linearIdx =
        helper::calcLinearIndex(op.indices(), tensorTy.getShape());
    mapBusTensor2UseCount[bus][linearIdx]++;
  } else {
    assert(false && "We only support bus and tensor of bus.");
  }
}
