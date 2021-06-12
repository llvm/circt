//=========- -------BusFanoutInfo.cpp - Analysis pass for bus fanout-------===//
//
// Calculates fanout info of each bus in a FuncOp.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/helper.h"

using namespace mlir;
using namespace hir;

class BusFanoutInfo {
public:
  BusFanoutInfo(Operation *);

private:
  void dispatchOp(Operation *);
  // visit ops that can define a bus.
  void visitDef(hir::AllocaOp);
  void visitDef(hir::FuncOp);
  // visit ops that use a bus.
  void visitUse(hir::SendOp);
  void visitUse(hir::RecvOp);
  void visitUse(hir::SplitOp);

public:
  llvm::DenseMap<Value, SmallVector<unsigned, 1>> mapBusUseCount;
};

BusFanoutInfo::BusFanoutInfo(Operation *op) {
  auto funcOp = dyn_cast<hir::FuncOp>(op);
  assert(funcOp);

  visitDef(funcOp);

  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::AllocaOp>(operation))
      visitDef(op);
    return WalkResult::advance();
  });
  assert(!result.wasInterrupted());
}

void BusFanoutInfo::visitDef(hir::AllocaOp op) {
  std::string moduleAttr =
      op.moduleAttr().dyn_cast<StringAttr>().getValue().str();
  if (moduleAttr != "bus")
    return;

  // initialize the map with a vector of usage count filled with zeros.
  auto buses = op.getResults();
  for (Value bus : buses) {
    mapBusUseCount[bus] = SmallVector<unsigned, 1>({});
    if (auto tensorTy = bus.getType().dyn_cast<TensorType>()) {
      assert(tensorTy.getElementType().dyn_cast<hir::BusType>());
      mapBusUseCount[bus].append(helper::getSizeFromShape(tensorTy.getShape()),
                                 0);
    } else {
      Type busTy = bus.getType().dyn_cast<hir::BusType>();
      assert(busTy); // only support bus and tensor<bus>.
      mapBusUseCount[bus].push_back(0);
    }
  }

  // update the map with the use count.
  for (Value bus : buses) {
    mapBusUseCount[bus] = SmallVector<unsigned, 1>({});
    for (auto &use : bus.getUses()) {
      Operation *userOperation = use.getOwner();
      if (auto sendOp = dyn_cast<hir::SendOp>(userOperation)) {
        visitUse(sendOp);
      } else {
        assert(false && "unsupported op use uselist of bus");
      }
    }
  }
}

void visitUse(hir::SendOp op) {}
