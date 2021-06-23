//=========- BitwidthReductionPass.cpp - Verify schedule of HIR
// dialect---------===//
//
// This file implements the HIR schedule verifier.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <list>

using namespace llvm;
using namespace circt;
using namespace hir;

namespace {
class TimeInstant {
public:
  TimeInstant() : t(Value()), delay(0) {}
  TimeInstant(Value t, unsigned delay) : t(t), delay(delay) {}
  Value t;
  unsigned delay;
};
/// Checks for out of bound memef access subscripts..
class BitwidthReductionPass
    : public mlir::PassWrapper<BitwidthReductionPass,
                               OperationPass<hir::FuncOp>> {
public:
  void runOnOperation() override;

private:
  bool inspectOp(hir::FuncOp op);
  bool inspectOp(hir::ConstantOp op);
  bool inspectOp(ForOp op);
  bool inspectOp(UnrollForOp op);
  bool inspectOp(LoadOp op);
  bool inspectOp(StoreOp op);
  bool inspectOp(Operation *op);
  bool inspectBody(Block &body);

private:
  unsigned getBitWidth(Value v) {
    if (isIntegerConst(v)) {
      int val = std::abs(getIntegerConstOrError(v));
      if (val > 0)
        return std::ceil(std::log2(val + 1));
      return 1;
    }
    IntegerType integerType = v.getType().dyn_cast<IntegerType>();
    assert(integerType);
    return integerType.getWidth();
  }

  void setIntegerConst(Value v, int constant) {
    mapValueToIntConst[v] = constant;
  }

  int isIntegerConst(Value v) {
    assert(v);
    auto it = mapValueToIntConst.find(v);
    if (it == mapValueToIntConst.end()) {
      return false;
    }
    return true;
  }

  int getIntegerConstOrError(Value v) {
    assert(v);
    auto it = mapValueToIntConst.find(v);
    if (it == mapValueToIntConst.end()) {
      emitError(v.getLoc(), "failed to find integer const.");
      return 0;
    }
    return mapValueToIntConst[v];
  }

private:
  llvm::DenseMap<Value, int> mapValueToIntConst;
  std::vector<Operation *> opsToErase;
};

bool BitwidthReductionPass::inspectOp(FuncOp op) {
  Block &entryBlock = op.getBody().front();
  inspectBody(entryBlock);
  return true;
}

bool BitwidthReductionPass::inspectOp(hir::ConstantOp op) {
  setIntegerConst(op.res(), op.value().dyn_cast<IntegerAttr>().getInt());
  return true;
}

bool BitwidthReductionPass::inspectOp(ForOp op) {
  Value lb = op.lb();
  Value ub = op.ub();
  Value step = op.step();

  unsigned minBitWidth =
      std::max(getBitWidth(lb), std::max(getBitWidth(ub), getBitWidth(step)));

  Block *body = op.getBody();
  BlockArgument newIdx =
      body->addArgument(IntegerType::get(op.getContext(), minBitWidth));
  BlockArgument newTLoop =
      body->addArgument(hir::TimeType::get(op.getContext()));
  body->getArgument(0).replaceAllUsesWith(newIdx);
  body->getArgument(1).replaceAllUsesWith(newTLoop);
  body->eraseArgument(0);
  body->eraseArgument(0);

  inspectBody(op.getLoopBody().front());
  return true;
}

bool BitwidthReductionPass::inspectOp(UnrollForOp op) {
  inspectBody(op.getLoopBody().front());
  return true;
}

bool BitwidthReductionPass::inspectOp(LoadOp op) { return true; }

bool BitwidthReductionPass::inspectOp(StoreOp op) { return true; }

bool BitwidthReductionPass::inspectOp(Operation *inst) {
  if (auto op = dyn_cast<hir::ConstantOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::CallOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::AllocaOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::DelayOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::ForOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::UnrollForOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::ReturnOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::LoadOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::StoreOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::YieldOp>(inst)) {
    return inspectOp(op);
  }
  if (auto op = dyn_cast<hir::TerminatorOp>(inst)) {
    return true;
  }
  return true;
}

bool BitwidthReductionPass::inspectBody(Block &block) {

  // Print the operations within the entity.
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    if (!inspectOp(&(*iter))) {
      return false;
    }
  }
  for (auto *operation : opsToErase) {
    operation->erase();
  }
  return true;
}
} // end anonymous namespace

void BitwidthReductionPass::runOnOperation() { inspectOp(getOperation()); }
namespace circt {
namespace hir {
void registerBitwidthReductionPass() {
  mlir::PassRegistration<BitwidthReductionPass>(
      "hir-reduce-bitwidth", "Reduce bitwidth of integers when safe.");
}
} // namespace hir
} // namespace circt
