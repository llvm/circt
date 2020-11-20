//=========- MemrefTrace.cpp - Verify schedule of HIR
// dialect---------===//
//
// This file implements the HIR schedule verifier.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "circt/Dialect/HIR/Verification/SheduleVerifier.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
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

using namespace mlir;
using namespace hir;
using namespace llvm;

namespace {
class TimeInstant {
public:
  TimeInstant() : t(Value()), delay(0) {}
  TimeInstant(Value t, unsigned delay) : t(t), delay(delay) {}
  Value t;
  unsigned delay;
};

class Tensor {
public:
  Tensor(llvm::ArrayRef<unsigned> dims) {
    unsigned size = 1;
    for (auto dim : dims) {
      size *= dim;
    }
    store.resize(size, 0);
    for (auto dim : dims) {
      this->dims.push_back(dim);
    }
  }

  unsigned &at(llvm::ArrayRef<unsigned> indices) {
    unsigned lin_idx = indices[indices.size() - 1];
    for (int i = indices.size() - 2; i >= 0; i++) {
      lin_idx *= indices[i] * dims[i + 1];
    }
    return store[lin_idx];
  }

private:
  llvm::SmallVector<unsigned, 4> dims;
  std::vector<unsigned> store;
};

/// Checks for out of bound memef access subscripts..
class MemrefTrace : public PassWrapper<MemrefTrace, OperationPass<hir::DefOp>> {
public:
  void runOnOperation() override;

private:
  bool inspectOp(DefOp op);
  bool inspectOp(hir::ConstantOp op);
  bool inspectOp(ForOp op);
  bool inspectOp(UnrollForOp op);
  bool inspectOp(MemReadOp op);
  bool inspectOp(hir::AddOp op);
  bool inspectOp(hir::SubtractOp op);
  bool inspectOp(MemWriteOp op);
  bool inspectOp(hir::ReturnOp op);
  bool inspectOp(hir::YieldOp op);
  bool inspectOp(hir::WireWriteOp op);
  bool inspectOp(hir::WireReadOp op);
  bool inspectOp(hir::AllocOp op);
  bool inspectOp(hir::DelayOp op);
  bool inspectOp(hir::CallOp op);
  bool inspectOp(Operation *op);
  bool inspectBody(Block &body);

private:
  unsigned getBitWidth(Value v) {
    if (isIntegerConst(v)) {
      int val = std::abs(getIntegerConstOrError(v));
      if (val > 0)
        return std::ceil(std::log2(val + 1));
      else
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
  llvm::DenseMap<Value, Tensor *> mapMemrefToTensor;
  llvm::DenseMap<Value, int> mapValueToIntConst;
  std::list<Tensor> memrefTensors;
};

bool MemrefTrace::inspectOp(DefOp op) {
  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  for (auto arg : args) {
    if (arg.isa<hir::MemrefType>()) {
      memrefTensors.push_back(
          Tensor(arg.getType().dyn_cast<MemrefType>().getShape()));

      mapMemrefToTensor[arg] = &memrefTensors.back();
    }
  }
  inspectBody(entryBlock);
  return true;
}

bool MemrefTrace::inspectOp(hir::ConstantOp op) {
  setIntegerConst(op.res(), op.value().getLimitedValue());
  return true;
}

bool MemrefTrace::inspectOp(ForOp op) {
  Value lb = op.lb();
  Value ub = op.ub();
  Value step = op.step();

  unsigned min_bitwidth =
      std::max(getBitWidth(lb), std::max(getBitWidth(ub), getBitWidth(step)));

  Block *body = op.getBody();
  BlockArgument new_idx =
      body->addArgument(IntegerType::get(min_bitwidth, op.getContext()));
  BlockArgument new_tloop =
      body->addArgument(hir::TimeType::get(op.getContext()));
  body->getArgument(0).replaceAllUsesWith(new_idx);
  body->getArgument(1).replaceAllUsesWith(new_tloop);
  body->eraseArgument(0);
  body->eraseArgument(0);

  inspectBody(op.getLoopBody().front());
  return true;
}

bool MemrefTrace::inspectOp(UnrollForOp op) {
  inspectBody(op.getLoopBody().front());
  return true;
}

bool MemrefTrace::inspectOp(MemReadOp op) { return true; }

bool MemrefTrace::inspectOp(MemWriteOp op) { return true; }

bool MemrefTrace::inspectOp(hir::AddOp op) {
  Value result = op.res();
  Value left = op.left();
  Value right = op.right();

  return true;
}

bool MemrefTrace::inspectOp(hir::SubtractOp op) {
  Value result = op.res();
  Value left = op.left();
  Value right = op.right();

  return true;
}

bool MemrefTrace::inspectOp(hir::ReturnOp op) { return true; }

bool MemrefTrace::inspectOp(hir::YieldOp op) { return true; }

bool MemrefTrace::inspectOp(hir::WireWriteOp op) { return true; }

bool MemrefTrace::inspectOp(hir::WireReadOp op) { return true; }

bool MemrefTrace::inspectOp(hir::AllocOp op) {
  auto results = op.res();
  memrefTensors.push_back(
      Tensor(results[0].getType().dyn_cast<MemrefType>().getShape()));
  for (auto result : results) {
    assert(result.isa<hir::MemrefType>());
    mapMemrefToTensor[result] = &memrefTensors.back();
  }
  return true;
}

bool MemrefTrace::inspectOp(hir::DelayOp op) { return true; }

bool MemrefTrace::inspectOp(hir::CallOp op) { return true; }

bool MemrefTrace::inspectOp(Operation *inst) {
  if (auto op = dyn_cast<hir::ConstantOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::CallOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::AllocOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::DelayOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::ForOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::UnrollForOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::ReturnOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::MemReadOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::MemWriteOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::WireReadOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::WireWriteOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::AddOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::SubtractOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::YieldOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::TerminatorOp>(inst)) {
    // Do nothing.
  } else {
    emitError(inst->getLoc(), "Unsupported Operation for reschedule!");
    return false;
  }
}

bool MemrefTrace::inspectBody(Block &block) {

  // Print the operations within the entity.
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    if (!inspectOp(&(*iter))) {
      return false;
    }
  }
}
} // end anonymous namespace

void MemrefTrace::runOnOperation() { inspectOp(getOperation()); }
namespace mlir {
namespace hir {
void registerMemrefTrace() {
  PassRegistration<MemrefTrace>("hir-reschedule",
                                "Reschedule to improve performance.");
}
} // namespace hir
} // namespace mlir
