//===---ConstantPropAnalysis.cpp - Analysis pass for constant propagation---==//
//
// Calculates constant value for each !hir.const ssa var.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/helper.h"

using namespace mlir;
using namespace hir;

class ConstantValueInfo {
public:
  ConstantValueInfo(Operation *);

private:
  void dispatchOp(Operation *);
  // visit ops that use a bus.
  void visitOp(hir::ConstantOp);
  void visitOp(hir::AddOp);
  void visitOp(hir::SubtractOp);

public:
  llvm::DenseMap<Value, int64_t> mapValueToConstInt;
};

ConstantValueInfo::ConstantValueInfo(Operation *op) {
  auto funcOp = dyn_cast<hir::FuncOp>(op);
  assert(funcOp);

  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::ConstantOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::AddOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::SubtractOp>(operation))
      visitOp(op);
    return WalkResult::advance();
  });
  assert(!result.wasInterrupted());
}

void ConstantValueInfo::visitOp(ConstantOp op) {
  Attribute value = op.value();
  IntegerAttr valueInt = value.dyn_cast<IntegerAttr>();
  if (valueInt)
    mapValueToConstInt[op.getResult()] = valueInt.getInt();
}

void ConstantValueInfo::visitOp(AddOp) {}
