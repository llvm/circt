//=========- ScheduleVerifier.cpp - Verify schedule of HIR dialect---------===//
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

using namespace mlir;
using namespace hir;
using namespace llvm;

namespace {

/// Checks for out of bound memef access subscripts..
class ScheduleVerifier : public PassWrapper<ScheduleVerifier, FunctionPass> {
public:
  void runOnFunction() override;

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
  bool inspectOp(hir::PopOp op);
  bool inspectOp(Operation *op);
  void inspectBody(Block &body);
};

bool ScheduleVerifier::inspectOp(DefOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::ConstantOp op) { return true; }
bool ScheduleVerifier::inspectOp(ForOp op) { return true; }
bool ScheduleVerifier::inspectOp(UnrollForOp op) { return true; }
bool ScheduleVerifier::inspectOp(MemReadOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::AddOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::SubtractOp op) { return true; }
bool ScheduleVerifier::inspectOp(MemWriteOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::ReturnOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::YieldOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::WireWriteOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::WireReadOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::AllocOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::DelayOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::CallOp op) { return true; }
bool ScheduleVerifier::inspectOp(hir::PopOp op) { return true; }

bool inspectOp(Operation *inst) {
  if (auto op = dyn_cast<hir::ConstantOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::CallOp>(inst)) {
    return inspectOp(op);
  } else if (auto op = dyn_cast<hir::PopOp>(inst)) {
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
    emitError(inst->getLoc(), "Unsupported Operation for verification!");
    return false;
  }
}
void ScheduleVerifier::inspectBody(Block &block) {

  // Print the operations within the entity.
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    if (!inspectOp(&(*iter))) {
      exit(1);
    }
  }
}
} // end anonymous namespace

void ScheduleVerifier::runOnFunction() {
  inspectBody(getFunction().getBody().front());
}

void registerScheduleVerifier() {
  PassRegistration<ScheduleVerifier>("hir-schedule-verifier",
                                     "Verify schedule in HIR functions.");
}
