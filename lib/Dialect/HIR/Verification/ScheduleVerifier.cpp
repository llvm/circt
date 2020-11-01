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

class Schedule {
private:
  llvm::DenseMap<Value, TimeInstant> mapValueToTimeInstant;

public:
  bool insert(Value v, Value t, unsigned delay) {
    assert(t);
    assert(t.getType().isa<TimeType>());
    assert(v);
    assert(!v.getType().isa<hir::WireType>());
    assert(!v.getType().isa<hir::MemrefType>());
    assert(!v.getType().isa<hir::ConstType>());
    if (v != t)
      while (1) {
        auto it = mapValueToTimeInstant.find(t);
        if (it == mapValueToTimeInstant.end()) {
          emitError(t.getLoc(),
                    "Could not find any mapping for this time var.");
          return false;
        }

        TimeInstant instant = it->getSecond();
        if (instant.t == t) {
          if (instant.delay > 0) {
            emitError(t.getLoc(), "Circular mapping found for this time var.");
            return false;
          } else
            break;
        }
        delay += instant.delay;
        t = instant.t;
      }
    TimeInstant v_instant(t, delay);
    mapValueToTimeInstant[v] = v_instant;
  }

  TimeInstant getTimeInstantOrError(Value v) {
    auto it = mapValueToTimeInstant.find(v);
    if (it == mapValueToTimeInstant.end()) {
      emitError(v.getLoc(), "Could not find time instant in schedule.");
      return TimeInstant();
    }
    return it->getSecond();
  }

  bool check(mlir::Location loc, Value v, Value t, unsigned delay) {
    // consts are valid at any time.
    Type v_type = v.getType();
    if (v_type.isa<hir::ConstType>() || v_type.isa<hir::MemrefType>() ||
        v_type.isa<WireType>())
      return true;
    TimeInstant instant = getTimeInstantOrError(v);
    if (!instant.t)
      return false;
    if (instant.t == t && instant.delay == delay)
      return true;

    emitError(loc, "Schedule mismatch!")
        .attachNote(v.getLoc())
        .append("Prior definition here.");

    return false;
  }
};
/// Checks for out of bound memef access subscripts..
class ScheduleVerifier
    : public PassWrapper<ScheduleVerifier, OperationPass<hir::DefOp>> {
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
  bool inspectOp(hir::PopOp op);
  bool inspectOp(Operation *op);
  bool inspectBody(Block &body);

private:
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
  }
  llvm::DenseMap<Value, int> mapValueToIntConst;

private:
  Schedule schedule;
  std::stack<TimeInstant> yieldPoints;
};

bool ScheduleVerifier::inspectOp(DefOp op) {
  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  Value tstart = args.back();
  // Indentity map for root level time vars.
  schedule.insert(tstart, tstart, 0);
  inspectBody(entryBlock);
  return true;
}

bool ScheduleVerifier::inspectOp(hir::ConstantOp op) {
  setIntegerConst(op.res(), op.value().getLimitedValue());
  return true;
}

bool ScheduleVerifier::inspectOp(ForOp op) {
  Value idx = op.getInductionVar();
  Value tloop = op.getIterTimeVar();
  schedule.insert(tloop, tloop, 0);
  schedule.insert(idx, tloop, 0);

  bool ok;
  ok &= schedule.check(op.getLoc(), op.lb(), op.tstart(), 0);
  ok &= schedule.check(op.getLoc(), op.ub(), op.tstart(), 0);
  ok &= schedule.check(op.getLoc(), op.step(), op.tstart(), 0);
  ok &= schedule.check(op.getLoc(), op.tstep(), op.tstart(), 0);
  inspectBody(op.getLoopBody().front());
  return ok;
}

bool ScheduleVerifier::inspectOp(UnrollForOp op) {
  Value tloop = op.getIterTimeVar();
  Value idx = op.getInductionVar();
  Value tstart = op.tstart();
  schedule.insert(tloop, tstart, 0);
  schedule.insert(idx, tloop, 0);
  yieldPoints.push(TimeInstant());
  for (int i = op.lb().getLimitedValue(); i < op.ub().getLimitedValue();
       i += op.step().getLimitedValue()) {
    inspectBody(op.getLoopBody().front());
    TimeInstant yieldPoint = yieldPoints.top();
    schedule.insert(tloop, yieldPoint.t, yieldPoint.delay);
    schedule.insert(idx, yieldPoint.t, yieldPoint.delay);
  }
  return true;
}

bool ScheduleVerifier::inspectOp(MemReadOp op) {
  auto addr = op.addr();
  Value result = op.res();
  Value tstart = op.tstart();
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;

  for (auto addrI : addr) {
    schedule.check(op.getLoc(), addrI, tstart, delay);
  }
  // FIXME: Assume MemReadOp delay is one cycle.
  schedule.insert(result, tstart, delay + 1);
  return true;
}

bool ScheduleVerifier::inspectOp(MemWriteOp op) {
  auto addr = op.addr();
  Value value = op.value();
  Value tstart = op.tstart();
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;

  schedule.check(op.getLoc(), value, tstart, delay);
  for (auto addrI : addr) {
    schedule.check(op.getLoc(), addrI, tstart, delay);
  }
  return true;
}

bool ScheduleVerifier::inspectOp(hir::AddOp op) {
  Value result = op.res();
  Value left = op.left();
  Value right = op.right();
  if (result.getType().isa<hir::ConstType>()) {
    setIntegerConst(result, getIntegerConstOrError(left) +
                                getIntegerConstOrError(right));
  } else {
    TimeInstant instant = schedule.getTimeInstantOrError(left);
    if (!instant.t)
      return false;
    schedule.check(op.getLoc(), right, instant.t, instant.delay);
    schedule.insert(result, instant.t, instant.delay);
  }
  return true;
}

bool ScheduleVerifier::inspectOp(hir::SubtractOp op) {
  Value result = op.res();
  Value left = op.left();
  Value right = op.right();
  if (result.getType().isa<hir::ConstType>()) {
    setIntegerConst(result, getIntegerConstOrError(left) -
                                getIntegerConstOrError(right));
  } else {
    TimeInstant instant = schedule.getTimeInstantOrError(left);
    if (!instant.t)
      return false;
    schedule.check(op.getLoc(), right, instant.t, instant.delay);
    schedule.insert(result, instant.t, instant.delay);
  }
  return true;
}

bool ScheduleVerifier::inspectOp(hir::ReturnOp op) { return true; }

bool ScheduleVerifier::inspectOp(hir::YieldOp op) {
  Value tstart = op.tstart();
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  yieldPoints.top().t = tstart;
  yieldPoints.top().delay = delay;
  return true;
}

bool ScheduleVerifier::inspectOp(hir::WireWriteOp op) {
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  schedule.check(op.getLoc(), op.value(), op.tstart(), delay);
  return true;
}

bool ScheduleVerifier::inspectOp(hir::WireReadOp op) {
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  schedule.insert(op.res(), op.tstart(), delay);
  return true;
}

bool ScheduleVerifier::inspectOp(hir::AllocOp op) { return true; }

bool ScheduleVerifier::inspectOp(hir::DelayOp op) {
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  unsigned latency = getIntegerConstOrError(op.delay());
  schedule.check(op.getLoc(), op.input(), op.tstart(), delay);
  schedule.insert(op.res(), op.tstart(), delay + latency);
  return true;
}

bool ScheduleVerifier::inspectOp(hir::CallOp op) {
  ResultRange results = op.res();
  auto operands = op.operands();
  Value tstart = op.tstart();
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  ArrayAttr input_latency = op.getAttrOfType<ArrayAttr>("input_latency");
  ArrayAttr output_latency = op.getAttrOfType<ArrayAttr>("output_latency");
  assert(input_latency.size() == operands.size());
  assert(output_latency.size() == results.size());
  for (int i = 0; i < operands.size(); i++) {
    auto latency = input_latency[i].cast<IntegerAttr>().getInt();
    schedule.check(op.getLoc(), operands[i], op.tstart(), latency);
    return true;
  }
  for (int i = 0; i < results.size(); i++) {
    auto latency = output_latency[i].cast<IntegerAttr>().getInt();
    schedule.insert(results[i], op.tstart(), latency);
    return true;
  }
}

bool ScheduleVerifier::inspectOp(hir::PopOp op) { return true; }

bool ScheduleVerifier::inspectOp(Operation *inst) {
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
bool ScheduleVerifier::inspectBody(Block &block) {

  // Print the operations within the entity.
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    if (!inspectOp(&(*iter))) {
      return false;
    }
  }
}
} // end anonymous namespace

void ScheduleVerifier::runOnOperation() { inspectOp(getOperation()); }
namespace mlir {
namespace hir {
void registerScheduleVerifier() {
  PassRegistration<ScheduleVerifier>("hir-schedule-verifier",
                                     "Verify schedule in HIR functions.");
}
} // namespace hir
} // namespace mlir
