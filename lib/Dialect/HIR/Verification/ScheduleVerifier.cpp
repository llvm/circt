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
    assert(v.getType().isa<IntegerType>() || v.getType().isa<TimeType>());
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

  bool check(mlir::Location op_loc, mlir::Location def_loc, Value v, Value t,
             unsigned delay, std::string use_loc) {
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
    std::string error;
    if (instant.t != t)
      error = "Schedule error: mismatched time instants in " + use_loc + "!";
    else
      error = "Schedule error: mismatched delay (" +
              std::to_string(instant.delay) + " vs " + std::to_string(delay) +
              ") in " + use_loc + "!";
    emitError(op_loc, error)
        .attachNote(def_loc)
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
    return mapValueToIntConst[v];
  }

  mlir::Location getDefiningLoc(Value v) {
    auto it = mapValueToDefiningLoc.find(v);
    if (it != mapValueToDefiningLoc.end()) {
      return it->getSecond();
    } else {
      return v.getLoc();
    }
  }

  llvm::DenseMap<Value, int> mapValueToIntConst;

  // Used for loc of region parameters such as induction var.
  llvm::DenseMap<Value, mlir::Location> mapValueToDefiningLoc;
  std::vector<Operation *> opsToErase;

private:
  Schedule schedule;
  std::stack<TimeInstant> yieldPoints;
  ArrayAttr output_delays;
  Value tstart;
};

bool ScheduleVerifier::inspectOp(DefOp op) {
  Block &entryBlock = op.getBody().front();
  // args also contains tstart;
  auto args = entryBlock.getArguments();
  Value tstart = args.back();
  this->tstart = tstart;
  auto input_delays = op.input_delays();
  this->output_delays = op.output_delays();
  // Indentity map for root level time vars.
  schedule.insert(tstart, tstart, 0);
  for (int i = 0; i < input_delays.size(); i++) {
    mapValueToDefiningLoc.insert(std::make_pair(args[i], op.getLoc()));
    if (!args[i].getType().isa<IntegerType>())
      continue;
    int delay = input_delays[i].dyn_cast<IntegerAttr>().getInt();
    schedule.insert(args[i], tstart, delay);
  }
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
  unsigned delayValue = getIntegerConstOrError(op.offset());
  assert(delayValue > 0);
  schedule.insert(tloop, tloop, 0);
  schedule.insert(idx, tloop, 0);
  mapValueToDefiningLoc.insert(std::make_pair(idx, op.getLoc()));
  mapValueToDefiningLoc.insert(std::make_pair(tloop, op.getLoc()));
  bool ok;
  ok &= schedule.check(op.getLoc(), getDefiningLoc(op.lb()), op.lb(),
                       op.tstart(), delayValue - 1, "lower bound");
  ok &= schedule.check(op.getLoc(), getDefiningLoc(op.ub()), op.ub(),
                       op.tstart(), delayValue - 1, "upper bound");
  ok &= schedule.check(op.getLoc(), getDefiningLoc(op.step()), op.step(),
                       op.tstart(), delayValue - 1, "step");
  yieldPoints.push(TimeInstant());
  inspectBody(op.getLoopBody().front());
  yieldPoints.pop();
  return ok;
} // namespace

bool ScheduleVerifier::inspectOp(UnrollForOp op) {
  Value tloop = op.getIterTimeVar();
  Value idx = op.getInductionVar();
  Value tstart = op.tstart();
  mapValueToDefiningLoc.insert(std::make_pair(idx, op.getLoc()));
  mapValueToDefiningLoc.insert(std::make_pair(tloop, op.getLoc()));
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
  yieldPoints.pop();
  return true;
}

bool ScheduleVerifier::inspectOp(MemReadOp op) {
  auto addr = op.addr();
  Value result = op.res();
  Value tstart = op.tstart();
  int delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  assert(delay >= 0);

  int c = 1;
  for (auto addrI : addr) {
    schedule.check(op.getLoc(), getDefiningLoc(addrI), addrI, tstart, delay,
                   "address " + std::to_string(c));
    c++;
  }
  // FIXME: Assume MemReadOp delay is one cycle.
  schedule.insert(result, tstart, delay + 1);
  return true;
}

bool ScheduleVerifier::inspectOp(MemWriteOp op) {
  auto addr = op.addr();
  Value value = op.value();
  Value tstart = op.tstart();
  int delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  assert(delay >= 0);
  schedule.check(op.getLoc(), getDefiningLoc(value), value, tstart, delay,
                 "input var");
  int c = 1;
  for (auto addrI : addr) {
    mlir::Location loc_addrI = getDefiningLoc(addrI);
    schedule.check(op.getLoc(), loc_addrI, addrI, tstart, delay,
                   "address " + std::to_string(c));
    c++;
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
    schedule.check(op.getLoc(), getDefiningLoc(right), right, instant.t,
                   instant.delay, "right operand");
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
    schedule.check(op.getLoc(), getDefiningLoc(right), right, instant.t,
                   instant.delay, "right operand");
    schedule.insert(result, instant.t, instant.delay);
  }
  return true;
}

bool ScheduleVerifier::inspectOp(hir::ReturnOp op) {
  auto operands = op.operands();
  for (int i = 0; i < operands.size(); i++) {
    Value operand = operands[i];
    int delay = this->output_delays[i].dyn_cast<IntegerAttr>().getInt();
    schedule.check(op.getLoc(), getDefiningLoc(operand), operand, this->tstart,
                   delay, "return operand " + std::to_string(i + 1));
  }
  return true;
}

bool ScheduleVerifier::inspectOp(hir::YieldOp op) {
  Value tstart = op.tstart();
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  yieldPoints.top().t = tstart;
  yieldPoints.top().delay = delay;
  return true;
}

bool ScheduleVerifier::inspectOp(hir::WireWriteOp op) {
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  schedule.check(op.getLoc(), getDefiningLoc(op.value()), op.value(),
                 op.tstart(), delay, "input var");
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
  schedule.check(op.getLoc(), getDefiningLoc(op.input()), op.input(),
                 op.tstart(), delay, "input var");
  schedule.insert(op.res(), op.tstart(), delay + latency);
  return true;
}

bool ScheduleVerifier::inspectOp(hir::CallOp op) {
  ResultRange results = op.res();
  auto operands = op.operands();
  Value tstart = op.tstart();
  unsigned delay = op.offset() ? getIntegerConstOrError(op.offset()) : 0;
  ArrayAttr input_delays = op.getAttrOfType<ArrayAttr>("input_delays");
  ArrayAttr output_delays = op.getAttrOfType<ArrayAttr>("output_delays");
  assert(input_delays.size() == operands.size());
  assert(output_delays.size() == results.size());
  for (int i = 0; i < operands.size(); i++) {
    auto delay = input_delays[i].cast<IntegerAttr>().getInt();
    schedule.check(op.getLoc(), getDefiningLoc(operands[i]), operands[i],
                   op.tstart(), delay, "operand " + std::to_string(i + 1));
  }
  for (int i = 0; i < results.size(); i++) {
    auto delay = output_delays[i].cast<IntegerAttr>().getInt();
    schedule.insert(results[i], op.tstart(), delay);
  }
  return true;
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
  for (auto operation : opsToErase) {
    operation->erase();
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
