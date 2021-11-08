//=========- ScheduleVerifier.cpp - Verify schedule of HIR dialect---------===//
//
// This file implements the HIR schedule verifier.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"

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

#include "../PassDetails.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <list>
#include <stack>

using namespace circt;
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
  TimeInstant getRootTimeInstant(TimeInstant instant) {
    TimeInstant out = instant;
    while (1) {
      auto it = mapValueToTimeInstant.find(out.t);
      if (it == mapValueToTimeInstant.end()) {
        emitError(out.t.getLoc(),
                  "Could not find any mapping for this time var.");
        return out;
      }

      TimeInstant prevInstant = it->getSecond();
      if (prevInstant.t == out.t) {
        if (prevInstant.delay > 0) {
          emitError(out.t.getLoc(),
                    "Circular mapping found for this time var.");
          return out;
        }
        break;
      }
      out.delay += prevInstant.delay;
      out.t = prevInstant.t;
    }
    return out;
  }

  bool insert(Value v, Value t, unsigned delay) {
    assert(t);
    assert(t.getType().isa<TimeType>());
    assert(v);
    if (!(v.getType().isa<IntegerType>() ||
          v.getType().isa<mlir::FloatType>() || v.getType().isa<TimeType>())) {
      assert(false);
    }
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
          }
          break;
        }
        delay += instant.delay;
        t = instant.t;
      }
    TimeInstant vInstant(t, delay);
    mapValueToTimeInstant[v] = vInstant;
    return true;
  }

  TimeInstant getTimeInstantOrError(Value v) {
    auto it = mapValueToTimeInstant.find(v);
    if (it == mapValueToTimeInstant.end()) {
      emitError(v.getLoc(), "Could not find time instant in schedule.");
      return TimeInstant();
    }
    return it->getSecond();
  }

  bool check(mlir::Location opLoc, mlir::Location defLoc, Value v, Value t,
             unsigned delay, std::string useLoc) {
    // consts are valid at all time instants.
    Type vType = v.getType();
    if (vType.isa<IndexType>() || vType.isa<hir::MemrefType>())
      return true;
    TimeInstant instant = getTimeInstantOrError(v);
    TimeInstant instant2 = getRootTimeInstant(TimeInstant(t, delay));

    if (!instant.t) {
      assert(false);
      return false;
    }
    if (instant.t == instant2.t && instant.delay == instant2.delay)
      return true;

    std::string error;
    if (instant.t != instant2.t)
      error = "Schedule error: mismatched time instants in " + useLoc + "!";
    else
      error = "\n\tSchedule error: mismatched delay (" +
              std::to_string(instant.delay) + " vs " + std::to_string(delay) +
              ") in " + useLoc + "!";
    emitError(opLoc, error).attachNote(defLoc).append("Prior definition here.");

    return false;
  }
}; // namespace
/// Checks for out of bound memef access subscripts..
class ScheduleVerifier : public hir::ScheduleVerifierBase<ScheduleVerifier> {
public:
  void runOnOperation() override;

private:
  bool inspectOp(hir::FuncOp op);
  bool inspectOp(mlir::arith::ConstantOp op);
  bool inspectOp(ForOp op);
  bool inspectOp(hir::LoadOp op);
  bool inspectOp(hir::StoreOp op);
  bool inspectOp(hir::NextIterOp op);
  bool inspectOp(hir::BusSendOp op);
  bool inspectOp(hir::BusRecvOp op);
  bool inspectOp(hir::AllocaOp op);
  bool inspectOp(hir::DelayOp op);
  bool inspectOp(hir::CallOp op);
  bool inspectOp(Operation *op);
  bool inspectBody(Block &body);

private:
  void setIntegerConst(Value v, int constant) {
    mapValueToIntConst[v] = constant;
  }

  bool isIntegerConst(Value v) {
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
      assert(false);
    }
    return mapValueToIntConst[v];
  }

  mlir::Location getDefiningLoc(Value v) {
    auto it = mapValueToDefiningLoc.find(v);
    if (it != mapValueToDefiningLoc.end()) {
      return it->getSecond();
    }
    return v.getLoc();
  }

  llvm::DenseMap<Value, int> mapValueToIntConst;

  // Used for loc of region parameters such as induction var.
  llvm::DenseMap<Value, mlir::Location> mapValueToDefiningLoc;
  std::vector<Operation *> opsToErase;

private:
  Schedule schedule;
  std::stack<TimeInstant> yieldPoints;
  ArrayRef<DictionaryAttr> resultAttrs;
  Value tstart;

private:
  llvm::DenseMap<Value, Value> mapValueToTimeVar;
  llvm::DenseMap<Value, int64_t> mapValueToOffset;
  llvm::SetVector<Value> setOfForeverValidValues;
};

bool ScheduleVerifier::inspectOp(hir::FuncOp op) {
  Block &entryBlock = op.getFuncBody().front();
  // args also contains tstart;
  auto args = entryBlock.getArguments();
  Value tstart = args.back();
  this->tstart = tstart;
  hir::FuncType funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  auto inputAttrs = funcTy.getInputAttrs();
  this->resultAttrs = funcTy.getResultAttrs();
  // Indentity map for root level time vars.
  schedule.insert(tstart, tstart, 0);
  for (unsigned i = 0; i < inputAttrs.size(); i++) {
    mapValueToDefiningLoc.insert(std::make_pair(args[i], op.getLoc()));
    if (!helper::isBuiltinSizedType(args[i].getType()))
      continue;
    int delay = inputAttrs[i]
                    .dyn_cast<DictionaryAttr>()
                    .getNamed("delay")
                    .getValue()
                    .second.dyn_cast<IntegerAttr>()
                    .getInt();
    schedule.insert(args[i], tstart, delay);
  }
  return inspectBody(entryBlock);
}

bool ScheduleVerifier::inspectOp(mlir::arith::ConstantOp op) {

  if (op.getResult().getType().dyn_cast<IndexType>())
    setIntegerConst(op.getResult(),
                    op.value().dyn_cast<IntegerAttr>().getInt());
  return true;
}

bool ScheduleVerifier::inspectOp(ForOp op) {
  Value idx = op.getInductionVar();
  Value tloop = op.getIterTimeVar();
  unsigned delayValue = op.offset().getValue();
  assert(delayValue > 0);
  schedule.insert(tloop, tloop, 0);
  schedule.insert(idx, tloop, 0);
  mapValueToDefiningLoc.insert(std::make_pair(idx, op.getLoc()));
  mapValueToDefiningLoc.insert(std::make_pair(tloop, op.getLoc()));
  bool ok = true;
  ok &= schedule.check(op.getLoc(), getDefiningLoc(op.lb()), op.lb(),
                       op.tstart(), delayValue - 1, "lower bound");
  ok &= schedule.check(op.getLoc(), getDefiningLoc(op.ub()), op.ub(),
                       op.tstart(), delayValue - 1, "upper bound");
  ok &= schedule.check(op.getLoc(), getDefiningLoc(op.step()), op.step(),
                       op.tstart(), delayValue - 1, "step");
  yieldPoints.push(TimeInstant());
  ok &= inspectBody(op.getLoopBody().front());
  yieldPoints.pop();
  return ok;
} // namespace

bool ScheduleVerifier::inspectOp(hir::LoadOp op) {
  auto indices = op.indices();
  Value result = op.res();
  Value tstart = op.tstart();
  int offset = op.offset() ? op.offset().getValue() : 0;
  assert(offset >= 0);

  bool ok = true;
  int c = 1;
  for (auto idx : indices) {
    ok &= schedule.check(op.getLoc(), getDefiningLoc(idx), idx, tstart, offset,
                         "indicesess " + std::to_string(c));
    c++;
  }
  auto delay = op.delay().getValueOr(0);

  schedule.insert(result, tstart, offset + delay);
  return ok;
}

bool ScheduleVerifier::inspectOp(hir::StoreOp op) {
  auto indices = op.indices();
  Value value = op.value();
  Value tstart = op.tstart();
  int offset = op.offset() ? op.offset().getValue() : 0;
  assert(offset >= 0);
  bool ok = true;
  ok &= schedule.check(op.getLoc(), getDefiningLoc(value), value, tstart,
                       offset, "input var");
  int c = 1;
  for (auto idx : indices) {
    mlir::Location locAddrI = getDefiningLoc(idx);
    ok &= schedule.check(op.getLoc(), locAddrI, idx, tstart, offset,
                         "indicesess " + std::to_string(c));
    c++;
  }
  return ok;
}

bool ScheduleVerifier::inspectOp(hir::NextIterOp op) {
  Value tstart = op.tstart();
  unsigned offset = op.offset() ? op.offset().getValue() : 0;
  yieldPoints.top().t = tstart;
  yieldPoints.top().delay = offset;
  return true;
}

bool ScheduleVerifier::inspectOp(hir::BusSendOp op) {
  unsigned offset = op.offset() ? op.offset().getValue() : 0;
  return schedule.check(op.getLoc(), getDefiningLoc(op.value()), op.value(),
                        op.tstart(), offset, "input var");
}

bool ScheduleVerifier::inspectOp(hir::BusRecvOp op) {
  unsigned offset = op.offset() ? op.offset().getValue() : 0;
  schedule.insert(op.res(), op.tstart(), offset);
  return true;
}

bool ScheduleVerifier::inspectOp(hir::AllocaOp op) { return true; }

bool ScheduleVerifier::inspectOp(hir::DelayOp op) {
  unsigned offset = op.offset() ? op.offset().getValue() : 0;
  unsigned latency = op.delay();
  bool ok = schedule.check(op.getLoc(), getDefiningLoc(op.input()), op.input(),
                           op.tstart(), offset, "input var");
  schedule.insert(op.res(), op.tstart(), offset + latency);
  return ok;
}

bool ScheduleVerifier::inspectOp(hir::CallOp op) {
  mlir::ResultRange results = op.results();
  auto operands = op.operands();
  unsigned tstartDelay = op.offset() ? op.offset().getValue() : 0;
  ArrayAttr inputDelays = op->getAttrOfType<ArrayAttr>("inputDelays");
  ArrayAttr outputDelays = op->getAttrOfType<ArrayAttr>("outputDelays");
  assert(inputDelays.size() == operands.size());
  assert(outputDelays.size() == results.size());
  bool ok = true;
  for (unsigned i = 0; i < operands.size(); i++) {
    auto argDelay = inputDelays[i].cast<IntegerAttr>().getInt();
    ok &= schedule.check(op.getLoc(), getDefiningLoc(operands[i]), operands[i],
                         op.tstart(), tstartDelay + argDelay,
                         "operand " + std::to_string(i + 1));
  }
  for (unsigned i = 0; i < results.size(); i++) {
    auto resDelay = outputDelays[i].cast<IntegerAttr>().getInt();
    schedule.insert(results[i], op.tstart(), tstartDelay + resDelay);
  }
  return ok;
}

bool ScheduleVerifier::inspectOp(Operation *inst) {
  if (auto op = dyn_cast<mlir::arith::ConstantOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::CallOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::AllocaOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::DelayOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::ForOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::LoadOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::StoreOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::BusRecvOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::BusSendOp>(inst))
    return inspectOp(op);
  if (auto op = dyn_cast<hir::NextIterOp>(inst))
    return inspectOp(op);
  emitError(inst->getLoc(), "Unsupported Operation for verification!");
  return false;
}

bool ScheduleVerifier::inspectBody(Block &block) {

  // Print the operations within the entity.
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    if (!inspectOp(&(*iter))) {
      // emitWarning(iter->getLoc(), "errored");
      // return false;
    }
  }
  for (auto *operation : opsToErase) {
    operation->erase();
  }
  return true;
}
} // end anonymous namespace

void ScheduleVerifier::runOnOperation() { inspectOp(getOperation()); }

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createScheduleVerificationPass() {
  return std::make_unique<ScheduleVerifier>();
}
} // namespace hir
} // namespace circt
