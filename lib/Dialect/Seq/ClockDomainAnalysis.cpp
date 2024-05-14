//===- ClockDomainAnalysis.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "clock-domain"

using namespace circt;
using namespace seq;
using namespace hw;

class ClockDomainAnalysis final {
  using InstancePath = SmallVector<HWInstanceLike>;
  using ClockSet = llvm::SmallSetVector<Value, 4>;
  using GT = llvm::GraphTraits<circt::hw::detail::HWOperation *>;

public:
  std::function<bool(Operation *)> isClockedFunc;
  std::function<bool(Operation *)> isClockTransformationFunc;

  bool isClockedOp(Operation *op) {
    return isa<seq::Clocked>(op) && (!isClockedFunc || isClockedFunc(op));
  }
  bool isClockTransformationOp(Operation *op) {
    return (isa<seq::ClockGateOp, seq::ClockDividerOp, seq::ClockInverterOp,
                seq::FromClockOp>(op) &&
            (!isClockTransformationFunc || isClockTransformationFunc(op)));
  }

  ClockDomainAnalysis(InstanceGraph &graph) : graph(graph) {}
  void traverseFrom(OpOperand *root, Value baseClock) {

    SmallVector<std::pair<InstancePath, OpOperand *>> q;
    DenseSet<std::pair<InstancePath, Operation *>> visited;
    q.emplace_back(InstancePath{}, root);
    while (!q.empty()) {
      auto [path, operand] = q.front();
      q.pop_back();
      clockDomains[{path, operand}].insert(baseClock);

      auto *owner = operand->getOwner();

      if (!visited.insert({path, owner}).second)
        continue;

      // Follow into instances.
      if (auto inst = dyn_cast<HWInstanceLike>(owner)) {
        auto refs = inst.getReferencedModuleNamesAttr();
        assert(refs.size() == 1 && "invalid instance");
        Operation *targetOp =
            graph.lookup(cast<StringAttr>(refs[0]))->getModule();

        if (auto module = dyn_cast<HWModuleOp>(targetOp)) {
          unsigned argNo = operand->getOperandNumber();
          Block *body = module.getBodyBlock();

          path.push_back(inst);
          for (auto &use : body->getArgument(argNo).getUses()) {
            q.emplace_back(path, &use);
          }
          path.pop_back();
        }
        // Nothing to do for extern modules.
        continue;
      }

      // Follow through to instances from output.
      if (auto output = dyn_cast<hw::OutputOp>(owner)) {
        if (path.empty()) {
          // This can happen only if baseClock can be traced to escape to its
          // parent module. Nothing to do here, since calling context not
          // available.
          LLVM_DEBUG(llvm::dbgs()
                     << baseClock << " can be traced to the output " << output);
          continue;
        }
        unsigned resultNo = operand->getOperandNumber();
        HWInstanceLike inst = path.pop_back_val();
        for (auto &use : inst->getResult(resultNo).getUses())
          q.emplace_back(path, &use);

        continue;
      }

      if (isClockTransformationOp(owner)) {
        derivedClocks[{path, owner->getResult(0)}] = baseClock;
      }
      if (isClockedOp(owner))
        clockDomains[{path, operand}].insert(baseClock);

      for (auto &use : owner->getUses())
        q.emplace_back(path, &use);

      continue;

      // Record the clocks of clocked operations.
      if (auto clocked = dyn_cast<seq::Clocked>(owner)) {
        clocks.insert({path, clocked.getClk()});
        setClockDomain(path, operand, clocked.getClk());
        continue;
      }

      // Record clocks from source ops.
      if (auto source = dyn_cast<libdn::ChannelSourceOp>(owner)) {
        clocks.insert({path, source.getClock()});
        setClockDomain(path, operand, source.getClock());
        continue;
      }

      // Record the triggers of custom clocked ops (prints).
      if (auto always = dyn_cast<sv::AlwaysOp>(owner)) {
        for (Value clock : always.getClocks()) {
          if (auto cast = clock.getDefiningOp<seq::FromClockOp>()) {
            clocks.insert({path, cast.getInput()});
            setClockDomain(path, useOperand, cast.getInput());
          }
        }
        continue;
      }
      if (auto always = owner->getParentOfType<sv::AlwaysOp>()) {
        for (Value clock : always.getClocks()) {
          if (auto cast = clock.getDefiningOp<seq::FromClockOp>()) {
            clocks.insert({path, cast.getInput()});
            setClockDomain(path, useOperand, cast.getInput());
          }
        }
        continue;
      }

      if (auto ccOp = dyn_cast<sv::CoverConcurrentOp>(owner)) {
        auto cast = ccOp.getClock().getDefiningOp<seq::FromClockOp>();
        assert(cast && "unknown clock source");
        clocks.insert({path, cast.getInput()});
        setClockDomain(path, operand, cast.getInput());
        continue;
      }

      if (auto assumeOp = dyn_cast<sv::AssumeConcurrentOp>(owner)) {
        auto cast = assumeOp.getClock().getDefiningOp<seq::FromClockOp>();
        assert(cast && "unknown clock source");
        clocks.insert({path, cast.getInput()});
        setClockDomain(path, operand, cast.getInput());
        continue;
      }

      if (auto acOp = dyn_cast<sv::AssertConcurrentOp>(owner)) {
        auto cast = acOp.getClock().getDefiningOp<seq::FromClockOp>();
        assert(cast && "unknown clock source");
        clocks.insert({path, cast.getInput()});
        setClockDomain(path, operand, cast.getInput());
        continue;
      }

      owner->dump();
      llvm_unreachable("not implemented");
    }
  }
  void run(SmallVector<Value> &baseClocks) {

    for (auto clock : baseClocks) {
      for (auto useOp : clock.getUses()) {
        traverseFrom(&useOp, clock);
      }
    }
  }

private:
  DenseMap<std::pair<InstancePath, Value>, Value> derivedClocks;
  DenseMap<std::pair<InstancePath, OpOperand *>, ClockSet> clockDomains;

  InstanceGraph &graph;
};
