//===- Engine.cpp - Simulator Engine class implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Engine class.
//
//===----------------------------------------------------------------------===//

#include "State.h"
#include "Trace.h"

#include "circt/Conversion/ConvertToLLVM/ConvertToLLVM.h"
#include "circt/Dialect/LLHD/Simulator/Engine.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/TargetSelect.h"

using namespace circt::llhd::sim;

Engine::Engine(
    llvm::raw_ostream &out, ModuleOp module,
    llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)> mlirTransformer,
    llvm::function_ref<llvm::Error(llvm::Module *)> llvmTransformer,
    std::string root, int mode, ArrayRef<StringRef> sharedLibPaths)
    : out(out), root(root), traceMode(mode) {
  state = std::make_unique<State>();
  state->root = root + '.' + root;

  buildLayout(module);

  auto rootEntity = module.lookupSymbol<EntityOp>(root);

  // Insert explicit instantiation of the design root.
  OpBuilder insertInst =
      OpBuilder::atBlockTerminator(&rootEntity.getBody().getBlocks().front());
  insertInst.create<InstOp>(rootEntity.getBlocks().front().back().getLoc(),
                            llvm::None, root, root, ArrayRef<Value>(),
                            ArrayRef<Value>());

  if (failed(mlirTransformer(module))) {
    llvm::errs() << "failed to apply the MLIR passes\n";
    exit(EXIT_FAILURE);
  }

  this->module = module;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto maybeEngine = mlir::ExecutionEngine::create(
      this->module, nullptr, llvmTransformer,
      /*jitCodeGenOptLevel=*/llvm::None, /*sharedLibPaths=*/sharedLibPaths);
  assert(maybeEngine && "failed to create JIT");
  engine = std::move(*maybeEngine);
}

Engine::~Engine() = default;

void Engine::dumpStateLayout() { state->dumpLayout(); }

void Engine::dumpStateSignalTriggers() { state->dumpSignalTriggers(); }

int Engine::simulate(int n, uint64_t maxTime) {
  assert(engine && "engine not found");
  assert(state && "state not found");

  auto tm = static_cast<TraceMode>(traceMode);
  Trace trace(state, out, tm);

  SmallVector<void *, 1> arg({&state});
  // Initialize tbe simulation state.
  auto invocationResult = engine->invokePacked("llhd_init", arg);
  if (invocationResult) {
    llvm::errs() << "Failed invocation of llhd_init: " << invocationResult;
    return -1;
  }

  if (traceMode >= 0) {
    // Add changes for all the signals' initial values.
    for (size_t i = 0, e = state->signals.size(); i < e; ++i) {
      trace.addChange(i);
    }
  }

  // Add a dummy event to get the simulation started.
  state->queue.push_back(Slot(Time()));
  ++state->queue.events;

  // Keep track of the instances that need to wakeup.
  llvm::SmallVector<unsigned, 8> wakeupQueue;

  // Add all instances to the wakeup queue for the first run and add the jitted
  // function pointers to all of the instances to make them readily available.
  for (size_t i = 0, e = state->instances.size(); i < e; ++i) {
    wakeupQueue.push_back(i);
    auto &inst = state->instances[i];
    auto expectedFPtr = engine->lookup(inst.unit);
    if (!expectedFPtr) {
      llvm::errs() << "Could not lookup " << inst.unit << "!\n";
      return -1;
    }
    inst.unitFPtr = *expectedFPtr;
  }

  int cycle = 0;
  while (state->queue.events > 0) {
    const auto &pop = state->queue.top();

    // Interrupt the simulation if a stop condition is met.
    if ((n > 0 && cycle >= n) || (maxTime > 0 && pop.time.time > maxTime)) {
      break;
    }

    // Update the simulation time.
    state->time = pop.time;

    if (traceMode >= 0)
      trace.flush();

    // Process signal changes.
    size_t i = 0, e = pop.changesSize;
    while (i < e) {
      const auto sigIndex = pop.changes[i].first;
      const auto &curr = state->signals[sigIndex];
      APInt buff(
          curr.size * 8,
          llvm::makeArrayRef(reinterpret_cast<uint64_t *>(curr.value.get()),
                             llvm::divideCeil(curr.size, 8)));

      // Apply the changes to the buffer until we reach the next signal.
      while (i < e && pop.changes[i].first == sigIndex) {
        const auto &change = pop.buffers[pop.changes[i].second];
        const auto offset = change.first;
        const auto &drive = change.second;
        if (drive.getBitWidth() < buff.getBitWidth())
          buff.insertBits(drive, offset);
        else
          buff = drive;

        ++i;
      }

      // Skip if the updated signal value is equal to the initial value.
      if (std::memcmp(curr.value.get(), buff.getRawData(), curr.size) == 0)
        continue;

      // Apply the signal update.
      std::memcpy(curr.value.get(), buff.getRawData(), curr.size);

      // Add sensitive instances.
      for (auto inst : curr.triggers) {
        // Skip if the process is not currently sensible to the signal.
        if (!state->instances[inst].isEntity) {
          const auto &sensList = state->instances[inst].sensitivityList;
          auto it = std::find_if(sensList.begin(), sensList.end(),
                                 [sigIndex](const SignalDetail &sig) {
                                   return sig.globalIndex == sigIndex;
                                 });
          if (sensList.end() != it &&
              state->instances[inst].procState->senses[it - sensList.begin()] ==
                  0)
            continue;

          // Invalidate scheduled wakeup
          state->instances[inst].expectedWakeup = Time();
        }
        wakeupQueue.push_back(inst);
      }

      // Dump the updated signal.
      if (traceMode >= 0)
        trace.addChange(sigIndex);
    }

    // Add scheduled process resumes to the wakeup queue.
    for (auto inst : pop.scheduled) {
      if (state->time == state->instances[inst].expectedWakeup)
        wakeupQueue.push_back(inst);
    }

    state->queue.pop();

    std::sort(wakeupQueue.begin(), wakeupQueue.end());
    wakeupQueue.erase(std::unique(wakeupQueue.begin(), wakeupQueue.end()),
                      wakeupQueue.end());

    // Run the instances present in the wakeup queue.
    for (auto i : wakeupQueue) {
      auto &inst = state->instances[i];
      auto signalTable = inst.sensitivityList.data();

      // Gather the instance arguments for unit invocation.
      SmallVector<void *, 3> args;
      if (inst.isEntity)
        args.assign({&state, &inst.entityState, &signalTable});
      else {
        args.assign({&state, &inst.procState, &signalTable});
      }
      // Run the unit.
      (*inst.unitFPtr)(args.data());
    }

    // Clear wakeup queue.
    wakeupQueue.clear();
    ++cycle;
  }

  if (traceMode >= 0) {
    // Flush any remainign changes
    trace.flush(/*force=*/true);
  }

  llvm::errs() << "Finished at " << state->time.dump() << " (" << cycle
               << " cycles)\n";
  return 0;
}

void Engine::buildLayout(ModuleOp module) {
  // Start from the root entity.
  auto rootEntity = module.lookupSymbol<EntityOp>(root);
  assert(rootEntity && "root entity not found!");

  // Build root instance, the parent and instance names are the same for the
  // root.
  Instance rootInst(state->root);
  rootInst.unit = root;
  rootInst.path = root;

  // Recursively walk the units starting at root.
  walkEntity(rootEntity, rootInst);

  // The root is always an instance.
  rootInst.isEntity = true;
  // Store the root instance.
  state->instances.push_back(std::move(rootInst));

  // Add triggers to signals.
  for (size_t i = 0, e = state->instances.size(); i < e; ++i) {
    auto &inst = state->instances[i];
    for (auto trigger : inst.sensitivityList) {
      state->signals[trigger.globalIndex].triggers.push_back(i);
    }
  }
}

void Engine::walkEntity(EntityOp entity, Instance &child) {
  entity.walk([&](Operation *op) {
    assert(op);

    // Add a signal to the signal table.
    if (auto sig = dyn_cast<SigOp>(op)) {
      uint64_t index = state->addSignal(sig.name().str(), child.name);
      child.sensitivityList.push_back(
          SignalDetail({nullptr, 0, child.sensitivityList.size(), index}));
    }

    // Build (recursive) instance layout.
    if (auto inst = dyn_cast<InstOp>(op)) {
      // Skip self-recursion.
      if (inst.callee() == child.name)
        return;
      if (auto e =
              op->getParentOfType<ModuleOp>().lookupSymbol(inst.callee())) {
        Instance newChild(child.unit + '.' + inst.name().str());
        newChild.unit = inst.callee().str();
        newChild.nArgs = inst.getNumOperands();
        newChild.path = child.path + "/" + inst.name().str();

        // Add instance arguments to sensitivity list. The first nArgs signals
        // in the sensitivity list represent the unit's arguments, while the
        // following ones represent the unit-defined signals.
        llvm::SmallVector<Value, 8> args;
        args.insert(args.end(), inst.inputs().begin(), inst.inputs().end());
        args.insert(args.end(), inst.outputs().begin(), inst.outputs().end());

        for (size_t i = 0, e = args.size(); i < e; ++i) {
          // The signal comes from an instance's argument.
          if (auto blockArg = args[i].dyn_cast<BlockArgument>()) {
            auto detail = child.sensitivityList[blockArg.getArgNumber()];
            detail.instIndex = i;
            newChild.sensitivityList.push_back(detail);
          } else if (auto sig = dyn_cast<SigOp>(args[i].getDefiningOp())) {
            // The signal comes from one of the instance's owned signals.
            auto it = std::find_if(
                child.sensitivityList.begin(), child.sensitivityList.end(),
                [&](SignalDetail &detail) {
                  return state->signals[detail.globalIndex].name ==
                             sig.name() &&
                         state->signals[detail.globalIndex].owner == child.name;
                });
            if (it != child.sensitivityList.end()) {
              auto detail = *it;
              detail.instIndex = i;
              newChild.sensitivityList.push_back(detail);
            }
          }
        }

        // Recursively walk a new entity, otherwise it is a process and cannot
        // define new signals or instances.
        if (auto ent = dyn_cast<EntityOp>(e)) {
          newChild.isEntity = true;
          walkEntity(ent, newChild);
        } else {
          newChild.isEntity = false;
        }

        // Store the created instance.
        state->instances.push_back(std::move(newChild));
      }
    }
  });
}
