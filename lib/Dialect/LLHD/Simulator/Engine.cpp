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

#include "circt/Conversion/LLHDToLLVM.h"
#include "circt/Dialect/LLHD/Simulator/Engine.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/TargetSelect.h"

using namespace circt::llhd::sim;

Engine::Engine(
    llvm::raw_ostream &out, ModuleOp module,
    llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)> mlirTransformer,
    llvm::function_ref<llvm::Error(llvm::Module *)> llvmTransformer,
    std::string root, TraceMode tm, ArrayRef<StringRef> sharedLibPaths)
    : out(out), root(root), traceMode(tm) {
  state = std::make_unique<State>(root + '.' + root);

  buildLayout(module);

  auto rootEntity = module.lookupSymbol<EntityOp>(root);

  // Insert explicit instantiation of the design root.
  OpBuilder insertInst =
      OpBuilder::atBlockEnd(&rootEntity.getBody().getBlocks().front());
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

int Engine::simulate(uint64_t maxCycle, uint64_t maxTime) {
  assert(engine && "engine not found");
  assert(state && "state not found");

  Trace trace(state, out, traceMode);

  SmallVector<void *, 1> arg({&state});
  // Initialize tbe simulation state.
  auto res = engine->invokePacked("llhd_init", arg);
  if (res) {
    llvm::errs() << "Failed invocation of llhd_init: " << res;
    return -1;
  }

  if (traceMode != TraceMode::None) {
    // Add changes for all the signals' initial values.
    for (size_t i = 0, e = state->getSignalSize(); i < e; ++i) {
      trace.addChange(i);
    }
  }

  // Add a dummy event to get the simulation started.
  state->pushEvent(Slot(Time()));

  // Keep track of the instances that need to wakeup.
  llvm::SmallVector<unsigned, 8> wakeupQueue;

  // Add all instances to the wakeup queue for the first run and add the jitted
  // function pointers to all of the instances to make them readily available.
  for (size_t i = 0, e = state->getInstanceSize(); i < e; ++i) {
    wakeupQueue.push_back(i);
    auto &instOp = state->getInstance(i);
    auto expectedFPtr = engine->lookupPacked(instOp.getBaseUnitName());
    if (!expectedFPtr) {
      llvm::errs() << "Could not lookup " << instOp.getBaseUnitName() << "!\n";
      return -1;
    }
    instOp.registerRunner(*expectedFPtr);
  }

  uint64_t cycle = 0;
  while (state->hasEvents()) {
    const auto &event = state->popEvent();

    // Interrupt the simulation if a stop condition is met.
    if (cycle >= maxCycle || event.getTimeTime() > maxTime)
      break;

    // Update the simulation time.
    state->updateSimTime(event.getTime());

    if (traceMode != TraceMode::None)
      trace.flush();

    // Process signal changes.
    size_t i = 0, e = event.getNumChanges();
    while (i < e) {
      const auto sigIdx = event.getChangedSignalIndex(i);
      auto &signal = state->getSignal(sigIdx);
      APInt newSignal(
          signal.Size() * 8,
          llvm::makeArrayRef(reinterpret_cast<uint64_t *>(signal.Value()),
                             llvm::divideCeil(signal.Size(), 8)));

      // Apply the changes to the buffer until we reach the next signal.
      while (i < e && event.getChangedSignalIndex(i) == sigIdx) {
        const auto &changedSignal = event.getChangedSignal(i);
        if (changedSignal.getBitWidth() < newSignal.getBitWidth()) {
          newSignal.insertBits(changedSignal, event.getChangedSignalInsPos(i));
        } else {
          newSignal = changedSignal;
        }

        ++i;
      }

      if (!signal.updateWhenChanged(newSignal.getRawData()))
        continue;

      // Add sensitive instances to wakeup list
      for (auto ii : signal.getTriggeredInstanceIndices()) {
        auto &instance = state->getInstance(ii);
        // Skip if the process is not currently sensible to the signal.
        if (instance.isProc()) {
          if (instance.isWaitingOnSignal(sigIdx))
            continue;

          instance.invalidWakeupTime();
        }

        wakeupQueue.push_back(ii);
      }

      // Dump the updated signal.
      if (traceMode != TraceMode::None)
        trace.addChange(sigIdx);
    }

    // Add scheduled process resumes to the wakeup queue.
    for (auto ii : event.getScheduledWakeups()) {
      if (state->getInstance(ii).shouldWakeup(state->getSimTime()))
        wakeupQueue.push_back(ii);
    }

    // Sort the instance index and deduplicate(~3% of total simulation time)
    std::sort(wakeupQueue.begin(), wakeupQueue.end());
    auto pivot = std::unique(wakeupQueue.begin(), wakeupQueue.end());
    wakeupQueue.erase(pivot, wakeupQueue.end());

    // Run the instances present in the wakeup queue.
    for (auto i : wakeupQueue)
      state->getInstance(i).run(reinterpret_cast<void *>(&state));

    // Clear wakeup queue.
    wakeupQueue.clear();

    ++cycle;
  }

  // Force flush any remaining changes
  if (traceMode != TraceMode::None)
    trace.flush(true);

  llvm::errs() << "Finished at " << state->getSimTime().toString()
               << " (" << cycle << " cycles)\n";
  return 0;
}

void Engine::buildLayout(ModuleOp module) {
  // Start from the root entity.
  auto rootEntity = module.lookupSymbol<EntityOp>(root);
  assert(rootEntity && "root entity not found!");

  // Build root instance(always an entity type), the parent and instance names
  // are the same for the root.
  Instance rootInst(state->getRoot(), root, root, Instance::Entity);

  // Recursively walk the units starting at root.
  walkEntity(rootEntity, rootInst);

  // Store the root instance.
  state->pushInstance(rootInst);
  // Bind signals and instances
  state->bindSignalWithInstance();
}

void Engine::walkEntity(EntityOp entity, Instance &instance) {
  entity.walk([&](Operation *op) {
    assert(op);

    if (auto sigOp = dyn_cast<SigOp>(op)) {
      // Collect signal information in current instance.
      uint64_t index = state->addSignal(instance, sigOp.name().str());
      instance.initSignalDetail(index);
    } else if (auto instOp = dyn_cast<InstOp>(op)) {
      // Build (recursive) instance layout.

      // Skip self-recursion.
      if (instOp.callee() == instance.getName())
        return;

      if (auto sym =
            op->getParentOfType<ModuleOp>().lookupSymbol(instOp.callee())) {
        std::string instName = instance.getBaseUnitName() + '.' +
            instOp.name().str();
        std::string instPath = instance.getPath() + "/" + instOp.name().str();
        Instance childInst(instName, instPath, instOp.callee().str(),
                           instOp.getNumOperands());

        // Add instance arguments to sensitivity list. The first numArgs signals
        // in the sensitivity list represent the unit's arguments(inputs), while
        // the following ones represent the unit-defined signals(outputs).

        // The signal comes from parent instance's argument.
        for (size_t i = 0, e = instOp.inputs().size(); i < e; ++i) {
          childInst.pushSignalDetail(instance.getSignalDetail(i));
        }

        // The signal comes from parent instance's owned signals.
        for (auto output : instOp.outputs()) {
          auto sigOp = cast<SigOp>(output.getDefiningOp());

          auto SDI = std::find_if(
              instance.getSensitivityList().begin(),
              instance.getSensitivityList().end(),
              [&](const SignalDetail &detail) {
                auto& outSig = state->getSignal(detail.globalIndex);
                return outSig.getName() == sigOp.name() &&
                       outSig.getOwner() == instance.getName();
              });

          if (SDI != instance.getSensitivityList().end()) {
            childInst.pushSignalDetail(*SDI);
          }
        }

        // Recursively walk a new entity, otherwise it is a process and cannot
        // define new signals or instances.
        if (auto entityOp = dyn_cast<EntityOp>(sym)) {
          childInst.setType(Instance::Entity);
          walkEntity(entityOp, childInst);
        } else {
          childInst.setType(Instance::Proc);
        }

        // Store the created instance.
        state->pushInstance(childInst);
      }
    }
  });
}
