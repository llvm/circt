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

void Engine::dumpStateLayout() { state->dumpLayout(); }

void Engine::dumpStateSignalTriggers() { state->dumpSignalTriggers(); }

int Engine::simulate(uint64_t maxCycle, uint64_t maxTime) {
  assert(engine && "engine not found");
  assert(state && "state not found");

  Trace trace(state, out, traceMode);

  // FIXME: This 1 sized vector looks weired
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
    auto &inst = state->getInstance(i);
    auto expectedFPtr = engine->lookupPacked(inst.getBaseUnitName());
    if (!expectedFPtr) {
      llvm::errs() << "Could not lookup " << inst.getBaseUnitName() << "!\n";
      return -1;
    }
    inst.registerRunner(*expectedFPtr);
  }

  uint64_t cycle = 0;
  while (state->hasEvents()) {
    const auto &event = state->popEvent();

    // Interrupt the simulation if a stop condition is met.
    if (cycle >= maxCycle || event.getTimeTime() >= maxTime)
      break;

    // Update the simulation time.
    state->updateTime(event.getTime());

    trace.flush();

    // Process signal changes.
    size_t i = 0, e = event.getNumChanges();
    while (i < e) {
      const auto change = event.getChange(i);
      const auto &signal = state->getSignal(change.first);
      APInt buff(
          signal.Size() * 8,
          llvm::makeArrayRef(reinterpret_cast<uint64_t *>(signal.Value()),
                             llvm::divideCeil(signal.Size(), 8)));

      // Apply the changes to the buffer until we reach the next signal.
      while (i < e && event.getChange(i).first == change.first) {
        const auto &buffer = event.getChangedBuffer(change.second);
        const auto &drive = buffer.second;
        if (drive.getBitWidth() < buff.getBitWidth())
          buff.insertBits(drive, buffer.first);
        else
          buff = drive;

        ++i;
      }

      // Skip if the updated signal value is equal to the initial value.
      if (std::memcmp(signal.Value(), buff.getRawData(), signal.Size()) == 0)
        continue;

      // Apply the signal update.
      std::memcpy(signal.Value(), buff.getRawData(), signal.Size());

      // Add sensitive instances.
      for (auto ii : signal.getTriggeredInstanceIndices()) {
        auto &inst = state->getInstance(ii);
        // Skip if the process is not currently sensible to the signal.
        if (inst.isProc()) {
          // FIXME: It may have bug in original logic here!
          if (!inst.isWaitingOnSignal(change.first))
            continue;

          // Invalidate scheduled wakeup
          inst.invalidWakeupTime();
        }
        wakeupQueue.push_back(ii);
      }

      // Dump the updated signal.
      trace.addChange(change.first);
    }

    // Add scheduled process resumes to the wakeup queue.
    for (auto ii : event.getScheduledWakups()) {
      if (state->getInstance(i).shouldWakeup(state->getTime()))
        wakeupQueue.push_back(ii);
    }

    std::sort(wakeupQueue.begin(), wakeupQueue.end());
    wakeupQueue.erase(std::unique(wakeupQueue.begin(), wakeupQueue.end()),
                      wakeupQueue.end());

    // Run the instances present in the wakeup queue.
    for (auto i : wakeupQueue)
      state->getInstance(i).run(reinterpret_cast<void*>(&state));

    // Clear wakeup queue.
    wakeupQueue.clear();
    ++cycle;
  }

  // Force flush any remaining changes
  trace.flush(true);

  llvm::errs() << "Finished at " << state->getTime().toString()
               << " (" << cycle << " cycles)\n";
  return 0;
}

void Engine::buildLayout(ModuleOp module) {
  // Start from the root entity.
  auto rootEntity = module.lookupSymbol<EntityOp>(root);
  assert(rootEntity && "root entity not found!");

  // Build root instance, the parent and instance names are the same for the
  // root.
  Instance rootInst(state->getRoot(), root, root);

  // Recursively walk the units starting at root.
  walkEntity(rootEntity, rootInst);

  // The root is always an instance.
  rootInst.setType(Instance::Entity);
  // Store the root instance.
  state->pushInstance(rootInst);
  // Add triggers to signals.
  state->associateTrigger2Signal();
}

void Engine::walkEntity(EntityOp entity, Instance &child) {
  entity.walk([&](Operation *op) {
    assert(op);

    // Add a signal to the signal table.
    if (auto sig = dyn_cast<SigOp>(op)) {
      uint64_t index = state->addSignal(sig.name().str(), child.getName());
      child.initSignalDetail(index);
    }

    // Build (recursive) instance layout.
    if (auto inst = dyn_cast<InstOp>(op)) {
      // Skip self-recursion.
      if (inst.callee() == child.getName())
        return;

      if (auto sym =
            op->getParentOfType<ModuleOp>().lookupSymbol(inst.callee())) {
        std::string instName = child.getBaseUnitName() + '.' + inst.name().str();
        std::string instPath = child.getPath() + "/" + inst.name().str();
        Instance newChild(instName, instPath, inst.callee().str(),
                          inst.getNumOperands());

        // Add instance arguments to sensitivity list. The first numArgs signals
        // in the sensitivity list represent the unit's arguments(inputs), while
        // the following ones represent the unit-defined signals(outputs).
        llvm::SmallVector<Value, 8> args;
        args.insert(args.end(), inst.inputs().begin(), inst.inputs().end());
        args.insert(args.end(), inst.outputs().begin(), inst.outputs().end());

        for (size_t i = 0, e = args.size(); i < e; ++i) {
          // The signal comes from an instance's argument.
          if (auto blockArg = args[i].dyn_cast<BlockArgument>()) {
            auto detail = child.getSignalDetail(blockArg.getArgNumber());
            detail.instIndex = i;
            newChild.pushSignalDetail(detail);
          } else if (auto sig = dyn_cast<SigOp>(args[i].getDefiningOp())) {
            // The signal comes from one of the instance's owned signals.
            auto it = std::find_if(
                child.getSensitivityList().begin(),
                child.getSensitivityList().end(),
                [&](SignalDetail &detail) {
                  auto& outSig = state->getSignal(detail.globalIndex);
                  return outSig.getName() == sig.name() &&
                         outSig.getOwner() == child.getName();
                });
            if (it != child.getSensitivityList().end()) {
              auto detail = *it;
              detail.instIndex = i;
              newChild.pushSignalDetail(detail);
            }
          }
        }

        // Recursively walk a new entity, otherwise it is a process and cannot
        // define new signals or instances.
        if (auto ent = dyn_cast<EntityOp>(sym)) {
          newChild.setType(Instance::Entity);
          walkEntity(ent, newChild);
        } else {
          newChild.setType(Instance::Proc);
        }

        // Store the created instance.
        state->pushInstance(newChild);
      }
    }
  });
}
