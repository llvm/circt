//===- Engine.cpp - Simulator Engine class implementation -------*- C++ -*-===//
//
// This file implements the Engine class.
//
//===----------------------------------------------------------------------===//

#include "State.h"

#include "circt/Conversion/LLHDToLLVM/LLHDToLLVM.h"
#include "circt/Dialect/LLHD/Simulator/Engine.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;
using namespace llhd::sim;

Engine::Engine(llvm::raw_ostream &out, ModuleOp module, MLIRContext &context,
               std::string root)
    : out(out), root(root) {
  state = std::make_unique<State>();

  buildLayout(module);

  auto rootEntity = module.lookupSymbol<EntityOp>(root);

  // Insert explicit instantiation of the design root.
  OpBuilder insertInst =
      OpBuilder::atBlockTerminator(&rootEntity.getBody().getBlocks().front());
  insertInst.create<InstOp>(rootEntity.getBlocks().front().back().getLoc(),
                            llvm::None, root, root, ArrayRef<Value>(),
                            ArrayRef<Value>());

  // Add the 0-time event.
  state->queue.push(Slot(Time()));

  mlir::PassManager pm(&context);
  pm.addPass(llhd::createConvertLLHDToLLVMPass());
  if (failed(pm.run(module))) {
    llvm::errs() << "failed to convert module to LLVM";
    exit(EXIT_FAILURE);
  }

  this->module = module;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto maybeEngine = mlir::ExecutionEngine::create(this->module);
  assert(maybeEngine && "failed to create JIT");
  engine = std::move(*maybeEngine);
}

Engine::~Engine() = default;

void Engine::dumpStateLayout() { state->dumpLayout(); }

void Engine::dumpStateSignalTriggers() { state->dumpSignalTriggers(); }

int Engine::simulate(int n) {
  assert(engine && "engine not found");
  assert(state && "state not found");

  SmallVector<void *, 1> arg({&state});
  // Initialize tbe simulation state.
  auto invocationResult = engine->invoke("llhd_init", arg);
  if (invocationResult) {
    llvm::errs() << "Failed invocation of llhd_init: " << invocationResult;
    return -1;
  }

  // Dump the signals' initial values.
  for (int i = 0; i < state->nSigs; ++i) {
    state->dumpSignal(out, i);
  }

  int i = 0;

  // Keep track of the instances that need to wakeup.
  std::vector<std::string> wakeupQueue;
  // All instances are run in the first cycle.
  for (auto k : state->instances.keys())
    wakeupQueue.push_back(k.str());

  while (!state->queue.empty()) {
    if (n > 0 && i >= n) {
      break;
    }
    auto pop = state->popQueue();

    // Update the simulation time.
    assert(state->time < pop.time || pop.time.time == 0);
    state->time = pop.time;

    // Apply the signal changes and dump the signals that actually changed
    // value.
    for (auto change : pop.changes) {
      // Get a buffer to apply the changes on.
      Signal *curr = &(state->signals[change.first]);
      APInt buff(
          curr->size * 8,
          ArrayRef<uint64_t>(reinterpret_cast<uint64_t *>(curr->detail.value),
                             curr->size));

      // Apply all the changes to the buffer, in order of execution.
      for (auto drive : change.second) {
        if (drive.second.getBitWidth() < buff.getBitWidth())
          buff.insertBits(drive.second, drive.first);
        else
          buff = drive.second;
      }

      // Skip if the updated signal value is equal to the initial value.
      if (std::memcmp(curr->detail.value, buff.getRawData(), curr->size) == 0)
        continue;

      // Apply the signal update.
      std::memcpy(curr->detail.value, buff.getRawData(),
                  state->signals[change.first].size);

      // Trigger all sensitive instances.
      // The owner is always triggered.
      wakeupQueue.push_back(state->signals[change.first].owner);
      // Add sensitive instances.
      for (auto inst : state->signals[change.first].triggers) {
        // Skip if the process is not currently sensible to the signal.
        if (!state->instances[inst].isEntity) {
          auto &sensList = state->instances[inst].sensitivityList;
          auto it = std::find(sensList.begin(), sensList.end(), change.first);
          if (sensList.end() != it &&
              state->instances[inst].procState->senses[it - sensList.begin()] ==
                  0)
            continue;
        }
        wakeupQueue.push_back(inst);
      }

      // Dump the updated signal.
      state->dumpSignal(out, change.first);
    }

    // TODO: don't wakeup a process instances if already woken up by an observed
    // signal.
    // Add scheduled process resumes to the wakeup queue.
    for (auto inst : pop.scheduled) {
      wakeupQueue.push_back(inst);
    }

    // Clear temporary subsignals.
    state->signals.erase(state->signals.begin() + state->nSigs,
                         state->signals.end());

    // Run the instances present in the wakeup queue.
    for (auto inst : wakeupQueue) {
      auto name = state->instances[inst].unit;
      auto sigTable = state->instances[inst].signalTable.data();
      auto sensitivityList = state->instances[inst].sensitivityList;
      auto outputList = state->instances[inst].outputs;
      // Combine inputs and outputs in one argument table.
      sensitivityList.insert(sensitivityList.end(), outputList.begin(),
                             outputList.end());
      auto argTable = sensitivityList.data();

      // Gather the instance arguments for unit invocation.
      SmallVector<void *, 3> args;
      if (state->instances[inst].isEntity)
        args.assign({&state, &sigTable, &argTable});
      else {
        args.assign({&state, &state->instances[inst].procState, &argTable});
      }
      // Run the unit.
      auto invocationResult = engine->invoke(name, args);
      if (invocationResult) {
        llvm::errs() << "Failed invocation of " << root << ": "
                     << invocationResult;
        return -1;
      }
    }

    // Clear wakeup queue.
    wakeupQueue.clear();
    i++;
  }
  llvm::errs() << "Finished after " << i << " steps.\n";
  return 0;
}

void Engine::buildLayout(ModuleOp module) {
  // Start from the root entity.
  auto rootEntity = module.lookupSymbol<EntityOp>(root);
  assert(rootEntity && "root entity not found!");

  // Build root instance, the parent and instance names are the same for the
  // root.
  Instance rootInst(root, root);
  rootInst.unit = root;

  // Recursively walk the units starting at root.
  walkEntity(rootEntity, rootInst);

  // The root is always an instance.
  rootInst.isEntity = true;
  // Store the root instance.
  state->instances[rootInst.name] = rootInst;

  // Add triggers and outputs to all signals.
  for (auto &inst : state->instances) {
    for (auto trigger : inst.getValue().sensitivityList) {
      state->signals[trigger].triggers.push_back(inst.getKey().str());
    }
    for (auto out : inst.getValue().outputs) {
      state->signals[out].outOf.push_back(inst.getKey().str());
    }
  }
  state->nSigs = state->signals.size();
}

void Engine::walkEntity(EntityOp entity, Instance &child) {
  entity.walk([&](Operation *op) -> WalkResult {
    assert(op);

    // Add a signal to the signal table.
    if (auto sig = dyn_cast<SigOp>(op)) {
      int index = state->addSignal(sig.name().str(), child.name);
      child.signalTable.push_back(index);
    }

    // Build (recursive) instance layout.
    if (auto inst = dyn_cast<InstOp>(op)) {
      // Skip self-recursion.
      if (inst.callee() == child.name)
        return WalkResult::advance();
      if (auto e =
              op->getParentOfType<ModuleOp>().lookupSymbol(inst.callee())) {
        Instance newChild(inst.name().str(), child.name);
        newChild.unit = inst.callee().str();

        // Gather sensitivity list.
        for (auto arg : inst.inputs()) {
          // Check if the argument comes from a parent's argument.
          if (auto a = arg.dyn_cast<BlockArgument>()) {
            unsigned int argInd = a.getArgNumber();
            // The argument comes either from one of the parent's inputs, or one
            // of the parent's outputs.
            if (argInd < newChild.sensitivityList.size())
              newChild.sensitivityList.push_back(child.sensitivityList[argInd]);
            else
              newChild.sensitivityList.push_back(
                  child.outputs[argInd - newChild.sensitivityList.size()]);
          } else if (auto sig = dyn_cast<SigOp>(arg.getDefiningOp())) {
            // Otherwise has to come from a sigop, search through the intantce's
            // signal table.
            for (auto s : child.signalTable) {
              if (state->signals[s].name == sig.name() &&
                  state->signals[s].owner == child.name) {
                newChild.sensitivityList.push_back(s);
                break;
              }
            }
          }
        }

        // Gather outputs list.
        for (auto out : inst.outputs()) {
          // Check if it comes from an argument.
          if (auto a = out.dyn_cast<BlockArgument>()) {
            unsigned int argInd = a.getArgNumber();
            // The argument comes either from one of the parent's inputs or one
            // of the parent's outputs.
            if (argInd < newChild.sensitivityList.size())
              newChild.outputs.push_back(child.sensitivityList[argInd]);
            else
              newChild.outputs.push_back(
                  child.outputs[argInd - newChild.sensitivityList.size()]);
          } else if (auto sig = dyn_cast<SigOp>(out.getDefiningOp())) {
            // Search through the signal table.
            for (auto s : child.signalTable) {
              if (state->signals[s].name == sig.name() &&
                  state->signals[s].owner == child.name) {
                newChild.outputs.push_back(s);
                break;
              }
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
        state->instances[newChild.name] = newChild;
      }
    }
    return WalkResult::advance();
  });
}
