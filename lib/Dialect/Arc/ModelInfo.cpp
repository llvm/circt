//===- ModelInfo.cpp - Information about Arc models -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines and computes information about Arc models.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace circt;
using namespace arc;

LogicalResult circt::arc::collectStates(Value storage, unsigned offset,
                                        SmallVector<StateInfo> &states) {
  struct StateCollectionJob {
    mlir::Value::user_iterator nextToProcess;
    mlir::Value::user_iterator end;
    unsigned offset;

    StateCollectionJob(Value storage, unsigned offset)
        : nextToProcess(storage.user_begin()), end(storage.user_end()),
          offset(offset) {}
  };

  SmallVector<StateCollectionJob, 4> jobStack{{storage, offset}};

  while (!jobStack.empty()) {
    StateCollectionJob &job = jobStack.back();

    if (job.nextToProcess == job.end) {
      jobStack.pop_back();
      continue;
    }

    Operation *op = *job.nextToProcess++;
    unsigned offset = job.offset;

    if (auto substorage = dyn_cast<AllocStorageOp>(op)) {
      if (!substorage.getOffset().has_value())
        return substorage.emitOpError(
            "without allocated offset; run state allocation first");
      Value substorageOutput = substorage.getOutput();
      jobStack.emplace_back(substorageOutput, offset + *substorage.getOffset());
      continue;
    }

    if (!isa<AllocStateOp, RootInputOp, RootOutputOp, AllocMemoryOp>(op))
      continue;

    SmallVector<StringAttr> names;

    auto opName = op->getAttrOfType<StringAttr>("name");
    if (opName && !opName.getValue().empty())
      names.push_back(opName);

    if (auto nameAttrs = op->getAttrOfType<ArrayAttr>("names"))
      for (auto attr : nameAttrs)
        if (auto nameAttr = dyn_cast<StringAttr>(attr))
          if (!nameAttr.empty())
            names.push_back(nameAttr);

    if (names.empty())
      continue;

    auto opOffset = op->getAttrOfType<IntegerAttr>("offset");
    if (!opOffset)
      return op->emitOpError(
          "without allocated offset; run state allocation first");

    StateInfo stateInfo;
    if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
      auto result = op->getResult(0);
      stateInfo.type = StateInfo::Register;
      if (isa<RootInputOp>(op))
        stateInfo.type = StateInfo::Input;
      else if (isa<RootOutputOp>(op))
        stateInfo.type = StateInfo::Output;
      else if (auto alloc = dyn_cast<AllocStateOp>(op)) {
        if (alloc.getTap())
          stateInfo.type = StateInfo::Wire;
      }
      stateInfo.offset = opOffset.getValue().getZExtValue() + offset;
      stateInfo.numBits = cast<StateType>(result.getType()).getBitWidth();
      for (auto name : names) {
        stateInfo.name = name.getValue();
        states.push_back(stateInfo);
      }
      continue;
    }

    if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
      auto stride = op->getAttrOfType<IntegerAttr>("stride");
      if (!stride)
        return op->emitOpError(
            "without allocated stride; run state allocation first");
      auto memType = memOp.getType();
      auto intType = memType.getWordType();
      stateInfo.type = StateInfo::Memory;
      stateInfo.offset = opOffset.getValue().getZExtValue() + offset;
      stateInfo.numBits = intType.getWidth();
      stateInfo.memoryStride = stride.getValue().getZExtValue();
      stateInfo.memoryDepth = memType.getNumWords();
      for (auto name : names) {
        stateInfo.name = name.getValue();
        states.push_back(stateInfo);
      }
      continue;
    }
  }

  return success();
}

LogicalResult circt::arc::collectModels(mlir::ModuleOp module,
                                        SmallVector<ModelInfo> &models) {

  for (auto modelOp : module.getOps<ModelOp>()) {
    auto storageArg = modelOp.getBody().getArgument(0);
    auto storageType = cast<StorageType>(storageArg.getType());

    SmallVector<StateInfo> states;
    if (failed(collectStates(storageArg, 0, states)))
      return failure();
    llvm::sort(states, [](auto &a, auto &b) { return a.offset < b.offset; });

    models.emplace_back(std::string(modelOp.getName()), storageType.getSize(),
                        std::move(states), modelOp.getInitialFnAttr(),
                        modelOp.getFinalFnAttr());
  }

  return success();
}

void circt::arc::serializeModelInfoToJson(llvm::raw_ostream &outputStream,
                                          ArrayRef<ModelInfo> models) {
  llvm::json::OStream json(outputStream, 2);

  json.array([&] {
    for (const ModelInfo &model : models) {
      json.object([&] {
        json.attribute("name", model.name);
        json.attribute("numStateBytes", model.numStateBytes);
        json.attribute("initialFnSym", !model.initialFnSym
                                           ? ""
                                           : model.initialFnSym.getValue());
        json.attribute("finalFnSym",
                       !model.finalFnSym ? "" : model.finalFnSym.getValue());
        json.attributeArray("states", [&] {
          for (const auto &state : model.states) {
            json.object([&] {
              json.attribute("name", state.name);
              json.attribute("offset", state.offset);
              json.attribute("numBits", state.numBits);
              auto typeStr = [](StateInfo::Type type) {
                switch (type) {
                case StateInfo::Input:
                  return "input";
                case StateInfo::Output:
                  return "output";
                case StateInfo::Register:
                  return "register";
                case StateInfo::Memory:
                  return "memory";
                case StateInfo::Wire:
                  return "wire";
                }
                return "";
              };
              json.attribute("type", typeStr(state.type));
              if (state.type == StateInfo::Memory) {
                json.attribute("stride", state.memoryStride);
                json.attribute("depth", state.memoryDepth);
              }
            });
          }
        });
      });
    }
  });
}
