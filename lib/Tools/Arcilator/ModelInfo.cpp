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

#include "circt/Tools/Arcilator/ModelInfo.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace arcilator;

LogicalResult circt::arcilator::collectStates(Value storage, unsigned offset,
                                              std::vector<StateInfo> &states) {
  for (auto *op : storage.getUsers()) {
    if (auto substorage = dyn_cast<AllocStorageOp>(op)) {
      if (!substorage.getOffset().has_value())
        return substorage.emitOpError(
            "without allocated offset; run state allocation first");
      if (failed(collectStates(substorage.getOutput(),
                               *substorage.getOffset() + offset, states)))
        return failure();
      continue;
    }
    if (!isa<AllocStateOp, RootInputOp, RootOutputOp, AllocMemoryOp>(op))
      continue;
    auto opName = op->getAttrOfType<StringAttr>("name");
    if (!opName || opName.getValue().empty())
      continue;
    auto opOffset = op->getAttrOfType<IntegerAttr>("offset");
    if (!opOffset)
      return op->emitOpError(
          "without allocated offset; run state allocation first");
    if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
      auto result = op->getResult(0);
      auto &stateInfo = states.emplace_back();
      stateInfo.type = StateInfo::Register;
      if (isa<RootInputOp>(op))
        stateInfo.type = StateInfo::Input;
      else if (isa<RootOutputOp>(op))
        stateInfo.type = StateInfo::Output;
      else if (auto alloc = dyn_cast<AllocStateOp>(op)) {
        if (alloc.getTap())
          stateInfo.type = StateInfo::Wire;
      }
      stateInfo.name = opName.getValue();
      stateInfo.offset = opOffset.getValue().getZExtValue() + offset;
      stateInfo.numBits = result.getType().cast<StateType>().getBitWidth();
      continue;
    }
    if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
      auto stride = op->getAttrOfType<IntegerAttr>("stride");
      if (!stride)
        return op->emitOpError(
            "without allocated stride; run state allocation first");
      auto memType = memOp.getType();
      auto intType = memType.getWordType();
      auto &stateInfo = states.emplace_back();
      stateInfo.type = StateInfo::Memory;
      stateInfo.name = opName.getValue();
      stateInfo.offset = opOffset.getValue().getZExtValue() + offset;
      stateInfo.numBits = intType.getWidth();
      stateInfo.memoryStride = stride.getValue().getZExtValue();
      stateInfo.memoryDepth = memType.getNumWords();
      continue;
    }
  }
  return success();
}

LogicalResult circt::arcilator::collectModels(mlir::ModuleOp module,
                                              std::vector<ModelInfo> &models) {
  for (auto modelOp : module.getOps<ModelOp>()) {
    auto storageArg = modelOp.getBody().getArgument(0);
    auto storageType = storageArg.getType().cast<StorageType>();

    std::vector<StateInfo> states;
    if (failed(collectStates(storageArg, 0, states)))
      return failure();
    llvm::sort(states, [](auto &a, auto &b) { return a.offset < b.offset; });

    models.emplace_back(std::string(modelOp.getName()), storageType.getSize(),
                        std::move(states));
  }

  return success();
}

void circt::arcilator::serializeModelInfoToJson(
    llvm::raw_ostream &outputStream, std::vector<ModelInfo> &models) {
  llvm::json::OStream json(outputStream, 2);

  json.array([&] {
    for (ModelInfo &model : models) {
      json.object([&] {
        json.attribute("name", model.name);
        json.attribute("numStateBytes", model.numStateBytes);
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
