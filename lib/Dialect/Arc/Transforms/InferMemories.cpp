//===- InferMemories.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-infer-memories"

using namespace circt;
using namespace arc;

namespace {
struct InferMemoriesPass : public InferMemoriesBase<InferMemoriesPass> {
  void runOnOperation() override;
  SmallVector<Operation *> opsToDelete;
  SmallPtrSet<StringAttr, 2> schemaNames;
  DenseMap<StringAttr, DictionaryAttr> memoryParams;
};
} // namespace

void InferMemoriesPass::runOnOperation() {
  auto module = getOperation();
  opsToDelete.clear();
  schemaNames.clear();
  memoryParams.clear();

  // Find the matching generator schemas.
  for (auto schemaOp : module.getOps<hw::HWGeneratorSchemaOp>()) {
    if (schemaOp.getDescriptor() == "FIRRTL_Memory") {
      schemaNames.insert(schemaOp.getSymNameAttr());
      opsToDelete.push_back(schemaOp);
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Found " << schemaNames.size() << " schemas\n");

  // Find generated ops using these schemas.
  for (auto genOp : module.getOps<hw::HWModuleGeneratedOp>()) {
    if (!schemaNames.contains(genOp.getGeneratorKindAttr().getAttr()))
      continue;
    memoryParams[genOp.getModuleNameAttr()] = genOp->getAttrDictionary();
    opsToDelete.push_back(genOp);
  }
  LLVM_DEBUG(llvm::dbgs() << "Found " << memoryParams.size()
                          << " memory modules\n");

  // Convert instances of the generated ops into dedicated memories.
  unsigned numReplaced = 0;
  module.walk([&](hw::InstanceOp instOp) {
    auto it = memoryParams.find(instOp.getModuleNameAttr().getAttr());
    if (it == memoryParams.end())
      return;
    ++numReplaced;
    DictionaryAttr params = it->second;
    auto width = params.getAs<IntegerAttr>("width").getValue().getZExtValue();
    auto depth = params.getAs<IntegerAttr>("depth").getValue().getZExtValue();
    auto maskGranAttr = params.getAs<IntegerAttr>("maskGran");
    auto maskGran =
        maskGranAttr ? maskGranAttr.getValue().getZExtValue() : width;
    auto maskBits = width / maskGran;

    auto writeLatency =
        params.getAs<IntegerAttr>("writeLatency").getValue().getZExtValue();
    auto readLatency =
        params.getAs<IntegerAttr>("readLatency").getValue().getZExtValue();
    if (writeLatency != 1) {
      instOp.emitError("unsupported memory write latency ") << writeLatency;
      return signalPassFailure();
    }

    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    auto wordType = builder.getIntegerType(width);
    auto memType = MemoryType::get(&getContext(), depth, wordType, {});
    auto memOp = builder.create<MemoryOp>(memType);
    if (!instOp.getInstanceName().empty())
      memOp->setAttr("name", instOp.getInstanceNameAttr());

    unsigned argIdx = 0;
    unsigned resultIdx = 0;

    auto applyReadLatency = [&](Value clock, Value data) {
      for (unsigned i = 0; i < readLatency; ++i)
        data = builder.create<seq::CompRegOp>(data, clock, "mem_read_latency");
      return data;
    };

    SmallVector<std::array<Value, 6>> writePorts;

    // Handle read ports.
    auto numReadPorts =
        params.getAs<IntegerAttr>("numReadPorts").getValue().getZExtValue();
    for (unsigned portIdx = 0; portIdx != numReadPorts; ++portIdx) {
      auto address = instOp.getOperand(argIdx++);
      auto enable = instOp.getOperand(argIdx++);
      auto clock = instOp.getOperand(argIdx++);
      auto data = instOp.getResult(resultIdx++);
      Value readOp = builder.create<MemoryReadPortOp>(wordType, memOp, address,
                                                      clock, enable);
      readOp = applyReadLatency(clock, readOp);
      data.replaceAllUsesWith(readOp);
    }

    // Handle read-write ports.
    auto numReadWritePorts = params.getAs<IntegerAttr>("numReadWritePorts")
                                 .getValue()
                                 .getZExtValue();
    for (unsigned portIdx = 0; portIdx != numReadWritePorts; ++portIdx) {
      auto address = instOp.getOperand(argIdx++);
      auto enable = instOp.getOperand(argIdx++);
      auto clock = instOp.getOperand(argIdx++);
      auto writeMode = instOp.getOperand(argIdx++);
      auto writeData = instOp.getOperand(argIdx++);
      auto writeMask = maskBits > 1 ? instOp.getOperand(argIdx++) : Value{};
      auto readData = instOp.getResult(resultIdx++);

      auto constOne = builder.create<hw::ConstantOp>(builder.getI1Type(), 1);
      auto readMode = builder.create<comb::XorOp>(writeMode, constOne);
      auto readEnable = builder.create<comb::AndOp>(enable, readMode);
      Value readOp = builder.create<MemoryReadPortOp>(wordType, memOp, address,
                                                      clock, readEnable);

      if (writeMask) {
        unsigned maskWidth = writeMask.getType().cast<IntegerType>().getWidth();
        SmallVector<Value> toConcat;
        for (unsigned i = 0; i < maskWidth; ++i) {
          Value bit = builder.create<comb::ExtractOp>(writeMask, i, 1);
          Value replicated = builder.create<comb::ReplicateOp>(bit, maskGran);
          toConcat.push_back(replicated);
        }
        writeMask =
            builder.create<comb::ConcatOp>(writeData.getType(), toConcat);
      }

      readOp = applyReadLatency(clock, readOp);
      readData.replaceAllUsesWith(readOp);

      auto writeEnable = builder.create<comb::AndOp>(enable, writeMode);
      writePorts.push_back(
          {memOp, address, clock, writeEnable, writeData, writeMask});
    }

    // Handle write ports.
    auto numWritePorts =
        params.getAs<IntegerAttr>("numWritePorts").getValue().getZExtValue();
    for (unsigned portIdx = 0; portIdx != numWritePorts; ++portIdx) {
      auto address = instOp.getOperand(argIdx++);
      auto enable = instOp.getOperand(argIdx++);
      auto clock = instOp.getOperand(argIdx++);
      auto data = instOp.getOperand(argIdx++);
      auto mask = maskBits > 1 ? instOp.getOperand(argIdx++) : Value{};

      if (mask) {
        unsigned maskWidth = mask.getType().cast<IntegerType>().getWidth();
        SmallVector<Value> toConcat;
        for (unsigned i = 0; i < maskWidth; ++i) {
          Value bit = builder.create<comb::ExtractOp>(mask, i, 1);
          Value replicated = builder.create<comb::ReplicateOp>(bit, maskGran);
          toConcat.push_back(replicated);
        }
        mask = builder.create<comb::ConcatOp>(data.getType(), toConcat);
      }

      writePorts.push_back({memOp, address, clock, enable, data, mask});
    }

    // Create the actual write ports with a dependency arc to all read
    // ports.
    for (auto [memOp, address, clock, enable, data, mask] : writePorts) {
      builder.create<MemoryWritePortOp>(memOp, address, clock, enable, data,
                                        mask);
    }

    opsToDelete.push_back(instOp);
  });
  LLVM_DEBUG(llvm::dbgs() << "Inferred " << numReplaced << " memories\n");

  for (auto *op : opsToDelete)
    op->erase();
}

std::unique_ptr<Pass> arc::createInferMemoriesPass() {
  return std::make_unique<InferMemoriesPass>();
}
