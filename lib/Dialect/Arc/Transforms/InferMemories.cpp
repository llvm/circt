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
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
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

  SymbolCache cache;
  cache.addDefinitions(module);
  Namespace names;
  names.add(cache);

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
    auto addressTy = dyn_cast<IntegerType>(instOp.getOperand(0).getType());
    if (!addressTy) {
      instOp.emitError("expected integer type for memory addressing, got ")
          << addressTy;
      return signalPassFailure();
    }
    auto memType = MemoryType::get(&getContext(), depth, wordType, addressTy);
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

    SmallVector<std::tuple<Value, Value, SmallVector<Value>, bool, bool>>
        writePorts;

    // Handle read ports.
    auto numReadPorts =
        params.getAs<IntegerAttr>("numReadPorts").getValue().getZExtValue();
    for (unsigned portIdx = 0; portIdx != numReadPorts; ++portIdx) {
      auto address = instOp.getOperand(argIdx++);
      ++argIdx; // skip enable argument
      auto clock = instOp.getOperand(argIdx++);
      auto data = instOp.getResult(resultIdx++);

      if (address.getType() != addressTy) {
        instOp.emitOpError("expected ")
            << addressTy << ", but got " << address.getType();
        return signalPassFailure();
      }

      // NOTE: the result of a disabled read port is undefined, currently we
      // define it to be the same as if it was enabled, but we could also set it
      // to any constant (e.g., by inserting a mux).
      Value readOp = builder.create<MemoryReadPortOp>(wordType, memOp, address);
      // NOTE: if the read-latency is 0, the memory read is combinatorial
      // without any buffer
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

      if (address.getType() != addressTy) {
        instOp.emitOpError("expected ")
            << addressTy << ", but got " << address.getType();
        return signalPassFailure();
      }

      // NOTE: the result of a disabled read port is undefined, currently we
      // define it to be the same as if it was enabled, but we could also set it
      // to any constant (e.g., by inserting a mux).
      Value readOp = builder.create<MemoryReadPortOp>(wordType, memOp, address);

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

      // NOTE: if the read-latency is 0, the memory read is combinatorial
      // without any buffer
      readOp = applyReadLatency(clock, readOp);
      readData.replaceAllUsesWith(readOp);

      auto writeEnable = builder.create<comb::AndOp>(enable, writeMode);
      SmallVector<Value> inputs({address, writeData, writeEnable});
      if (writeMask)
        inputs.push_back(writeMask);
      writePorts.push_back({memOp, clock, inputs, true, !!writeMask});
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

      if (address.getType() != addressTy) {
        instOp.emitOpError("expected ")
            << addressTy << ", but got " << address.getType();
        return signalPassFailure();
      }

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
      SmallVector<Value> inputs({address, data});
      if (enable)
        inputs.push_back(enable);
      if (mask)
        inputs.push_back(mask);
      writePorts.push_back({memOp, clock, inputs, !!enable, !!mask});
    }

    // Create the actual write ports with a dependency arc to all read
    // ports.
    for (auto [memOp, clock, inputs, hasEnable, hasMask] : writePorts) {
      auto ipSave = builder.saveInsertionPoint();
      TypeRange types = ValueRange(inputs).getTypes();
      builder.setInsertionPointToStart(module.getBody());
      auto defOp = builder.create<DefineOp>(
          names.newName("mem_write"), builder.getFunctionType(types, types));
      auto &block = defOp.getBody().emplaceBlock();
      auto args = block.addArguments(
          types, SmallVector<Location>(types.size(), builder.getLoc()));
      builder.setInsertionPointToEnd(&block);
      builder.create<arc::OutputOp>(SmallVector<Value>(args));
      builder.restoreInsertionPoint(ipSave);
      builder.create<MemoryWritePortOp>(memOp, defOp.getName(), inputs, clock,
                                        hasEnable, hasMask);
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
