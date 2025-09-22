//===- InferMemories.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-infer-memories"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_INFERMEMORIES
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

namespace {
struct InferMemoriesPass
    : public arc::impl::InferMemoriesBase<InferMemoriesPass> {
  using InferMemoriesBase::InferMemoriesBase;

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

    // FIRRTL memories are currently underspecified. They have a single read
    // latency, but it is unclear where within this latency the read of the
    // underlying storage happens. The `HWMemSimImpl` pass implements memories
    // such that the storage is probed at the very end of the latency, and that
    // the probed value becomes available immediately. We keep the latencies
    // configurable here, in the hopes that we'll improve our memory
    // abstractions at some point.
    unsigned readPreLatency = readLatency; // cycles before storage is read
    unsigned readPostLatency = 0; // cycles to move read value to output

    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    auto wordType = builder.getIntegerType(width);
    auto addressTy = dyn_cast<IntegerType>(instOp.getOperand(0).getType());
    if (!addressTy) {
      instOp.emitError("expected integer type for memory addressing, got ")
          << addressTy;
      return signalPassFailure();
    }
    auto memType = MemoryType::get(&getContext(), depth, wordType, addressTy);
    auto memOp = MemoryOp::create(builder, memType);
    if (tapMemories && !instOp.getInstanceName().empty())
      memOp->setAttr("name", instOp.getInstanceNameAttr());

    unsigned argIdx = 0;
    unsigned resultIdx = 0;

    auto applyLatency = [&](Value clock, Value data, unsigned latency) {
      for (unsigned i = 0; i < latency; ++i)
        data = seq::CompRegOp::create(builder, data, clock,
                                      builder.getStringAttr(""), Value{},
                                      Value{}, Value{}, hw::InnerSymAttr{});
      return data;
    };

    SmallVector<std::tuple<Value, Value, SmallVector<Value>, bool, bool>>
        writePorts;

    // Use `<inst-name>/` as the prefix for all port taps.
    SmallString<64> tapPrefix(instOp.getInstanceName());
    if (!tapPrefix.empty())
      tapPrefix.push_back('/');
    auto tapPrefixBaseLen = tapPrefix.size();

    auto tap = [&](Value value, const Twine &name) {
      auto prefixedName = builder.getStringAttr(tapPrefix + "_" + name);
      arc::TapOp::create(builder, value, prefixedName);
    };

    // Handle read ports.
    auto numReadPorts =
        params.getAs<IntegerAttr>("numReadPorts").getValue().getZExtValue();
    for (unsigned portIdx = 0; portIdx != numReadPorts; ++portIdx) {
      auto address = instOp.getOperand(argIdx++);
      auto enable = instOp.getOperand(argIdx++);
      auto clock = instOp.getOperand(argIdx++);
      auto data = instOp.getResult(resultIdx++);

      if (address.getType() != addressTy) {
        instOp.emitOpError("expected ")
            << addressTy << ", but got " << address.getType();
        return signalPassFailure();
      }

      // Add port taps.
      if (tapPorts) {
        tapPrefix.resize(tapPrefixBaseLen);
        (Twine("R") + Twine(portIdx)).toVector(tapPrefix);
        tap(address, "addr");
        tap(enable, "en");
        tap(data, "data");
      }

      // Apply the latency before the underlying storage is accessed.
      address = applyLatency(clock, address, readPreLatency);
      enable = applyLatency(clock, enable, readPreLatency);

      // Read the underlying storage. (The result of a disabled read port is
      // undefined, currently we define it to be zero.)
      Value readOp =
          MemoryReadPortOp::create(builder, wordType, memOp, address);
      Value zero = hw::ConstantOp::create(builder, wordType, 0);
      readOp = comb::MuxOp::create(builder, enable, readOp, zero);

      // Apply the latency after the underlying storage was accessed. (If the
      // latency is 0, the memory read is combinatorial without any buffer.)
      readOp = applyLatency(clock, readOp, readPostLatency);
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

      // Add port taps.
      if (tapPorts) {
        tapPrefix.resize(tapPrefixBaseLen);
        (Twine("RW") + Twine(portIdx)).toVector(tapPrefix);
        tap(address, "addr");
        tap(enable, "en");
        tap(writeMode, "wmode");
        tap(writeData, "wdata");
        if (writeMask)
          tap(writeMask, "wmask");
        tap(readData, "rdata");
      }

      auto c1_i1 = hw::ConstantOp::create(builder, builder.getI1Type(), 1);
      auto notWriteMode = comb::XorOp::create(builder, writeMode, c1_i1);
      Value readEnable = comb::AndOp::create(builder, enable, notWriteMode);

      // Apply the latency before the underlying storage is accessed.
      Value readAddress = applyLatency(clock, address, readPreLatency);
      readEnable = applyLatency(clock, readEnable, readPreLatency);

      // Read the underlying storage. (The result of a disabled read port is
      // undefined, currently we define it to be zero.)
      Value readOp =
          MemoryReadPortOp::create(builder, wordType, memOp, readAddress);
      Value zero = hw::ConstantOp::create(builder, wordType, 0);
      readOp = comb::MuxOp::create(builder, readEnable, readOp, zero);

      if (writeMask) {
        unsigned maskWidth = cast<IntegerType>(writeMask.getType()).getWidth();
        SmallVector<Value> toConcat;
        for (unsigned i = 0; i < maskWidth; ++i) {
          Value bit = comb::ExtractOp::create(builder, writeMask, i, 1);
          Value replicated = comb::ReplicateOp::create(builder, bit, maskGran);
          toConcat.push_back(replicated);
        }
        std::reverse(toConcat.begin(), toConcat.end()); // I hate concat
        writeMask =
            comb::ConcatOp::create(builder, writeData.getType(), toConcat);
      }

      // Apply the latency after the underlying storage was accessed. (If the
      // latency is 0, the memory read is combinatorial without any buffer.)
      readOp = applyLatency(clock, readOp, readPostLatency);
      readData.replaceAllUsesWith(readOp);

      auto writeEnable = comb::AndOp::create(builder, enable, writeMode);
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

      // Add port taps.
      if (tapPorts) {
        tapPrefix.resize(tapPrefixBaseLen);
        (Twine("W") + Twine(portIdx)).toVector(tapPrefix);
        tap(address, "addr");
        tap(enable, "en");
        tap(data, "data");
        if (mask)
          tap(mask, "mask");
      }

      if (mask) {
        unsigned maskWidth = cast<IntegerType>(mask.getType()).getWidth();
        SmallVector<Value> toConcat;
        for (unsigned i = 0; i < maskWidth; ++i) {
          Value bit = comb::ExtractOp::create(builder, mask, i, 1);
          Value replicated = comb::ReplicateOp::create(builder, bit, maskGran);
          toConcat.push_back(replicated);
        }
        std::reverse(toConcat.begin(), toConcat.end()); // I hate concat
        mask = comb::ConcatOp::create(builder, data.getType(), toConcat);
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
      auto defOp = DefineOp::create(builder, names.newName("mem_write"),
                                    builder.getFunctionType(types, types));
      auto &block = defOp.getBody().emplaceBlock();
      auto args = block.addArguments(
          types, SmallVector<Location>(types.size(), builder.getLoc()));
      builder.setInsertionPointToEnd(&block);
      arc::OutputOp::create(builder, SmallVector<Value>(args));
      builder.restoreInsertionPoint(ipSave);
      MemoryWritePortOp::create(builder, memOp, defOp.getName(), inputs, clock,
                                hasEnable, hasMask);
    }

    opsToDelete.push_back(instOp);
  });
  LLVM_DEBUG(llvm::dbgs() << "Inferred " << numReplaced << " memories\n");

  for (auto *op : opsToDelete)
    op->erase();
}

std::unique_ptr<Pass>
arc::createInferMemoriesPass(const InferMemoriesOptions &options) {
  return std::make_unique<InferMemoriesPass>(options);
}
