//===- NormalizeMemory.cpp - Normalizes memory ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Transform memory types and compute masks.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OperationSupport.h"

using namespace circt;
using namespace firrtl;

namespace {
struct NormalizeMemoryPass
    : public NoramlizeMemoryPassBase<NormalizeMemoryPass> {
  void runOnMemory(MemOp mem);
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (auto mem = dyn_cast<MemOp>(op))
        runOnMemory(mem);
    });
  }
};
} // end anonymous namespace

void NormalizeMemoryPass::runOnMemory(MemOp mem) {
  OpBuilder builder(&getContext());

  auto memTy = mem.getDataType().dyn_cast<FVectorType>();
  if (!memTy || memTy.getBitWidthOrSentinel() <= 0)
    return;

  size_t memFlatWidth = memTy.getBitWidthOrSentinel();

    SmallVector<WireOp> oldPorts;

  // Wires for old ports
  for (unsigned int index = 0, end = mem.getNumResults(); index < end; ++index) {
    auto result = mem.getResult(index);
    auto wire = builder.create<WireOp>(mem.getLoc(), 
        result.getType(),
        (mem.name() + "_" + mem.getPortName(index).getValue()).str());
    oldPorts.push_back(wire);
    result.replaceAllUsesWith(wire.getResult());
  }

  // Now create a new memory of type flattened data.
  // ----------------------------------------------
  SmallVector<Type, 8> ports;
  SmallVector<Attribute, 8> portNames;

  // Create a new memoty data type of unsigned and computed width.
  auto flatType = UIntType::get(&getContext(), memFlatWidth);
  auto opPorts = mem.getPorts();
  for (size_t portIdx = 0, e = opPorts.size(); portIdx < e; ++portIdx) {
    auto port = opPorts[portIdx];
    ports.push_back(MemOp::getTypeForPort(mem.depth(), flatType, port.second,
                                          mem.getMaskBits()));
    portNames.push_back(port.first);
  }

  auto flatMem = builder.create<MemOp>(
      mem.getLoc(), ports, mem.readLatency(), mem.writeLatency(), mem.depth(),
      mem.ruw(), portNames, mem.name(), mem.annotations().getValue(),
      mem.portAnnotations().getValue(), mem.inner_symAttr());


  // Hook up the new memories to the wires the old memory was replaced with.
  for (size_t index = 0, rend = op.getNumResults(); index < rend; ++index) {
    auto result = oldPorts[index];
    auto rType = result.getType().cast<BundleType>();
    for (size_t fieldIndex = 0, fend = rType.getNumElements();
         fieldIndex != fend; ++fieldIndex) {
      auto name = rType.getElement(fieldIndex).name.getValue();
      auto oldField = builder->create<SubfieldOp>(result, fieldIndex);
      // data and mask depend on the memory type which was split.  They can also
      // go both directions, depending on the port direction.
      if (name == "data" || name == "mask" || name == "wdata" ||
          name == "wmask" || name == "rdata") {
        if (localFlattenAggregateMemData) {
          // If memory was flattened instead of one memory per aggregate field.
          Value newField =
              getSubWhatever(newMemories[0].getResult(index), fieldIndex);
          Value realOldField = oldField;
          if (rType.getElement(fieldIndex).isFlip) {
            // Cast the memory read data from flat type to aggregate.
            newField = builder->createOrFold<BitCastOp>(
                oldField.getType().cast<FIRRTLType>(), newField);
            // Write the aggregate read data.
            mkConnect(builder, realOldField, newField);
          } else {
            // Cast the input aggregate write data to flat type.
            realOldField = builder->create<BitCastOp>(
                newField.getType().cast<FIRRTLType>(), oldField);
            // Mask bits require special handling, since some of the mask bits
            // need to be repeated, direct bitcasting wouldn't work. Depending
            // on the mask granularity, some mask bits will be repeated.
            if ((name == "mask" || name == "wmask") &&
                (maskWidths.size() < totalmaskWidths)) {
              Value catMasks;
              for (auto m : llvm::enumerate(maskWidths)) {
                // Get the mask bit.
                auto mBit = builder->createOrFold<BitsPrimOp>(
                    realOldField, m.index(), m.index());
                // Check how many times the mask bit needs to be prepend.
                for (size_t repeat = 0; repeat < m.value(); repeat++)
                  if (m.index() == 0 && repeat == 0)
                    catMasks = mBit;
                  else
                    catMasks = builder->createOrFold<CatPrimOp>(mBit, catMasks);
              }
              realOldField = catMasks;
            }
            // Now set the mask or write data.
            // Ensure that the types match.
            mkConnect(builder, newField,
                      builder->createOrFold<BitCastOp>(
                          newField.getType().cast<FIRRTLType>(), realOldField));
          }
        } else {
          for (auto field : fields) {
            auto realOldField = getSubWhatever(oldField, field.index);
            auto newField = getSubWhatever(
                newMemories[field.index].getResult(index), fieldIndex);
            if (rType.getElement(fieldIndex).isFlip)
              std::swap(realOldField, newField);
            mkConnect(builder, newField, realOldField);
          }
        }
      } else {
        for (auto mem : newMemories) {
          auto newField =
              builder->create<SubfieldOp>(mem.getResult(index), fieldIndex);
          mkConnect(builder, newField, oldField);
        }
      }
    }
  }


}

std::unique_ptr<mlir::Pass> circt::firrtl::createNormalizeMemoryPass() {
  return std::make_unique<NormalizeMemoryPass>();
}
