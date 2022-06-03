//===- FlattenMemroy.cpp - Flatten Memory Pass ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FlattenMemory pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-memory"

using namespace circt;
using namespace firrtl;

namespace {
struct FlattenMemoryPass : public FlattenMemoryBase<FlattenMemoryPass> {
  /// This pass flattens the aggregate data of memory into a UInt, and inserts
  /// appropriate bitcasts to access the data.
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "\n Running lower memory on module:"
                            << getOperation().getName());
    ModuleNamespace modNamespace(getOperation());
    SmallVector<Operation *> opsToErase;
    auto hasSubAnno = [&](MemOp op) -> bool {
      for (size_t portIdx = 0, e = op.getNumResults(); portIdx < e; ++portIdx)
        for (auto attr : op.getPortAnnotation(portIdx))
          if (attr.isa<SubAnnotationAttr>() ||
              attr.cast<DictionaryAttr>().get("circt.fieldID"))
            return true;

      return false;
    };
    getOperation().getBody()->walk([&](MemOp memOp) {
      LLVM_DEBUG(llvm::dbgs() << "\n Memory:" << memOp);
      // The vector of leaf elements type after flattening the data.
      SmallVector<IntType> flatMemType;
      // MaskGranularity : how many bits each mask bit controls.
      size_t maskGran = 1;
      // Total mask bitwidth after flattening.
      uint32_t totalmaskWidths = 0;
      // How many mask bits each field type requires.
      SmallVector<unsigned> maskWidths;

      // If subannotations present on aggregate fields, we cannot flatten the
      // memory. It must be split into one memory per aggregate field.
      // Do not overwrite the pass flag!
      if (hasSubAnno(memOp) || !flattenType(memOp.getDataType(), flatMemType))
        return;

      SmallVector<Operation *, 8> flatData;
      SmallVector<int32_t> memWidths;
      // Get the width of individual aggregate leaf elements.
      for (auto f : flatMemType) {
        LLVM_DEBUG(llvm::dbgs() << "\n field type:" << f);
        memWidths.push_back(f.getWidth().getValue());
      }
      maskGran = memWidths[0];
      size_t memFlatWidth = 0;
      // Compute the GCD of all data bitwidths.
      for (auto w : memWidths) {
        memFlatWidth += w;
        maskGran = llvm::GreatestCommonDivisor64(maskGran, w);
      }
      for (auto w : memWidths) {
        // How many mask bits required for each flattened field.
        auto mWidth = w / maskGran;
        maskWidths.push_back(mWidth);
        totalmaskWidths += mWidth;
      }
      // Now create a new memory of type flattened data.
      // ----------------------------------------------
      SmallVector<Type, 8> ports;
      SmallVector<Attribute, 8> portNames;

      auto *context = memOp.getContext();
      ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
      // Create a new memoty data type of unsigned and computed width.
      auto flatType = UIntType::get(context, memFlatWidth);
      auto opPorts = memOp.getPorts();
      for (size_t portIdx = 0, e = opPorts.size(); portIdx < e; ++portIdx) {
        auto port = opPorts[portIdx];
        ports.push_back(MemOp::getTypeForPort(memOp.depth(), flatType,
                                              port.second, totalmaskWidths));
        portNames.push_back(port.first);
      }

      auto flatMem = builder.create<MemOp>(
          ports, memOp.readLatency(), memOp.writeLatency(), memOp.depth(),
          memOp.ruw(), builder.getArrayAttr(portNames), memOp.nameAttr(),
          memOp.annotations(), memOp.portAnnotations(), memOp.inner_symAttr(),
          memOp.groupIDAttr(), inferNameKind(memOp.name()));
      // Hook up the new memory to the wires the old memory was replaced with.
      for (size_t index = 0, rend = memOp.getNumResults(); index < rend;
           ++index) {
        auto result = memOp.getResult(index);
        auto wire = builder.create<WireOp>(
            result.getType(),
            (memOp.name() + "_" + memOp.getPortName(index).getValue()).str());
        result.replaceAllUsesWith(wire.getResult());
        result = wire;
        auto newResult = flatMem.getResult(index);
        auto rType = result.getType().cast<BundleType>();
        for (size_t fieldIndex = 0, fend = rType.getNumElements();
             fieldIndex != fend; ++fieldIndex) {
          auto name = rType.getElement(fieldIndex).name.getValue();
          auto oldField = builder.create<SubfieldOp>(result, fieldIndex);
          Value newField = builder.create<SubfieldOp>(newResult, fieldIndex);
          // data and mask depend on the memory type which was split.  They can
          // also go both directions, depending on the port direction.
          if (!(name == "data" || name == "mask" || name == "wdata" ||
                name == "wmask" || name == "rdata")) {
            emitConnect(builder, newField, oldField);
            continue;
          }
          Value realOldField = oldField;
          if (rType.getElement(fieldIndex).isFlip) {
            // Cast the memory read data from flat type to aggregate.
            newField = builder.createOrFold<BitCastOp>(
                oldField.getType().cast<FIRRTLType>(), newField);
            // Write the aggregate read data.
            emitConnect(builder, realOldField, newField);
          } else {
            // Cast the input aggregate write data to flat type.
            // Cast the input aggregate write data to flat type.
            auto newFieldType = newField.getType().cast<FIRRTLType>();
            auto oldFieldBitWidth = getBitWidth(oldField.getType());
            // Following condition is true, if a data field is 0 bits. Then
            // newFieldType is of smaller bits than old.
            if (getBitWidth(newFieldType) != oldFieldBitWidth.getValue())
              newFieldType =
                  UIntType::get(context, oldFieldBitWidth.getValue());
            realOldField = builder.create<BitCastOp>(newFieldType, oldField);
            // Mask bits require special handling, since some of the mask bits
            // need to be repeated, direct bitcasting wouldn't work. Depending
            // on the mask granularity, some mask bits will be repeated.
            if ((name == "mask" || name == "wmask") &&
                (maskWidths.size() != totalmaskWidths)) {
              Value catMasks;
              for (auto m : llvm::enumerate(maskWidths)) {
                // Get the mask bit.
                auto mBit = builder.createOrFold<BitsPrimOp>(
                    realOldField, m.index(), m.index());
                // Check how many times the mask bit needs to be prepend.
                for (size_t repeat = 0; repeat < m.value(); repeat++)
                  if ((m.index() == 0 && repeat == 0) || !catMasks)
                    catMasks = mBit;
                  else
                    catMasks = builder.createOrFold<CatPrimOp>(mBit, catMasks);
              }
              realOldField = catMasks;
            }
            // Now set the mask or write data.
            // Ensure that the types match.
            emitConnect(
                builder, newField,
                builder.createOrFold<BitCastOp>(
                    newField.getType().cast<FIRRTLType>(), realOldField));
          }
        }
      }
      memOp.erase();
      return;
    });
  }

private:
  // Convert an aggregate type into a flat list of fields.
  // This is used to flatten the aggregate memory datatype.
  // Recursively populate the results with each ground type field.
  static bool flattenType(FIRRTLType type, SmallVectorImpl<IntType> &results) {
    std::function<bool(FIRRTLType)> flatten = [&](FIRRTLType type) -> bool {
      return TypeSwitch<FIRRTLType, bool>(type)
          .Case<BundleType>([&](auto bundle) {
            for (auto &elt : bundle)
              if (!flatten(elt.type))
                return false;
            return true;
          })
          .Case<FVectorType>([&](auto vector) {
            for (size_t i = 0, e = vector.getNumElements(); i != e; ++i)
              if (!flatten(vector.getElementType()))
                return false;
            return true;
          })
          .Case<IntType>([&](auto iType) {
            results.push_back({iType});
            return iType.getWidth().hasValue();
          })
          .Default([&](auto) { return false; });
    };
    if (flatten(type))
      return true;
    return false;
  }

  Value getSubWhatever(ImplicitLocOpBuilder *builder, Value val, size_t index) {
    if (BundleType bundle = val.getType().dyn_cast<BundleType>())
      return builder->create<SubfieldOp>(val, index);
    if (FVectorType fvector = val.getType().dyn_cast<FVectorType>())
      return builder->create<SubindexOp>(val, index);

    llvm_unreachable("Unknown aggregate type");
    return nullptr;
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createFlattenMemoryPass() {
  return std::make_unique<FlattenMemoryPass>();
}
