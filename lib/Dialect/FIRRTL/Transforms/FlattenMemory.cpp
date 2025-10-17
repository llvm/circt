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

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "lower-memory"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_FLATTENMEMORY
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct FlattenMemoryPass
    : public circt::firrtl::impl::FlattenMemoryBase<FlattenMemoryPass> {

  /// Returns true if the the memory has annotations on a subfield of any of the
  /// ports.
  static bool hasSubAnno(MemOp op) {
    for (size_t portIdx = 0, e = op.getNumResults(); portIdx < e; ++portIdx)
      for (auto attr : op.getPortAnnotation(portIdx))
        if (cast<DictionaryAttr>(attr).get("circt.fieldID"))
          return true;
    return false;
  };

  /// This pass flattens the aggregate data of memory into a UInt, and inserts
  /// appropriate bitcasts to access the data.
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "\n Running lower memory on module:"
                            << getOperation().getName());
    SmallVector<Operation *> opsToErase;
    getOperation().getBodyBlock()->walk([&](MemOp memOp) {
      LLVM_DEBUG(llvm::dbgs() << "\n Memory:" << memOp);

      // Cannot flatten a memory if it has debug ports, because debug port
      // implies a memtap and we cannot transform the datatype for a memory that
      // is tapped.
      for (auto res : memOp.getResults())
        if (isa<RefType>(res.getType()))
          return;

      // If subannotations present on aggregate fields, we cannot flatten the
      // memory. It must be split into one memory per aggregate field.
      // Do not overwrite the pass flag!
      if (hasSubAnno(memOp))
        return;

      // The vector of leaf elements type after flattening the data. If any of
      // the datatypes cannot be flattened, then we cannot flatten the memory.
      SmallVector<FIRRTLBaseType> flatMemType;
      if (!flattenType(memOp.getDataType(), flatMemType))
        return;

      // Calculate the width of the memory data type, and the width of
      // each individual aggregate leaf elements.
      size_t memFlatWidth = 0;
      SmallVector<int32_t> memWidths;
      for (auto f : flatMemType) {
        LLVM_DEBUG(llvm::dbgs() << "\n field type:" << f);
        auto w = f.getBitWidthOrSentinel();
        memWidths.push_back(w);
        memFlatWidth += w;
      }
      // If all the widths are zero, ignore the memory.
      if (!memFlatWidth)
        return;

      // Calculate the mask granularity of this memory, which is how many bits
      // of the data each mask bit controls. This is the greatest common
      // denominator of the widths of the flattened data types.
      auto maskGran = memWidths.front();
      for (auto w : ArrayRef(memWidths).drop_front())
        maskGran = std::gcd(maskGran, w);

      // Total mask bitwidth after flattening.
      uint32_t totalmaskWidths = 0;
      // How many mask bits each field type requires.
      SmallVector<unsigned> maskWidths;
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
      // Create a new memory data type of unsigned and computed width.
      auto flatType = UIntType::get(context, memFlatWidth);
      for (auto port : memOp.getPorts()) {
        ports.push_back(MemOp::getTypeForPort(memOp.getDepth(), flatType,
                                              port.second, totalmaskWidths));
        portNames.push_back(port.first);
      }

      // Create the new flattened memory.
      auto flatMem = MemOp::create(
          builder, ports, memOp.getReadLatency(), memOp.getWriteLatency(),
          memOp.getDepth(), memOp.getRuw(), builder.getArrayAttr(portNames),
          memOp.getNameAttr(), memOp.getNameKind(), memOp.getAnnotations(),
          memOp.getPortAnnotations(), memOp.getInnerSymAttr(),
          memOp.getInitAttr(), memOp.getPrefixAttr());

      // Hook up the new memory to the wires the old memory was replaced with.
      for (size_t index = 0, rend = memOp.getNumResults(); index < rend;
           ++index) {

        // Create a wire with the original type, and replace all uses of the old
        // memory with the wire.  We will be reconstructing the original type
        // in the wire from the bitvector of the flattened memory.
        auto result = memOp.getResult(index);
        auto wire =
            WireOp::create(
                builder, result.getType(),
                (memOp.getName() + "_" + memOp.getPortName(index)).str())
                .getResult();
        result.replaceAllUsesWith(wire);
        result = wire;
        auto newResult = flatMem.getResult(index);
        auto rType = type_cast<BundleType>(result.getType());
        for (size_t fieldIndex = 0, fend = rType.getNumElements();
             fieldIndex != fend; ++fieldIndex) {
          auto name = rType.getElement(fieldIndex).name;
          auto oldField = SubfieldOp::create(builder, result, fieldIndex);
          FIRRTLBaseValue newField =
              SubfieldOp::create(builder, newResult, fieldIndex);
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
            auto castField =
                builder.createOrFold<BitCastOp>(oldField.getType(), newField);
            // Write the aggregate read data.
            emitConnect(builder, realOldField, castField);
          } else {
            // Cast the input aggregate write data to flat type.
            auto newFieldType = newField.getType();
            auto oldFieldBitWidth = getBitWidth(oldField.getType());
            // Following condition is true, if a data field is 0 bits. Then
            // newFieldType is of smaller bits than old.
            if (getBitWidth(newFieldType) != *oldFieldBitWidth)
              newFieldType = UIntType::get(context, *oldFieldBitWidth);
            realOldField = BitCastOp::create(builder, newFieldType, oldField);
            // Mask bits require special handling, since some of the mask bits
            // need to be repeated, direct bitcasting wouldn't work. Depending
            // on the mask granularity, some mask bits will be repeated.
            if ((name == "mask" || name == "wmask") &&
                (maskWidths.size() != totalmaskWidths)) {
              Value catMasks;
              for (const auto &m : llvm::enumerate(maskWidths)) {
                // Get the mask bit.
                auto mBit = builder.createOrFold<BitsPrimOp>(
                    realOldField, m.index(), m.index());
                // Check how many times the mask bit needs to be prepend.
                for (size_t repeat = 0; repeat < m.value(); repeat++)
                  if ((m.index() == 0 && repeat == 0) || !catMasks)
                    catMasks = mBit;
                  else
                    catMasks = builder.createOrFold<CatPrimOp>(
                        ValueRange{mBit, catMasks});
              }
              realOldField = catMasks;
            }
            // Now set the mask or write data.
            // Ensure that the types match.
            emitConnect(builder, newField,
                        builder.createOrFold<BitCastOp>(newField.getType(),
                                                        realOldField));
          }
        }
      }
      ++numFlattenedMems;
      memOp.erase();
      return;
    });
  }

private:
  // Convert an aggregate type into a flat list of fields. This is used to
  // flatten the aggregate memory datatype. Recursively populate the results
  // with each ground type field.
  static bool flattenType(FIRRTLType type,
                          SmallVectorImpl<FIRRTLBaseType> &results) {
    std::function<bool(FIRRTLType)> flatten = [&](FIRRTLType type) -> bool {
      return FIRRTLTypeSwitch<FIRRTLType, bool>(type)
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
          .Case<IntType>([&](IntType type) {
            results.push_back(type);
            return type.getWidth().has_value();
          })
          .Case<FEnumType>([&](FEnumType type) {
            results.emplace_back(type);
            return true;
          })
          .Default([&](auto) { return false; });
    };
    // Return true only if this is an aggregate with more than one element.
    return flatten(type) && results.size() > 1;
  }

  Value getSubWhatever(ImplicitLocOpBuilder *builder, Value val, size_t index) {
    if (BundleType bundle = type_dyn_cast<BundleType>(val.getType()))
      return builder->create<SubfieldOp>(val, index);
    if (FVectorType fvector = type_dyn_cast<FVectorType>(val.getType()))
      return builder->create<SubindexOp>(val, index);

    llvm_unreachable("Unknown aggregate type");
    return nullptr;
  }
};
} // end anonymous namespace
