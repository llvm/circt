//===- HWLegalizeModulesPass.cpp - Lower unsupported IR features away -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers away features in the SV/Comb/HW dialects that are
// unsupported by some tools (e.g. multidimensional arrays) as specified by
// LoweringOptions.  This pass is run relatively late in the pipeline in
// preparation for emission.  Any passes run after this (e.g. PrettifyVerilog)
// must be aware they cannot introduce new invalid constructs.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/Builders.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// HWLegalizeModulesPass
//===----------------------------------------------------------------------===//

namespace {
struct HWLegalizeModulesPass
    : public sv::HWLegalizeModulesBase<HWLegalizeModulesPass> {
  void runOnOperation() override;

private:
  void processPostOrder(Block &block);
  bool tryLoweringPackedArrayOp(Operation &op);
  Value lowerLookupToCasez(Operation &op, Value input, Value index,
                           mlir::Type elementType,
                           SmallVector<Value> caseValues);
  bool processUsers(Operation &op, Value value, ArrayRef<Value> mapping);
  std::optional<std::pair<uint64_t, unsigned>>
  tryExtractIndexAndBitWidth(Value value);

  /// This is the current hw.module being processed.
  hw::HWModuleOp thisHWModule;

  bool anythingChanged;

  /// This tells us what language features we're allowed to use in generated
  /// Verilog.
  LoweringOptions options;

  /// This pass will be run on multiple hw.modules, this keeps track of the
  /// contents of LoweringOptions so we don't have to reparse the
  /// LoweringOptions for every hw.module.
  StringAttr lastParsedOptions;
};
} // end anonymous namespace

bool HWLegalizeModulesPass::tryLoweringPackedArrayOp(Operation &op) {
  return TypeSwitch<Operation *, bool>(&op)
      .Case<hw::AggregateConstantOp>([&](hw::AggregateConstantOp constOp) {
        // Replace individual element uses (if any) with input fields.
        SmallVector<Value> inputs;
        OpBuilder builder(constOp);
        for (auto field : llvm::reverse(constOp.getFields())) {
          if (auto intAttr = dyn_cast<IntegerAttr>(field))
            inputs.push_back(
                builder.create<hw::ConstantOp>(constOp.getLoc(), intAttr));
          else
            inputs.push_back(builder.create<hw::AggregateConstantOp>(
                constOp.getLoc(), constOp.getType(), field.cast<ArrayAttr>()));
        }
        if (!processUsers(op, constOp.getResult(), inputs))
          return false;

        // Remove original op.
        return true;
      })
      .Case<hw::ArrayConcatOp>([&](hw::ArrayConcatOp concatOp) {
        // Redirect individual element uses (if any) to the input arguments.
        SmallVector<std::pair<Value, uint64_t>> arrays;
        for (auto array : llvm::reverse(concatOp.getInputs())) {
          auto ty = hw::type_cast<hw::ArrayType>(array.getType());
          arrays.emplace_back(array, ty.getNumElements());
        }
        for (auto *user :
             llvm::make_early_inc_range(concatOp.getResult().getUsers())) {
          if (TypeSwitch<Operation *, bool>(user)
                  .Case<hw::ArrayGetOp>([&](hw::ArrayGetOp getOp) {
                    if (auto indexAndBitWidth =
                            tryExtractIndexAndBitWidth(getOp.getIndex())) {
                      auto [indexValue, bitWidth] = *indexAndBitWidth;
                      // FIXME: More efficient search
                      for (const auto &[array, size] : arrays) {
                        if (indexValue >= size) {
                          indexValue -= size;
                          continue;
                        }
                        OpBuilder builder(getOp);
                        getOp.getInputMutable().set(array);
                        getOp.getIndexMutable().set(
                            builder.createOrFold<hw::ConstantOp>(
                                getOp.getLoc(), APInt(bitWidth, indexValue)));
                        return true;
                      }
                    }

                    return false;
                  })
                  .Default([](auto op) { return false; }))
            continue;

          op.emitError("unsupported packed array expression");
          signalPassFailure();
        }

        // Remove the original op.
        return true;
      })
      .Case<hw::ArrayCreateOp>([&](hw::ArrayCreateOp createOp) {
        // Replace individual element uses (if any) with input arguments.
        SmallVector<Value> inputs(llvm::reverse(createOp.getInputs()));
        if (!processUsers(op, createOp.getResult(), inputs))
          return false;

        // Remove original op.
        return true;
      })
      .Case<hw::ArrayGetOp>([&](hw::ArrayGetOp getOp) {
        // Skip index ops with constant index.
        auto index = getOp.getIndex();
        if (auto *definingOp = index.getDefiningOp())
          if (isa<hw::ConstantOp>(definingOp))
            return false;

        // Generate case value element lookups.
        auto ty = hw::type_cast<hw::ArrayType>(getOp.getInput().getType());
        OpBuilder builder(getOp);
        SmallVector<Value> caseValues;
        for (size_t i = 0, e = ty.getNumElements(); i < e; i++) {
          auto loc = op.getLoc();
          auto index = builder.createOrFold<hw::ConstantOp>(
              loc, APInt(llvm::Log2_64_Ceil(e), i));
          auto element =
              builder.create<hw::ArrayGetOp>(loc, getOp.getInput(), index);
          caseValues.push_back(element);
        }

        // Transform array index op into casez statement.
        auto theWire = lowerLookupToCasez(op, getOp.getInput(), index,
                                          ty.getElementType(), caseValues);

        // Emit the read from the wire, replace uses and clean up.
        builder.setInsertionPoint(getOp);
        auto readWire =
            builder.create<sv::ReadInOutOp>(getOp.getLoc(), theWire);
        getOp.getResult().replaceAllUsesWith(readWire);
        return true;
      })
      .Case<sv::ArrayIndexInOutOp>([&](sv::ArrayIndexInOutOp indexOp) {
        // Skip index ops with constant index.
        auto index = indexOp.getIndex();
        if (auto *definingOp = index.getDefiningOp())
          if (isa<hw::ConstantOp>(definingOp))
            return false;

        // Skip index ops with unpacked arrays.
        auto inout = indexOp.getInput().getType();
        if (hw::type_isa<hw::UnpackedArrayType>(inout.getElementType()))
          return false;

        // Generate case value element lookups.
        auto ty = hw::type_cast<hw::ArrayType>(inout.getElementType());
        OpBuilder builder(&op);
        SmallVector<Value> caseValues;
        for (size_t i = 0, e = ty.getNumElements(); i < e; i++) {
          auto loc = op.getLoc();
          auto index = builder.createOrFold<hw::ConstantOp>(
              loc, APInt(llvm::Log2_64_Ceil(e), i));
          auto element = builder.create<sv::ArrayIndexInOutOp>(
              loc, indexOp.getInput(), index);
          auto readElement = builder.create<sv::ReadInOutOp>(loc, element);
          caseValues.push_back(readElement);
        }

        // Transform array index op into casez statement.
        auto theWire = lowerLookupToCasez(op, indexOp.getInput(), index,
                                          ty.getElementType(), caseValues);

        // Replace uses and clean up.
        indexOp.getResult().replaceAllUsesWith(theWire);
        return true;
      })
      .Case<sv::PAssignOp>([&](sv::PAssignOp assignOp) {
        // Transform array assignment into individual assignments for each array
        // element.
        auto inout = assignOp.getDest().getType();
        auto ty = hw::type_dyn_cast<hw::ArrayType>(inout.getElementType());
        if (!ty)
          return false;

        OpBuilder builder(assignOp);
        for (size_t i = 0, e = ty.getNumElements(); i < e; i++) {
          auto loc = op.getLoc();
          auto index = builder.createOrFold<hw::ConstantOp>(
              loc, APInt(llvm::Log2_64_Ceil(e), i));
          auto dstElement = builder.create<sv::ArrayIndexInOutOp>(
              loc, assignOp.getDest(), index);
          auto srcElement =
              builder.create<hw::ArrayGetOp>(loc, assignOp.getSrc(), index);
          builder.create<sv::PAssignOp>(loc, dstElement, srcElement);
        }

        // Remove original assignment.
        return true;
      })
      .Case<sv::RegOp>([&](sv::RegOp regOp) {
        // Transform array reg into individual regs for each array element.
        auto ty = hw::type_dyn_cast<hw::ArrayType>(regOp.getElementType());
        if (!ty)
          return false;

        OpBuilder builder(regOp);
        auto name = StringAttr::get(regOp.getContext(), "name");
        SmallVector<Value> elements;
        for (size_t i = 0, e = ty.getNumElements(); i < e; i++) {
          auto loc = op.getLoc();
          auto element = builder.create<sv::RegOp>(loc, ty.getElementType());
          if (auto nameAttr = regOp->getAttrOfType<StringAttr>(name)) {
            element.setNameAttr(
                StringAttr::get(regOp.getContext(), nameAttr.getValue()));
          }
          elements.push_back(element);
        }

        // Fix users to refer to individual element regs.
        if (!processUsers(op, regOp.getResult(), elements))
          return false;

        // Remove original reg.
        return true;
      })
      .Default([&](auto op) { return false; });
}

Value HWLegalizeModulesPass::lowerLookupToCasez(Operation &op, Value input,
                                                Value index,
                                                mlir::Type elementType,
                                                SmallVector<Value> caseValues) {
  // Create the wire for the result of the casez in the
  // hw.module.
  OpBuilder builder(&op);
  auto theWire = builder.create<sv::RegOp>(op.getLoc(), elementType,
                                           builder.getStringAttr("casez_tmp"));
  builder.setInsertionPoint(&op);

  auto loc = input.getDefiningOp()->getLoc();
  // A casez is a procedural operation, so if we're in a
  // non-procedural region we need to inject an always_comb
  // block.
  if (!op.getParentOp()->hasTrait<sv::ProceduralRegion>()) {
    auto alwaysComb = builder.create<sv::AlwaysCombOp>(loc);
    builder.setInsertionPointToEnd(alwaysComb.getBodyBlock());
  }

  // If we are missing elements in the array (it is non-power of
  // two), then add a default 'X' value.
  if (1ULL << index.getType().getIntOrFloatBitWidth() != caseValues.size()) {
    caseValues.push_back(builder.create<sv::ConstantXOp>(
        op.getLoc(), op.getResult(0).getType()));
  }

  APInt caseValue(index.getType().getIntOrFloatBitWidth(), 0);
  auto *context = builder.getContext();

  // Create the casez itself.
  builder.create<sv::CaseOp>(
      loc, CaseStmtType::CaseZStmt, index, caseValues.size(),
      [&](size_t caseIdx) -> std::unique_ptr<sv::CasePattern> {
        // Use a default pattern for the last value, even if we
        // are complete. This avoids tools thinking they need to
        // insert a latch due to potentially incomplete case
        // coverage.
        bool isDefault = caseIdx == caseValues.size() - 1;
        Value theValue = caseValues[caseIdx];
        std::unique_ptr<sv::CasePattern> thePattern;

        if (isDefault)
          thePattern = std::make_unique<sv::CaseDefaultPattern>(context);
        else
          thePattern = std::make_unique<sv::CaseBitPattern>(caseValue, context);
        ++caseValue;
        builder.create<sv::BPAssignOp>(loc, theWire, theValue);
        return thePattern;
      });

  return theWire;
}

bool HWLegalizeModulesPass::processUsers(Operation &op, Value value,
                                         ArrayRef<Value> mapping) {
  for (auto *user : llvm::make_early_inc_range(value.getUsers())) {
    if (TypeSwitch<Operation *, bool>(user)
            .Case<hw::ArrayGetOp>([&](hw::ArrayGetOp getOp) {
              if (auto indexAndBitWidth =
                      tryExtractIndexAndBitWidth(getOp.getIndex())) {
                getOp.replaceAllUsesWith(mapping[indexAndBitWidth->first]);
                return true;
              }

              return false;
            })
            .Case<sv::ArrayIndexInOutOp>([&](sv::ArrayIndexInOutOp indexOp) {
              if (auto indexAndBitWidth =
                      tryExtractIndexAndBitWidth(indexOp.getIndex())) {
                indexOp.replaceAllUsesWith(mapping[indexAndBitWidth->first]);
                return true;
              }

              return false;
            })
            .Default([](auto op) { return false; })) {
      user->erase();
      continue;
    }

    user->emitError("unsupported packed array expression");
    signalPassFailure();
    return false;
  }

  return true;
}

std::optional<std::pair<uint64_t, unsigned>>
HWLegalizeModulesPass::tryExtractIndexAndBitWidth(Value value) {
  if (auto constantOp = dyn_cast<hw::ConstantOp>(value.getDefiningOp())) {
    auto index = constantOp.getValue();
    return std::make_optional(
        std::make_pair(index.getZExtValue(), index.getBitWidth()));
  }
  return std::nullopt;
}

void HWLegalizeModulesPass::processPostOrder(Block &body) {
  if (body.empty())
    return;

  // Walk the block bottom-up, processing the region tree inside out.
  Block::iterator it = std::prev(body.end());
  while (it != body.end()) {
    auto &op = *it;

    // Advance the iterator, using the end iterator as a sentinel that we're at
    // the top of the block.
    if (it == body.begin())
      it = body.end();
    else
      --it;

    if (op.getNumRegions()) {
      for (auto &region : op.getRegions())
        for (auto &regionBlock : region.getBlocks())
          processPostOrder(regionBlock);
    }

    if (options.disallowPackedArrays) {
      // Try supported packed array op lowering.
      if (tryLoweringPackedArrayOp(op)) {
        it = --Block::iterator(op);
        op.erase();
        anythingChanged = true;
        continue;
      }

      // Otherwise, if the IR produces a packed array and we aren't allowing
      // multi-dimensional arrays, reject the IR as invalid.
      for (auto value : op.getResults()) {
        if (value.getType().isa<hw::ArrayType>()) {
          op.emitError("unsupported packed array expression");
          signalPassFailure();
        }
      }
    }
  }
}

void HWLegalizeModulesPass::runOnOperation() {
  thisHWModule = getOperation();

  // Parse the lowering options if necessary.
  auto optionsAttr = LoweringOptions::getAttributeFrom(
      cast<ModuleOp>(thisHWModule->getParentOp()));
  if (optionsAttr != lastParsedOptions) {
    if (optionsAttr)
      options = LoweringOptions(optionsAttr.getValue(), [&](Twine error) {
        thisHWModule.emitError(error);
      });
    else
      options = LoweringOptions();
    lastParsedOptions = optionsAttr;
  }

  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;

  // Walk the operations in post-order, transforming any that are interesting.
  processPostOrder(*thisHWModule.getBodyBlock());

  // If we did not change anything in the IR mark all analysis as preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createHWLegalizeModulesPass() {
  return std::make_unique<HWLegalizeModulesPass>();
}
