//===- LowerTypes.cpp - Lower FIRRTL types to ground types ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Conversion/FIRRTLToRTL/FIRRTLToRTL.h"
#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FIRRTL/Types.h"
#include "circt/Dialect/FIRRTL/Visitors.h"
#include "circt/Support/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace firrtl;

// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  // This is the underlying ground type of the field.
  FIRRTLType type;
  // This is a suffix to add to the field name to make it unique.
  std::string suffix;
  // This indicates whether the field was flipped to be an output.
  bool isOutput;

  // Helper to determine if a fully flattened type needs to be flipped.
  FIRRTLType getPortType() { return isOutput ? FlipType::get(type) : type; }
};

// Convert a nested bundle of fields into a flat list of fields.  This is used
// when working with instances and mems to flatten them.
static void flattenBundleTypes(FIRRTLType type, StringRef suffixSoFar,
                               bool isFlipped,
                               SmallVectorImpl<FlatBundleFieldEntry> &results) {
  if (auto flip = type.dyn_cast<FlipType>())
    return flattenBundleTypes(flip.getElementType(), suffixSoFar, !isFlipped,
                              results);

  // In the base case we record this field.
  auto bundle = type.dyn_cast<BundleType>();
  if (!bundle) {
    results.push_back({type, suffixSoFar.str(), isFlipped});
    return;
  }

  SmallString<16> tmpSuffix(suffixSoFar);

  // Otherwise, we have a bundle type.  Break it down.
  for (auto &elt : bundle.getElements()) {
    // Construct the suffix to pass down.
    tmpSuffix.resize(suffixSoFar.size());
    tmpSuffix.push_back('_');
    tmpSuffix.append(elt.first.strref());
    // Recursively process subelements.
    flattenBundleTypes(elt.second, tmpSuffix, isFlipped, results);
  }
}

// Helper to peel off the outer most flip type from a bundle that has all flips
// canonicalized to the outer level, or just return the bundle directly. For any
// other type, returns null.
static BundleType getCanonicalBundleType(Type originalType) {
  BundleType originalBundleType;

  if (auto flipType = originalType.dyn_cast<FlipType>())
    originalBundleType = flipType.getElementType().dyn_cast<BundleType>();
  else
    originalBundleType = originalType.dyn_cast<BundleType>();

  return originalBundleType;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLTypesLowering : public LowerFIRRTLTypesBase<FIRRTLTypesLowering>,
                             public FIRRTLVisitor<FIRRTLTypesLowering> {

  using ValueIdentifier = std::pair<Value, Identifier>;
  using FIRRTLVisitor<FIRRTLTypesLowering>::visitDecl;
  using FIRRTLVisitor<FIRRTLTypesLowering>::visitExpr;
  using FIRRTLVisitor<FIRRTLTypesLowering>::visitStmt;

  void runOnOperation() override;

  // Lowering operations.
  void visitDecl(InstanceOp op);
  void visitExpr(SubfieldOp op);
  void visitStmt(ConnectOp op);

private:
  // Lowering module block arguments.
  void lowerArg(BlockArgument arg, FIRRTLType type);

  // Helpers to manage state.
  Value addArg(Type type, unsigned oldArgNumber, StringRef nameSuffix = "");

  void setBundleLowering(Value oldValue, StringRef flatField, Value newValue);
  Value getBundleLowering(Value oldValue, StringRef flatField);
  void getAllBundleLowerings(Value oldValue, SmallVectorImpl<Value> &results);

  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;

  // State to keep track of arguments and operations to clean up at the end.
  SmallVector<unsigned, 8> argsToRemove;
  SmallVector<Operation *, 16> opsToRemove;

  // State to keep a mapping from (Value, Identifier) pairs to flattened values.
  DenseMap<ValueIdentifier, Value> loweredBundleValues;
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLTypesPass() {
  return std::make_unique<FIRRTLTypesLowering>();
}

// This is the main entrypoint for the lowering pass.
void FIRRTLTypesLowering::runOnOperation() {
  auto module = getOperation();
  auto *body = module.getBodyBlock();

  ImplicitLocOpBuilder theBuilder(module.getLoc(), &getContext());
  builder = &theBuilder;

  // Lower the module block arguments.
  SmallVector<BlockArgument, 8> args(body->args_begin(), body->args_end());
  for (auto arg : args)
    if (auto type = arg.getType().dyn_cast<FIRRTLType>())
      lowerArg(arg, type);

  // Lower the operations.
  for (auto &op : body->getOperations()) {
    builder->setInsertionPoint(&op);
    builder->setLoc(op.getLoc());
    dispatchVisitor(&op);
  }

  // Remove ops that have been lowered.
  while (!opsToRemove.empty())
    opsToRemove.pop_back_val()->erase();

  // Remove args that have been lowered.
  getOperation().eraseArguments(argsToRemove);
  argsToRemove.clear();

  // Reset lowered state.
  loweredBundleValues.clear();
}

//===----------------------------------------------------------------------===//
// Lowering module block arguments.
//===----------------------------------------------------------------------===//

// Lower arguments with bundle type by flattening them.
void FIRRTLTypesLowering::lowerArg(BlockArgument arg, FIRRTLType type) {
  unsigned argNumber = arg.getArgNumber();

  // Flatten any bundle types.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenBundleTypes(type, "", false, fieldTypes);

  for (auto field : fieldTypes) {
    // Create new block arguments.
    auto type = field.getPortType();
    auto newValue = addArg(type, argNumber, field.suffix);

    // If this field was flattened from a bundle.
    if (!field.suffix.empty()) {
      // Remove field separator prefix for consitency with the rest of the pass.
      auto fieldName = StringRef(field.suffix).drop_front(1);

      // Map the flattened suffix for the original bundle to the new value.
      setBundleLowering(arg, fieldName, newValue);
    } else {
      // Lower any other arguments by copying them to keep the relative order.
      arg.replaceAllUsesWith(newValue);
    }
  }

  // Remember to remove the original block argument.
  argsToRemove.push_back(argNumber);
}

//===----------------------------------------------------------------------===//
// Lowering operations.
//===----------------------------------------------------------------------===//

// Lower instance operations in the same way as module block arguments. Bundles
// are flattened, and other arguments are copied to keep the relative order. By
// ensuring both lowerings are the same, we can process every module in the
// circuit in parallel, and every instance will have the correct ports.
void FIRRTLTypesLowering::visitDecl(InstanceOp op) {
  // The instance's ports are represented as one value with a bundle type. Due
  // to how bundles with flips are canonicalized, there may be an outer flip
  // type that wraps the whole bundle.
  Type originalType = op.result().getType();
  bool isFlip = originalType.isa<FlipType>();
  BundleType originalBundleType = getCanonicalBundleType(originalType);
  assert(originalBundleType && "instance result was not a bundle type");

  // Create a new, flat bundle type for the new result
  SmallVector<BundleType::BundleElement, 8> bundleElements;
  for (auto element : originalBundleType.getElements()) {
    // Flatten any nested bundle types the usual way.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    flattenBundleTypes(element.second, element.first, isFlip, fieldTypes);

    for (auto field : fieldTypes) {
      // Store the flat type for the new bundle type.
      auto flatName = builder->getIdentifier(field.suffix);
      auto flatType = field.getPortType();
      auto newElement = BundleType::BundleElement(flatName, flatType);
      bundleElements.push_back(newElement);
    }
  }

  // Get the new bundle type and create a new instance.
  auto newType = BundleType::get(bundleElements, &getContext());

  auto newInstance =
      builder->create<InstanceOp>(newType, op.moduleNameAttr(), op.nameAttr());

  // Create new subfield ops for each field of the instance.
  for (auto element : bundleElements) {
    auto newSubfield =
        builder->create<SubfieldOp>(element.second, newInstance, element.first);

    // Map the flattened suffix for the original bundle to the new value.
    setBundleLowering(op, element.first, newSubfield);
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

// Lowering subfield operations has to deal with three different cases:
//   a) the input value is from a module block argument
//   b) the input value is from another subfield operation's result
//   c) the input value is from an instance
//
// This is accomplished by storing value and suffix mappings that point to the
// flattened value. If the subfield op is accessing the leaf field of a bundle,
// it replaces all uses with the flattened value. Otherwise, it flattens the
// rest of the bundle and adds the flattened values to the mapping for each
// partial suffix.
void FIRRTLTypesLowering::visitExpr(SubfieldOp op) {
  Value input = op.input();
  Value result = op.result();
  StringRef fieldname = op.fieldname();
  FIRRTLType resultType = result.getType().cast<FIRRTLType>();

  // Flatten any nested bundle types the usual way.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenBundleTypes(resultType, fieldname, false, fieldTypes);

  for (auto field : fieldTypes) {
    // Look up the mapping for this suffix.
    auto newValue = getBundleLowering(input, field.suffix);

    // The prefix is the field name and possibly field separator.
    auto prefixSize = fieldname.size();
    if (field.suffix.size() > fieldname.size())
      prefixSize += 1;

    // Get the remaining field suffix by removing the prefix.
    auto partialSuffix = StringRef(field.suffix).drop_front(prefixSize);

    // If we are at the leaf of a bundle.
    if (partialSuffix.empty())
      // Replace the result with the flattened value.
      result.replaceAllUsesWith(newValue);
    else
      // Map the partial suffix for the result value to the flattened value.
      setBundleLowering(result, partialSuffix, newValue);
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

// Lowering connects only has to deal with one special case: connecting two
// bundles. This situation should only arise when both of the arguments are a
// bundle that was:
//   a) originally a block argument
//   b) originally an instance's port
//
// When two such bundles are connected, none of the subfield visits have a
// chance to lower them, so we must ensure they have the same number of
// flattened values and flatten out this connect into multiple connects.
void FIRRTLTypesLowering::visitStmt(ConnectOp op) {
  Value dest = op.dest();
  Value src = op.src();

  // Attempt to get the bundle types, potentially unwrapping an outer flip type
  // that wraps the whole bundle.
  BundleType destType = getCanonicalBundleType(dest.getType());
  BundleType srcType = getCanonicalBundleType(src.getType());

  // If we aren't connecting two bundles, there is nothing to do.
  if (!destType || !srcType)
    return;

  // Get the lowered values for each side.
  SmallVector<Value, 8> destValues;
  getAllBundleLowerings(dest, destValues);

  SmallVector<Value, 8> srcValues;
  getAllBundleLowerings(src, srcValues);

  // Check that we got out the same number of values from each bundle.
  assert(destValues.size() == srcValues.size() &&
         "connected bundles don't match");

  for (auto tuple : llvm::zip_first(destValues, srcValues)) {
    Value newDest = std::get<0>(tuple);
    Value newSrc = std::get<1>(tuple);
    if (newDest.getType().isa<FlipType>())
      builder->create<ConnectOp>(newDest, newSrc);
    else
      builder->create<ConnectOp>(newSrc, newDest);
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

//===----------------------------------------------------------------------===//
// Helpers to manage state.
//===----------------------------------------------------------------------===//

static DictionaryAttr getArgAttrs(StringAttr nameAttr, StringRef suffix,
                                  OpBuilder *builder) {
  SmallString<16> newName(nameAttr.getValue());
  newName += suffix;

  StringAttr newNameAttr = builder->getStringAttr(newName);
  auto attr = builder->getNamedAttr("firrtl.name", newNameAttr);

  return builder->getDictionaryAttr(attr);
}

// Creates and returns a new block argument of the specified type to the module.
// This also maintains the name attribute for the new argument, possibly with a
// new suffix appended.
Value FIRRTLTypesLowering::addArg(Type type, unsigned oldArgNumber,
                                  StringRef nameSuffix) {
  FModuleOp module = getOperation();
  Block *body = module.getBodyBlock();

  // Append the new argument.
  auto newValue = body->addArgument(type);

  // Keep the module's type up-to-date.
  module.setType(builder->getFunctionType(body->getArgumentTypes(), {}));

  // Copy over the name attribute for the new argument.
  StringAttr nameAttr = getFIRRTLNameAttr(module.getArgAttrs(oldArgNumber));
  if (nameAttr) {
    auto newArgAttrs = getArgAttrs(nameAttr, nameSuffix, builder);
    module.setArgAttrs(newValue.getArgNumber(), newArgAttrs);
  }

  return newValue;
}

// Store the mapping from a bundle typed value to a mapping from its field names
// to flat values.
void FIRRTLTypesLowering::setBundleLowering(Value oldValue, StringRef flatField,
                                            Value newValue) {
  auto flatFieldId = builder->getIdentifier(flatField);
  auto &entry = loweredBundleValues[ValueIdentifier(oldValue, flatFieldId)];
  assert(!entry && "bundle lowering has already been set");
  entry = newValue;
}

// For a mapped bundle typed value and a flat subfield name, retrieve and return
// the flat value if it exists.
Value FIRRTLTypesLowering::getBundleLowering(Value oldValue,
                                             StringRef flatField) {
  auto flatFieldId = builder->getIdentifier(flatField);
  auto &entry = loweredBundleValues[ValueIdentifier(oldValue, flatFieldId)];
  assert(entry && "bundle lowering was not set");
  return entry;
}

// For a mapped bundle typed value, retrieve and return the flat values for each
// field.
void FIRRTLTypesLowering::getAllBundleLowerings(
    Value value, SmallVectorImpl<Value> &results) {
  BundleType bundleType = getCanonicalBundleType(value.getType());
  assert(bundleType && "attempted to get bundle lowerings for non-bundle type");
  for (auto element : bundleType.getElements())
    results.push_back(getBundleLowering(value, element.first));
}
