//===- LowerTypes.cpp - Lower FIRRTL types to ground types ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FIRRTL/Types.h"
#include "circt/Dialect/FIRRTL/Visitors.h"
#include "llvm/ADT/StringMap.h"
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
};

// Helper to determine if a fully flattened type needs to be flipped.
static FIRRTLType getPortType(FlatBundleFieldEntry field) {
  return field.isOutput ? FlipType::get(field.type) : field.type;
}

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

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLTypesLowering
    : public LowerFIRRTLTypesBase<FIRRTLTypesLowering>,
      public FIRRTLVisitor<FIRRTLTypesLowering, LogicalResult> {

  using ValueField = std::pair<Value, std::string>;
  using FIRRTLVisitor<FIRRTLTypesLowering, LogicalResult>::visitDecl;
  using FIRRTLVisitor<FIRRTLTypesLowering, LogicalResult>::visitExpr;
  using FIRRTLVisitor<FIRRTLTypesLowering, LogicalResult>::visitStmt;

  void runOnOperation() override;

private:
  // Lowering module block arguments.
  void lowerArg(BlockArgument arg, FIRRTLType type);

  // Lowering operations.
  LogicalResult visitDecl(InstanceOp op);
  LogicalResult visitExpr(SubfieldOp op);
  LogicalResult visitStmt(ConnectOp op);

  // Skip over any other operations.
  LogicalResult visitUnhandledOp(Operation *op) { return success(); }
  LogicalResult visitInvalidOp(Operation *op) { return success(); }

  // Helpers to manage state.
  Value addArgument(Type type, unsigned oldArgNumber,
                    StringRef nameSuffix = "");

  void setBundleLowering(Value oldValue, StringRef flatField, Value newValue);
  Optional<Value> getBundleLowering(Value oldValue, StringRef flatField);
  void getAllBundleLowerings(Value oldValue, SmallVectorImpl<Value> &results);

  void removeArg(unsigned argNumber);
  void removeOp(Operation *op);
  void cleanup();

  // The builder is set and maintained in the main loop.
  OpBuilder *builder;

  // State to keep track of arguments and operations to clean up at the end.
  SmallVector<unsigned, 8> argsToRemove;
  SmallVector<Operation *, 16> opsToRemove;

  // State to keep a mapping from original bundle values to flattened names and
  // flattened values.
  DenseMap<Value, llvm::StringMap<Value>> loweredBundleValues;
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

  OpBuilder theBuilder(&getContext());
  builder = &theBuilder;

  // Lower the module block arguments.
  SmallVector<BlockArgument, 8> args(body->args_begin(), body->args_end());
  for (auto arg : args)
    if (auto type = arg.getType().dyn_cast<FIRRTLType>())
      lowerArg(arg, type);

  // Lower the operations.
  for (auto &op : body->getOperations()) {
    builder->setInsertionPoint(&op);
    if (failed(dispatchVisitor(&op)))
      return;
  }

  cleanup();
}

//===----------------------------------------------------------------------===//
// Lowering module block arguments.
//===----------------------------------------------------------------------===//

// Lower arguments with bundle type by flattening them.
void FIRRTLTypesLowering::lowerArg(BlockArgument arg, FIRRTLType type) {
  // Flatten any bundle types.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenBundleTypes(type, "", false, fieldTypes);

  for (auto field : fieldTypes) {
    // Create new block arguments.
    auto type = getPortType(field);
    auto newValue = addArgument(type, arg.getArgNumber(), field.suffix);

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
}

//===----------------------------------------------------------------------===//
// Lowering operations.
//===----------------------------------------------------------------------===//

// Lower instance operations in the same way as module block arguments. Bundles
// are flattened, and other arguments are copied to keep the relative order. By
// ensuring both lowerings are the same, we can process every module in the
// circuit in parallel, and every instance will have the correct ports.
LogicalResult FIRRTLTypesLowering::visitDecl(InstanceOp op) {
  // The instance's ports are represented as one value with a bundle type.
  BundleType originalType = op.result().getType().cast<BundleType>();

  // Create a new, flat bundle type for the new result
  SmallVector<BundleType::BundleElement, 8> bundleElements;
  for (auto element : originalType.getElements()) {
    // Flatten any nested bundle types the usual way.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    flattenBundleTypes(element.second, element.first, false, fieldTypes);

    for (auto field : fieldTypes) {
      // Store the flat type for the new bundle type.
      auto flatName = builder->getIdentifier(field.suffix);
      auto flatType = getPortType(field);
      auto newElement = BundleType::BundleElement(flatName, flatType);
      bundleElements.push_back(newElement);
    }
  }

  // Get the new bundle type and create a new instance.
  auto newType = BundleType::get(bundleElements, &getContext());

  auto newInstance = builder->create<InstanceOp>(
      op.getLoc(), newType, op.moduleNameAttr(), op.nameAttr());

  // Create new subfield ops for each field of the instance.
  for (auto element : bundleElements) {
    auto newSubfield = builder->create<SubfieldOp>(op.getLoc(), element.second,
                                                   newInstance, element.first);

    // Map the flattened suffix for the original bundle to the new value.
    setBundleLowering(op, element.first, newSubfield);
  }

  removeOp(op);

  return success();
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
LogicalResult FIRRTLTypesLowering::visitExpr(SubfieldOp op) {
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
    assert(newValue.hasValue() && "no lowered value for instance subfield");

    // The prefix is the field name and possibly field separator.
    auto prefixSize = fieldname.size();
    if (field.suffix.size() > fieldname.size())
      prefixSize += 1;

    // Get the remaining field suffix by removing the prefix.
    auto partialSuffix = StringRef(field.suffix).drop_front(prefixSize);

    // If we are at the leaf of a bundle.
    if (partialSuffix.empty())
      // Replace the result with the flattened value.
      result.replaceAllUsesWith(newValue.getValue());
    else
      // Map the partial suffix for the result value to the flattened value.
      setBundleLowering(result, partialSuffix, newValue.getValue());
  }

  removeOp(op);

  return success();
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
LogicalResult FIRRTLTypesLowering::visitStmt(ConnectOp op) {
  Value dest = op.dest();
  Value src = op.src();

  // If we aren't connecting two bundles, there is nothing to do.
  if (!dest.getType().isa<BundleType>() || !src.getType().isa<BundleType>())
    return success();

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
    builder->create<ConnectOp>(op.getLoc(), newDest, newSrc);
  }

  removeOp(op);

  return success();
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
Value FIRRTLTypesLowering::addArgument(Type type, unsigned oldArgNumber,
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

  // Remember to delete the original block argument.
  removeArg(oldArgNumber);

  return newValue;
}

// Store the mapping from a bundle typed value to a mapping from its field names
// to flat values.
void FIRRTLTypesLowering::setBundleLowering(Value oldValue, StringRef flatField,
                                            Value newValue) {
  auto &entry = loweredBundleValues[oldValue][flatField];
  assert(!entry && "bundle lowering has already been set");
  entry = newValue;
}

// For a mapped bundle typed value and a flat subfield name, retrieve and return
// the flat value if it exists.
Optional<Value> FIRRTLTypesLowering::getBundleLowering(Value oldValue,
                                                       StringRef flatField) {
  if (oldValue && loweredBundleValues.count(oldValue) &&
      loweredBundleValues[oldValue].count(flatField))
    return Optional<Value>(loweredBundleValues[oldValue][flatField]);

  return None;
}

// For a mapped bundle typed value, retrieve and return the flat values for each
// field.
void FIRRTLTypesLowering::getAllBundleLowerings(
    Value value, SmallVectorImpl<Value> &results) {
  if (!loweredBundleValues.count(value))
    return;

  BundleType bundleType = value.getType().cast<BundleType>();
  for (auto element : bundleType.getElements()) {
    SmallString<16> loweredName;
    loweredName.append(element.first);

    assert(loweredBundleValues[value].count(loweredName) &&
           "no lowered bundle value");

    results.push_back(loweredBundleValues[value][loweredName]);
  }
}

// Remember an argument number to erase during cleanup.
void FIRRTLTypesLowering::removeArg(unsigned argNumber) {
  argsToRemove.push_back(argNumber);
}

// Remember an operation to erase during cleanup.
void FIRRTLTypesLowering::removeOp(Operation *op) { opsToRemove.push_back(op); }

// Handle deferred removals of operations and block arguments when done. Also
// clean up state.
void FIRRTLTypesLowering::cleanup() {
  while (!opsToRemove.empty())
    opsToRemove.pop_back_val()->erase();

  getOperation().eraseArguments(argsToRemove);
  argsToRemove.clear();

  loweredBundleValues.clear();
}
