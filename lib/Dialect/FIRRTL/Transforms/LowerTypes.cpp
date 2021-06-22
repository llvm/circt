//===- LowerTypes.cpp - Lower FIRRTL types to ground types ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering of FIRRTL types to ground types.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Parallel.h"
#include <algorithm>

using namespace circt;
using namespace firrtl;

namespace {
// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  // This is the underlying ground type of the field.
  FIRRTLType type;
  // This is a reference to the exact field.
  unsigned fieldID;
  // This is a suffix to add to the field name to make it unique.
  std::string suffix;
  // This indicates whether the field was flipped to be an output.
  bool isOutput;

  // Helper to determine if a fully flattened type needs to be flipped.
  FIRRTLType getPortType() { return type; }
};
} // end anonymous namespace

// Convert an aggregate type into a flat list of fields.  This is used
// when working with instances and mems to flatten them.
static void flattenType(FIRRTLType type,
                        SmallVectorImpl<FlatBundleFieldEntry> &results) {
  // Field ID is the index of each field when walking aggregate types in a
  // depth-first order. See FieldRef for more information about how this
  // works.
  unsigned fieldID = 0;
  std::function<void(FIRRTLType, StringRef, bool)> flatten =
      [&](FIRRTLType type, StringRef suffixSoFar, bool isFlipped) {
        TypeSwitch<FIRRTLType>(type)
            .Case<BundleType>([&](auto bundle) {
              SmallString<16> tmpSuffix(suffixSoFar);
              // Increment the field ID to point to the first element.
              fieldID++;

              // Otherwise, we have a bundle type.  Break it down.
              for (auto &elt : bundle.getElements()) {
                // Construct the suffix to pass down.
                tmpSuffix.resize(suffixSoFar.size());
                tmpSuffix.push_back('_');
                tmpSuffix.append(elt.name.getValue());
                // Recursively process subelements.
                flatten(elt.type, tmpSuffix, isFlipped ^ elt.isFlip);
              }
              return;
            })
            .Case<FVectorType>([&](auto vector) {
              // Increment the field ID to point to the first element.
              fieldID++;
              for (size_t i = 0, e = vector.getNumElements(); i != e; ++i) {
                flatten(vector.getElementType(),
                        (suffixSoFar + "_" + std::to_string(i)).str(),
                        isFlipped);
              }
              return;
            })
            .Default([&](auto) {
              results.push_back({type, fieldID, suffixSoFar.str(), isFlipped});
              // Increment the field ID past this type. This is important when
              // this type is nested in an aggregate type.
              fieldID++;
              return;
            });
      };

  return flatten(type, "", false);
}

// Helper to peel off the outer most flip type from an aggregate type that has
// all flips canonicalized to the outer level, or just return the bundle
// directly. For any ground type, returns null.
static FIRRTLType getCanonicalAggregateType(Type originalType) {
  FIRRTLType unflipped = originalType.dyn_cast<FIRRTLType>();

  return TypeSwitch<FIRRTLType, FIRRTLType>(unflipped)
      .Case<BundleType, FVectorType>([](auto a) { return a; })
      .Default([](auto) { return nullptr; });
}

/// Copy annotations from \p annotations to \p loweredAttrs, except annotations
/// with "target" key, that do not match the field suffix.
static void filterAnnotations(ArrayAttr annotations,
                              SmallVector<Attribute> &loweredAttrs,
                              StringRef suffix) {
  if (!annotations || annotations.empty())
    return;

  for (auto opAttr : annotations) {
    auto di = opAttr.dyn_cast<DictionaryAttr>();
    if (!di) {
      loweredAttrs.push_back(opAttr);
      continue;
    }
    auto targetAttr = di.get("target");
    if (!targetAttr) {
      loweredAttrs.push_back(opAttr);
      continue;
    }

    ArrayAttr subFieldTarget = targetAttr.cast<ArrayAttr>();
    SmallString<16> targetStr;
    for (auto fName : subFieldTarget) {
      std::string fNameStr = fName.cast<StringAttr>().getValue().str();
      // The fNameStr will begin with either '[' or '.', replace it with an
      // '_' to construct the suffix.
      fNameStr[0] = '_';
      // If it ends with ']', then just remove it.
      if (fNameStr.back() == ']')
        fNameStr.erase(fNameStr.size() - 1);

      targetStr += fNameStr;
    }
    // If no subfield attribute, then copy the annotation.
    if (targetStr.empty()) {
      loweredAttrs.push_back(opAttr);
      continue;
    }
    // If the subfield suffix doesn't match, then ignore the annotation.
    if (suffix.find(targetStr.str().str()) != 0)
      continue;

    NamedAttrList modAttr;
    for (auto attr : di.getValue()) {
      // Ignore the actual target annotation, but copy the rest of annotations.
      if (attr.first.str() == "target")
        continue;
      modAttr.push_back(attr);
    }
    loweredAttrs.push_back(
        DictionaryAttr::get(annotations.getContext(), modAttr));
  }
}

/// Copy annotations from \p annotations into a new AnnotationSet and return it.
/// This removes annotations with "target" key that does not match the field
/// suffix.
static AnnotationSet filterAnnotations(AnnotationSet annotations,
                                       StringRef suffix) {
  if (annotations.empty())
    return annotations;
  SmallVector<Attribute> loweredAttrs;
  filterAnnotations(annotations.getArrayAttr(), loweredAttrs, suffix);
  return AnnotationSet(ArrayAttr::get(annotations.getContext(), loweredAttrs));
}

//===----------------------------------------------------------------------===//
// Module Type Lowering
//===----------------------------------------------------------------------===//

namespace {
class TypeLoweringVisitor : public FIRRTLVisitor<TypeLoweringVisitor> {
public:
  TypeLoweringVisitor(MLIRContext *context) : context(context) {}
  using FIRRTLVisitor<TypeLoweringVisitor>::visitDecl;
  using FIRRTLVisitor<TypeLoweringVisitor>::visitExpr;
  using FIRRTLVisitor<TypeLoweringVisitor>::visitStmt;

  // If the referenced operation is a FModuleOp or an FExtModuleOp, perform type
  // lowering on all operations.
  void lowerModule(Operation *op);

  void visitDecl(FExtModuleOp op);
  void visitDecl(FModuleOp op);
  void visitDecl(InstanceOp op);
  void visitDecl(MemOp op);
  void visitDecl(NodeOp op);
  void visitDecl(RegOp op);
  void visitDecl(WireOp op);
  void visitDecl(RegResetOp op);
  void visitExpr(InvalidValueOp op);
  void visitExpr(SubfieldOp op);
  void visitExpr(SubindexOp op);
  void visitExpr(SubaccessOp op);
  void visitExpr(MuxPrimOp op);
  void visitExpr(AsPassivePrimOp op);
  void visitStmt(ConnectOp op);
  void visitStmt(WhenOp op);
  void visitStmt(PartialConnectOp op);

private:
  // Lowering module block arguments.
  void lowerArg(FModuleOp module, BlockArgument arg, FIRRTLType type);

  // Helpers to manage state.
  Value addArg(FModuleOp module, Type type, unsigned oldArgNumber,
               Direction direction, StringRef nameSuffix = "");

  void setBundleLowering(FieldRef fieldRef, Value newValue);
  Value getBundleLowering(FieldRef fieldRef);
  void getAllBundleLowerings(Value oldValue,
                             SmallVectorImpl<std::pair<Value, bool>> &results);

  MLIRContext *context;

  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;

  // State to keep track of arguments and operations to clean up at the end.
  SmallVector<unsigned, 8> argsToRemove;
  SmallVector<Operation *, 16> opsToRemove;

  // State to keep a mapping from (Value, Identifier) pairs to flattened values.
  DenseMap<FieldRef, Value> loweredBundleValues;

  // State to track the new attributes for the module.
  SmallVector<NamedAttribute, 8> newModuleAttrs;
  SmallVector<Attribute> newArgNames;
  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute, 8> newArgAttrs;
  size_t originalNumModuleArgs;

  /// Connect two values, truncating the source value if it is a larger width.
  ConnectOp emitTruncatingConnect(Value dest, Value src);

  void recursivePartialConnect(Value a, FIRRTLType aType, Value b,
                               FIRRTLType bType, unsigned aIndex,
                               unsigned bIndex, bool aFlip = false);
};
} // end anonymous namespace

void TypeLoweringVisitor::lowerModule(Operation *op) {
  if (auto module = dyn_cast<FModuleOp>(op))
    return visitDecl(module);
  if (auto extModule = dyn_cast<FExtModuleOp>(op))
    return visitDecl(extModule);
}

void TypeLoweringVisitor::visitDecl(FModuleOp module) {
  auto *body = module.getBodyBlock();

  ImplicitLocOpBuilder theBuilder(module.getLoc(), context);
  builder = &theBuilder;

  // Lower the module block arguments.
  SmallVector<BlockArgument, 8> args(body->args_begin(), body->args_end());
  originalNumModuleArgs = args.size();
  for (auto arg : args)
    if (auto type = arg.getType().dyn_cast<FIRRTLType>())
      lowerArg(module, arg, type);

  // Lower the operations.
  for (auto &op : body->getOperations()) {
    builder->setInsertionPoint(&op);
    builder->setLoc(op.getLoc());
    dispatchVisitor(&op);
  }

  // Remove ops that have been lowered. Erasing in reverse order so we don't
  // have to worry about calling dropAllUses before deleting an operation.
  for (auto *op : llvm::reverse(opsToRemove))
    op->erase();

  if (argsToRemove.empty())
    return;

  // Remove block args that have been lowered.
  body->eraseArguments(argsToRemove);

  // Remember the original argument attributess.
  SmallVector<NamedAttribute, 8> originalArgAttrs;
  DictionaryAttr originalAttrs = module->getAttrDictionary();

  // Copy over any attributes that weren't original argument attributes.
  auto *argAttrBegin = originalArgAttrs.begin();
  auto *argAttrEnd = originalArgAttrs.end();
  for (auto attr : originalAttrs)
    if (std::lower_bound(argAttrBegin, argAttrEnd, attr) == argAttrEnd)
      // Drop old "portNames", directions, and argument attributes.  These are
      // handled differently below.
      if (attr.first != "portNames" && attr.first != direction::attrKey &&
          attr.first != mlir::function_like_impl::getArgDictAttrName())
        newModuleAttrs.push_back(attr);

  newModuleAttrs.push_back(NamedAttribute(Identifier::get("portNames", context),
                                          builder->getArrayAttr(newArgNames)));
  newModuleAttrs.push_back(
      NamedAttribute(Identifier::get(direction::attrKey, context),
                     direction::packAttribute(newArgDirections, context)));

  // Attach new argument attributes.
  newModuleAttrs.push_back(NamedAttribute(
      builder->getIdentifier(mlir::function_like_impl::getArgDictAttrName()),
      builder->getArrayAttr(newArgAttrs)));

  // Update the module's attributes.
  module->setAttrs(newModuleAttrs);
  newModuleAttrs.clear();

  // Keep the module's type up-to-date.
  auto moduleType = builder->getFunctionType(body->getArgumentTypes(), {});
  module->setAttr(module.getTypeAttrName(), TypeAttr::get(moduleType));
}

//===----------------------------------------------------------------------===//
// Lowering module block arguments.
//===----------------------------------------------------------------------===//

// Lower arguments with bundle type by flattening them.
void TypeLoweringVisitor::lowerArg(FModuleOp module, BlockArgument arg,
                                   FIRRTLType type) {
  unsigned argNumber = arg.getArgNumber();

  // Flatten any bundle types.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(type, fieldTypes);

  for (auto field : fieldTypes) {

    // Create new block arguments.
    auto type = field.type;
    // Flip the direction if the field is an output.
    auto direction = (Direction)(
        (unsigned)getModulePortDirection(module, argNumber) ^ field.isOutput);
    auto newValue = addArg(module, type, argNumber, direction, field.suffix);

    // If this field was flattened from a bundle.
    if (!field.suffix.empty()) {
      // Map the flattened suffix for the original bundle to the new value.
      setBundleLowering(FieldRef(arg, field.fieldID), newValue);
    } else {
      // Lower any other arguments by copying them to keep the relative order.
      arg.replaceAllUsesWith(newValue);
    }
  }

  // Remember to remove the original block argument.
  argsToRemove.push_back(argNumber);
}

void TypeLoweringVisitor::visitDecl(FExtModuleOp extModule) {
  OpBuilder builder(context);

  // Create an array of the result types and results names.
  SmallVector<Type, 8> inputTypes;
  SmallVector<NamedAttribute, 8> attributes;
  SmallVector<Attribute, 8> argAttrDicts;

  SmallVector<Attribute> portNames;
  SmallVector<Direction> portDirections;
  unsigned oldArgNumber = 0;
  SmallString<8> attrNameBuf;
  for (auto &port : extModule.getPorts()) {
    // Flatten the port type.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    flattenType(port.type, fieldTypes);

    // Pre-populate argAttrs with the current arg attributes that are not
    // annotations.  Populate oldAnnotations with the current annotations.
    SmallVector<NamedAttribute> argAttrs;
    AnnotationSet oldAnnotations =
        AnnotationSet::forPort(extModule, oldArgNumber, argAttrs);

    // For each field, add record its name and type.
    for (auto field : fieldTypes) {
      Attribute pName;
      inputTypes.push_back(field.type);
      if (port.name)
        pName = builder.getStringAttr((port.getName() + field.suffix).str());
      else
        pName = builder.getStringAttr("");
      portNames.push_back(pName);
      // Flip the direction if the field is an output.
      portDirections.push_back((Direction)(
          (unsigned)getModulePortDirection(extModule, oldArgNumber) ^
          field.isOutput));

      // Populate newAnnotations with the old annotations filtered to those
      // associated with just this field.
      AnnotationSet newAnnotations =
          filterAnnotations(oldAnnotations, field.suffix);

      // Populate the new arg attributes.
      argAttrDicts.push_back(newAnnotations.getArgumentAttrDict(argAttrs));
    }
    ++oldArgNumber;
  }

  // Add port names attribute.
  attributes.push_back(
      {Identifier::get(mlir::function_like_impl::getArgDictAttrName(), context),
       builder.getArrayAttr(argAttrDicts)});
  attributes.push_back(
      {Identifier::get("portNames", context), builder.getArrayAttr(portNames)});
  attributes.push_back({Identifier::get(direction::attrKey, context),
                        direction::packAttribute(portDirections, context)});

  // Copy over any lingering attributes which are not "portNames", directions,
  // or argument attributes.
  for (auto a : extModule->getAttrs()) {
    if (a.first == "portNames" || a.first == direction::attrKey ||
        a.first == mlir::function_like_impl::getArgDictAttrName())
      continue;
    attributes.push_back(a);
  }

  // Set the attributes.
  extModule->setAttrs(builder.getDictionaryAttr(attributes));

  // Set the type and then bulk set all the names.
  extModule.setType(builder.getFunctionType(inputTypes, {}));
}

//===----------------------------------------------------------------------===//
// Lowering operations.
//===----------------------------------------------------------------------===//

// Lower instance operations in the same way as module block arguments. Bundles
// are flattened, and other arguments are copied to keep the relative order. By
// ensuring both lowerings are the same, we can process every module in the
// circuit in parallel, and every instance will have the correct ports.
void TypeLoweringVisitor::visitDecl(InstanceOp op) {
  // Create a new, flat bundle type for the new result.
  SmallVector<Type, 8> resultTypes;
  SmallVector<StringAttr, 8> resultNames;
  SmallVector<size_t, 8> numFieldsPerResult;
  SmallVector<unsigned, 8> resultFieldIDs;
  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    // Flatten any nested bundle types the usual way.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    flattenType(op.getType(i).cast<FIRRTLType>(), fieldTypes);

    for (auto field : fieldTypes) {
      // Store the flat type for the new bundle type.
      resultNames.push_back(builder->getStringAttr(field.suffix));
      resultTypes.push_back(field.getPortType());
      resultFieldIDs.push_back(field.fieldID);
    }
    numFieldsPerResult.push_back(fieldTypes.size());
  }

  auto newInstance = builder->create<InstanceOp>(
      resultTypes, op.moduleNameAttr(), op.nameAttr(), op.annotations());

  // Record the mapping of each old result to each new result.
  size_t nextResult = 0;
  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    // If this result was a non-bundle value, just RAUW it.
    if (numFieldsPerResult[i] == 1 &&
        resultNames[nextResult].getValue().empty()) {
      op.getResult(i).replaceAllUsesWith(newInstance.getResult(nextResult));
      ++nextResult;
      continue;
    }

    // Otherwise lower bundles.
    for (size_t j = 0, e = numFieldsPerResult[i]; j != e; ++j) {
      auto newPortName = resultNames[nextResult].getValue();
      // Drop the leading underscore.
      newPortName = newPortName.drop_front(1);
      // Map the flattened suffix for the original bundle to the new value.
      setBundleLowering(FieldRef(op.getResult(i), resultFieldIDs[nextResult]),
                        newInstance.getResult(nextResult));
      ++nextResult;
    }
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

/// Lower memory operations. A new memory is created for every leaf
/// element in a memory's data type.
void TypeLoweringVisitor::visitDecl(MemOp op) {
  auto type = op.getDataType();
  auto depth = op.depth();

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(type, fieldTypes);

  // Mutable store of the types of the ports of a new memory. This is
  // cleared and re-used.
  SmallVector<Type, 4> resultPortTypes;
  llvm::SmallSetVector<Attribute, 4> resultPortNames;

  // Insert a unique port into resultPortNames with base name nameStr.
  auto uniquePortName = [&](StringRef baseName) {
    size_t i = 0;
    std::string suffix = "";
    while (!resultPortNames.insert(
        builder->getStringAttr(baseName.str() + suffix)))
      suffix = std::to_string(i++);
  };

  // Store any new wires created during lowering. This ensures that
  // wires are re-used if they already exist.
  llvm::StringMap<Value> newWires;

  // Loop over the leaf aggregates.
  for (auto field : fieldTypes) {

    // Determine the new port type for this memory. New ports are
    // constructed by checking the kind of the memory.
    resultPortTypes.clear();
    resultPortNames.clear();
    for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
      auto kind = op.getPortKind(i);
      auto name = op.getPortName(i);

      // Any read or write ports are just added.
      if (kind != MemOp::PortKind::ReadWrite) {
        resultPortTypes.push_back(op.getTypeForPort(depth, field.type, kind));
        uniquePortName(name.getValue());
        continue;
      }

      // Any readwrite ports are lowered to 1x read and 1x write.
      resultPortTypes.push_back(
          op.getTypeForPort(depth, field.type, MemOp::PortKind::Read));
      resultPortTypes.push_back(
          op.getTypeForPort(depth, field.type, MemOp::PortKind::Write));

      auto nameStr = name.getValue().str();
      uniquePortName(nameStr + "_r");
      uniquePortName(nameStr + "_w");
    }

    // Construct the new memory for this flattened field.
    //
    // TODO: Annotations are just copied to the lowered memory.
    // Change this to copy all global annotations and only those which
    // target specific ports.
    auto newName = op.name().str() + field.suffix;
    auto newMem = builder->create<MemOp>(
        resultPortTypes, op.readLatencyAttr(), op.writeLatencyAttr(),
        op.depthAttr(), op.ruwAttr(),
        builder->getArrayAttr(resultPortNames.getArrayRef()),
        builder->getStringAttr(newName), op.annotations());

    // Setup the lowering to the new memory. We need to track both the
    // new memory index ("i") and the old memory index ("j") to deal
    // with the situation where readwrite ports have been split into
    // separate ports.
    for (size_t i = 0, j = 0, e = newMem.getNumResults(); i != e; ++i, ++j) {

      BundleType underlying = newMem.getResult(i)
                                  .getType()
                                  .cast<FIRRTLType>()
                                  .getPassiveType()
                                  .cast<BundleType>();

      auto kind = newMem.getPortKind(newMem.getPortName(i).getValue());
      auto oldKind = op.getPortKind(op.getPortName(j).getValue());
      auto oldType = op.getResult(j)
                         .getType()
                         .cast<FIRRTLType>()
                         .getPassiveType()
                         .cast<BundleType>();

      auto skip = kind == MemOp::PortKind::Write &&
                  oldKind == MemOp::PortKind::ReadWrite;

      // Loop over all elements in the port. Because readwrite ports
      // have been split, this only needs to deal with the fields of
      // read or write ports. If the port is replacing a readwrite
      // port, then this is linked against the old field.
      for (auto elt : underlying.getElements()) {

        auto oldName = elt.name.getValue();
        if (oldKind == MemOp::PortKind::ReadWrite) {
          if (oldName == "mask")
            oldName = "wmask";
          if (oldName == "data" && kind == MemOp::PortKind::Read)
            oldName = "rdata";
          if (oldName == "data" && kind == MemOp::PortKind::Write)
            oldName = "wdata";
        }
        // Get the element ID of the old port.
        auto elementID = oldType.getFieldID(*oldType.getElementIndex(oldName));

        auto getWire = [&](FIRRTLType type,
                           const std::string &wireName) -> Value {
          auto wire = newWires[wireName];
          if (!wire) {
            wire = builder->create<WireOp>(
                type.getPassiveType(), newMem.name().str() + "_" + wireName);
            newWires[wireName] = wire;
          }
          return wire;
        };

        // These ports ("addr", "clk", "en") require special
        // handling. When these are lowered, they result in multiple
        // new connections. E.g., an assignment to a clock needs to be
        // split into an assignment to all clocks. This is handled by
        // creating a dummy wire, setting the dummy wire as the
        // lowering target, and then connecting every new port
        // subfield to that.
        if (oldName == "clk" || oldName == "en" || oldName == "addr") {
          FIRRTLType theType = elt.type.getPassiveType();

          // Construct a new wire if needed.
          auto wireName =
              (op.getPortName(j).getValue().str() + "_" + oldName).str();
          auto wire = getWire(theType, wireName);

          if (!(oldKind == MemOp::PortKind::ReadWrite &&
                kind == MemOp::PortKind::Write))
            setBundleLowering(FieldRef(op.getResult(j), elementID), wire);

          // Handle "en" specially if this used to be a readwrite port.
          if (oldKind == MemOp::PortKind::ReadWrite && oldName == "en") {
            auto wmode =
                getWire(theType, op.getPortName(j).getValue().str() + "_wmode");
            if (!skip) {
              auto wmodeID =
                  oldType.getFieldID(*oldType.getElementIndex("wmode"));
              setBundleLowering(FieldRef(op.getResult(j), wmodeID), wmode);
            }
            Value gate;
            if (kind == MemOp::PortKind::Read)
              gate = builder->create<NotPrimOp>(wmode);
            else
              gate = wmode;
            wire = builder->create<AndPrimOp>(wire, gate);
          }

          builder->create<ConnectOp>(
              builder->create<SubfieldOp>(newMem.getResult(i), elt.name), wire);
          continue;
        }

        // Data ports ("data", "mask") are trivially lowered because
        // each data leaf winds up in a new, separate memory. No wire
        // creation is needed.
        setBundleLowering(
            FieldRef(op.getResult(j), elementID + field.fieldID),
            builder->create<SubfieldOp>(newMem.getResult(i), elt.name));
      }

      // Don't increment the index of the old memory if this is the
      // first, new port (the read port).
      if (kind == MemOp::PortKind::Read &&
          oldKind == MemOp::PortKind::ReadWrite)
        --j;
    }
  }

  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitDecl(NodeOp op) {
  Value result = op.result();

  // Attempt to get the bundle types, potentially unwrapping an outer flip type
  // that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(result.getType());

  // If the node is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Loop over the leaf aggregates.
  auto name = op.name().str();
  for (auto field : fieldTypes) {
    SmallString<16> loweredName;
    if (!name.empty())
      loweredName = name + field.suffix;
    // For all annotations on the parent op, filter them based on the target
    // attribute.
    SmallVector<Attribute> loweredAttrs;
    filterAnnotations(op.annotations(), loweredAttrs, field.suffix);
    auto initializer = getBundleLowering(FieldRef(op.input(), field.fieldID));
    auto node = builder->create<NodeOp>(field.type, initializer, loweredName,
                                        loweredAttrs);
    setBundleLowering(FieldRef(result, field.fieldID), node);
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

/// Lower a wire op with a bundle to mutliple non-bundled wires.
void TypeLoweringVisitor::visitDecl(WireOp op) {
  Value result = op.result();

  // Attempt to get the bundle types, potentially unwrapping an outer flip type
  // that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(result.getType());

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Loop over the leaf aggregates.
  auto name = op.name().str();
  for (auto field : fieldTypes) {
    SmallString<16> loweredName;
    if (!name.empty())
      loweredName = name + field.suffix;
    SmallVector<Attribute> loweredAttrs;
    // For all annotations on the parent op, filter them based on the target
    // attribute.
    filterAnnotations(op.annotations(), loweredAttrs, field.suffix);
    auto wire = builder->create<WireOp>(field.type, loweredName, loweredAttrs);
    setBundleLowering(FieldRef(result, field.fieldID), wire);
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
void TypeLoweringVisitor::visitDecl(RegOp op) {
  Value result = op.result();

  // Attempt to get the bundle types, potentially unwrapping an outer flip type
  // that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(result.getType());

  // If the reg is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Loop over the leaf aggregates.
  auto name = op.name().str();
  for (auto field : fieldTypes) {
    SmallString<16> loweredName;
    if (!name.empty())
      loweredName = name + field.suffix;
    SmallVector<Attribute> loweredAttrs;
    // For all annotations on the parent op, filter them based on the target
    // attribute.
    filterAnnotations(op.annotations(), loweredAttrs, field.suffix);
    setBundleLowering(FieldRef(result, field.fieldID),
                      builder->create<RegOp>(field.getPortType(), op.clockVal(),
                                             loweredName, loweredAttrs));
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

/// Lower a RegReset op with a bundle to multiple non-bundled RegResets.
void TypeLoweringVisitor::visitDecl(RegResetOp op) {
  Value result = op.result();

  // Attempt to get the bundle types, potentially unwrapping an outer flip type
  // that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(result.getType());

  // If the RegReset is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Loop over the leaf aggregates.
  auto name = op.name().str();
  for (auto field : fieldTypes) {
    std::string loweredName = "";
    if (!name.empty())
      loweredName = name + field.suffix;
    auto resetValLowered =
        getBundleLowering(FieldRef(op.resetValue(), field.fieldID));
    setBundleLowering(FieldRef(result, field.fieldID),
                      builder->create<RegResetOp>(
                          field.getPortType(), op.clockVal(), op.resetSignal(),
                          resetValLowered, loweredName, op.annotations()));
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

// Lower the subfield operation. This is accomplished by storing value and
// suffix mappings that point to the flattened value. If the subfield op is
// accessing the leaf field of a bundle, it replaces all uses with the flattened
// value. Otherwise, it flattens the rest of the bundle and adds the flattened
// values to the mapping for each partial suffix
void TypeLoweringVisitor::visitExpr(SubfieldOp op) {
  // Flatten any nested bundle types the usual way.
  FIRRTLType resultType = op.getType();
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Get the input bundle type.
  auto bundleType = op.input().getType().cast<BundleType>();

  // Get the base element ID of input bundle.
  auto parentID =
      bundleType.getFieldID(*bundleType.getElementIndex(op.fieldname()));

  for (auto field : fieldTypes) {
    // Get the lowered value of the current field from the input value.
    auto newValue =
        getBundleLowering(FieldRef(op.input(), parentID + field.fieldID));

    // If we are at the leaf of a bundle.
    if (field.suffix.empty())
      // Replace the result with the flattened value.
      op.replaceAllUsesWith(newValue);
    else
      // Map the partial suffix for the result value to the flattened value.
      setBundleLowering(FieldRef(op, field.fieldID), newValue);
  }
  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

// Gracefully die on subaccess operations
void TypeLoweringVisitor::visitExpr(SubaccessOp op) {
  op.emitError("SubaccessOp not handled.");
  // We need to do enough transformation to not segfault
  // Lower operation to an access of item 0
  auto input = op.input();
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(input.getType().cast<FIRRTLType>(), fieldTypes);
  op.replaceAllUsesWith(
      getBundleLowering(FieldRef(input, fieldTypes[0].fieldID)));
  opsToRemove.push_back(op);
}

// This is currently the same lowering as SubfieldOp, but using a fieldname
// derived from the fixed index.
//
// TODO: Unify this and SubfieldOp handling.
void TypeLoweringVisitor::visitExpr(SubindexOp op) {

  // Flatten any nested bundle types the usual way.
  FIRRTLType resultType = op.getType();
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Get the input vector type.
  auto vectorType = op.input().getType().cast<FVectorType>();

  // Get the base element ID of the input vector.
  auto parentID = vectorType.getFieldID(op.index());

  for (auto field : fieldTypes) {
    // Get the lowered value of the current field from the input value.
    auto newValue =
        getBundleLowering(FieldRef(op.input(), parentID + field.fieldID));

    // If we are at the leaf of a bundle.
    if (field.suffix.empty())
      // Replace the result with the flattened value.
      op.replaceAllUsesWith(newValue);
    else
      // Map the partial suffix for the result value to the flattened value.
      setBundleLowering(FieldRef(op, field.fieldID), newValue);
  }
  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitExpr(MuxPrimOp op) {
  // Attempt to get the bundle types, potentially unwrapping an outer flip type
  // that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(op.getType());

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  // Get a string name for each result.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Get each lhs value.
  SmallVector<std::pair<Value, bool>, 8> highValues;
  getAllBundleLowerings(op.high(), highValues);

  // Get each rhs value.
  SmallVector<std::pair<Value, bool>, 8> lowValues;
  getAllBundleLowerings(op.low(), lowValues);

  // Create a mux op for each element.
  auto result = op.result();
  auto sel = op.sel();
  for (auto it : llvm::zip(highValues, lowValues, fieldTypes)) {
    auto field = std::get<2>(it);
    auto muxOp = builder->create<MuxPrimOp>(sel, std::get<0>(it).first,
                                            std::get<1>(it).first);
    setBundleLowering(FieldRef(result, field.fieldID), muxOp);
  }
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitExpr(AsPassivePrimOp op) {
  // Attempt to get the bundle types, potentially unwrapping an outer flip type
  // that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(op.getType());

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  // Get a string name for each result.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Get each value.
  SmallVector<std::pair<Value, bool>, 8> inputValues;
  getAllBundleLowerings(op.input(), inputValues);

  // Create a cast op for each element.
  auto result = op.result();
  for (auto it : llvm::zip(inputValues, fieldTypes)) {
    auto field = std::get<1>(it);
    auto passOp = builder->create<AsPassivePrimOp>(std::get<0>(it).first);
    setBundleLowering(FieldRef(result, field.fieldID), passOp);
  }
  opsToRemove.push_back(op);
}

// Lowering connects only has to deal with one special case: connecting two
// bundles. This situation should only arise when both of the arguments are a
// bundle that was:
//   a) originally a block argument
//   b) originally an instance's port
//   c) originally from a duplex operation, like a wire or register.
//
// When two such bundles are connected, none of the subfield visits have a
// chance to lower them, so we must ensure they have the same number of
// flattened values and flatten out this connect into multiple connects.
void TypeLoweringVisitor::visitStmt(ConnectOp op) {
  Value dest = op.dest();
  Value src = op.src();

  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType destType = getCanonicalAggregateType(dest.getType());
  FIRRTLType srcType = getCanonicalAggregateType(src.getType());

  // If we aren't connecting two bundles, there is nothing to do.
  if (!destType || !srcType)
    return;

  SmallVector<std::pair<Value, bool>, 8> destValues;
  getAllBundleLowerings(dest, destValues);

  SmallVector<std::pair<Value, bool>, 8> srcValues;
  getAllBundleLowerings(src, srcValues);

  for (auto tuple : llvm::zip_first(destValues, srcValues)) {

    auto newDest = std::get<0>(tuple).first;
    auto newDestFlipped = std::get<0>(tuple).second;
    auto newSrc = std::get<1>(tuple).first;

    // Flow checks guarantee that the connection is valid.  Therfore,
    // no flow checks are needed and just the type of the LHS
    // determines whether or not this is a reverse connection.
    if (newDestFlipped)
      std::swap(newSrc, newDest);

    builder->create<ConnectOp>(newDest, newSrc);
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

/// This creates a connect operation between two values and emits a truncation
/// if the LHS is smaller than the RHS.  This code was lifted from a
/// PartialConnect canonicalization pattern.
ConnectOp TypeLoweringVisitor::emitTruncatingConnect(Value dest, Value src) {
  // Get the types.
  auto srcType = src.getType().cast<FIRRTLType>();
  auto destType = dest.getType().cast<FIRRTLType>();

  auto srcWidth = srcType.getBitWidthOrSentinel();
  auto destWidth = destType.getBitWidthOrSentinel();

  // Check if the source is larger than the destination, if so we will have to
  // emit a truncation.
  if (destType.isa<IntType>() && srcType.isa<IntType>() && srcWidth >= 0 &&
      destWidth >= 0 && destWidth < srcWidth) {
    // firrtl.tail always returns uint even for sint operands.
    IntType tmpType = destType.cast<IntType>();
    if (tmpType.isSigned())
      tmpType = UIntType::get(destType.getContext(), destWidth);
    src = builder->create<TailPrimOp>(tmpType, src, srcWidth - destWidth);
    // Insert the cast back to signed if needed.
    if (tmpType != destType)
      src = builder->create<AsSIntPrimOp>(destType, src);
  }
  return builder->create<ConnectOp>(dest, src);
}

void TypeLoweringVisitor::recursivePartialConnect(Value a, FIRRTLType aType,
                                                  Value b, FIRRTLType bType,
                                                  unsigned aID, unsigned bID,
                                                  bool aFlip) {
  TypeSwitch<FIRRTLType>(aType)
      .Case<BundleType>([&](auto aType) {
        auto bBundle = bType.dyn_cast_or_null<BundleType>();
        if (!bBundle)
          return;
        for (unsigned aIndex = 0, e = aType.getNumElements(); aIndex < e;
             ++aIndex) {
          auto aField = aType.getElements()[aIndex].name.getValue();
          auto bIndex = bBundle.getElementIndex(aField);
          if (!bIndex)
            continue;
          auto &aElt = aType.getElements()[aIndex];
          auto &bElt = bBundle.getElements()[*bIndex];
          recursivePartialConnect(
              a, aElt.type, b, bElt.type, aID + aType.getFieldID(aIndex),
              bID + bBundle.getFieldID(*bIndex), aFlip ^ aElt.isFlip);
        }
      })
      .Case<FVectorType>([&](auto aType) {
        auto bVector = bType.dyn_cast_or_null<FVectorType>();
        if (!bVector)
          return;

        auto e = std::min<unsigned>(aType.getNumElements(),
                                    bVector.getNumElements());
        for (size_t i = 0; i != e; ++i) {
          recursivePartialConnect(
              a, aType.getElementType(), b, bVector.getElementType(),
              aID + aType.getFieldID(i), bID + bVector.getFieldID(i), aFlip);
        }
      })
      .Case<AnalogType>([&](auto analogType) {
        SmallVector<Value, 2> operands{getBundleLowering(FieldRef(a, aID)),
                                       getBundleLowering(FieldRef(b, bID))};
        builder->create<AttachOp>(operands);
      })
      .Default([&](auto) {
        auto newA = getBundleLowering(FieldRef(a, aID));
        auto newB = getBundleLowering(FieldRef(b, bID));
        if (aFlip)
          std::swap(newA, newB);
        // We transform all partial connects into regular connect statements,
        // however, partial connect allows the LHS statement to be a smaller
        // width than the RHS. We might need to truncate the RHS value.
        emitTruncatingConnect(newA, newB);
      });
}

void TypeLoweringVisitor::visitStmt(PartialConnectOp op) {
  Value dest = op.dest();
  Value src = op.src();
  auto destType = dest.getType().cast<FIRRTLType>();
  auto srcType = src.getType().cast<FIRRTLType>();

  // Partial connects are completely removed and replaced by regular connects.
  // This makes this one of the few ops that actually has to do something during
  // this pass the types are not bundles.
  if (destType.isa<BundleType, FVectorType>()) {
    // Bundle types have to be recursively lowered.
    this->recursivePartialConnect(dest, destType, src, srcType.getPassiveType(),
                                  0, 0);
  } else if (destType.isa<AnalogType>()) {
    // If we are connecting analogs, replace with attach.
    SmallVector<Value, 2> operands{dest, src};
    builder->create<AttachOp>(operands);
  } else {
    // Replace this partial connect with a regular connect.
    emitTruncatingConnect(dest, src);
  }
  opsToRemove.push_back(op);
}

// Lowering invalid may need to create a new invalid for each field
void TypeLoweringVisitor::visitExpr(InvalidValueOp op) {
  Value result = op.result();

  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(result.getType());

  // If we aren't connecting two bundles, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, fieldTypes);

  // Loop over the leaf aggregates.
  for (auto field : fieldTypes) {
    setBundleLowering(FieldRef(result, field.fieldID),
                      builder->create<InvalidValueOp>(field.getPortType()));
  }

  // Remember to remove the original op.
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitStmt(WhenOp op) {
  // The WhenOp itself does not require any lowering, the only value it uses is
  // a one-bit predicate.  Recursively visit all regions so internal operations
  // are lowered.

  // Visit operations in the then block.
  for (auto &op : op.getThenBlock()) {
    builder->setInsertionPoint(&op);
    builder->setLoc(op.getLoc());
    dispatchVisitor(&op);
  }

  // If there is no else block, return.
  if (!op.hasElseRegion())
    return;

  // Visit operations in the else block.
  for (auto &op : op.getElseBlock()) {
    builder->setInsertionPoint(&op);
    builder->setLoc(op.getLoc());
    dispatchVisitor(&op);
  }
}

//===----------------------------------------------------------------------===//
// Helpers to manage state.
//===----------------------------------------------------------------------===//

// Creates and returns a new block argument of the specified type to the
// module. This also maintains the name attribute for the new argument,
// possibly with a new suffix appended.
Value TypeLoweringVisitor::addArg(FModuleOp module, Type type,
                                  unsigned oldArgNumber, Direction direction,
                                  StringRef nameSuffix) {
  Block *body = module.getBodyBlock();

  // Append the new argument.
  auto newValue = body->addArgument(type);

  // Save the name attribute for the new argument.
  StringAttr nameAttr = getModulePortName(module, oldArgNumber);
  Attribute newArg =
      builder->getStringAttr(nameAttr.getValue().str() + nameSuffix.str());
  newArgNames.push_back(newArg);
  newArgDirections.push_back(direction);

  // Decode the annotations and any additional attributes.
  SmallVector<NamedAttribute> attributes;
  auto annotations = AnnotationSet::forPort(module, oldArgNumber, attributes);

  AnnotationSet newAnnotations = filterAnnotations(annotations, nameSuffix);

  // Populate the new arg attributes.
  newArgAttrs.push_back(newAnnotations.getArgumentAttrDict(attributes));
  return newValue;
}

// Store the mapping from a bundle typed value to a mapping from its field
// names to flat values.
void TypeLoweringVisitor::setBundleLowering(FieldRef fieldRef, Value newValue) {
  auto &entry = loweredBundleValues[fieldRef];
  if (entry == newValue)
    return;
  assert(!entry && "bundle lowering has already been set");
  entry = newValue;
}

// For a mapped bundle typed value and a flat subfield name, retrieve and
// return the flat value if it exists.
Value TypeLoweringVisitor::getBundleLowering(FieldRef fieldRef) {
  auto &entry = loweredBundleValues[fieldRef];
#ifndef NDEBUG
  if (!entry) {
    {
      auto value = fieldRef.getValue();
      auto diag =
          mlir::emitError(value.getLoc(), "bundle lowering was not set");
      if (auto op = value.getDefiningOp())
        diag.attachNote(op->getLoc()) << "see current operation: " << op;
    }
    llvm::report_fatal_error("bundle lowering was not set");
  }
#endif
  return entry;
}

// For a mapped aggregate typed value, retrieve and return the flat values for
// each field.
void TypeLoweringVisitor::getAllBundleLowerings(
    Value value, SmallVectorImpl<std::pair<Value, bool>> &results) {
  TypeSwitch<FIRRTLType>(getCanonicalAggregateType(value.getType()))
      .Case<BundleType, FVectorType>([&](auto aggregateType) {
        // Flatten the original value's bundle type.
        SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
        flattenType(aggregateType, fieldTypes);

        for (auto element : fieldTypes) {
          // Store the resulting lowering for this flat value.
          results.push_back(
              {getBundleLowering(FieldRef(value, element.fieldID)),
               element.isOutput});
        }
      })
      .Default([&](auto) {});
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerTypesPass : public LowerFIRRTLTypesBase<LowerTypesPass> {
  void runOnOperation() override;

private:
  void runAsync();
  void runSync();
};
} // end anonymous namespace

void LowerTypesPass::runAsync() {
  // Collect the operations to iterate in a vector. We can't use parallelFor
  // with the regular op list, since it requires a RandomAccessIterator. This
  // also lets us use parallelForEachN, which means we don't have to
  // llvm::enumerate the ops with their index. TODO(mlir): There should really
  // be a way to do this without collecting the operations first.
  auto &body = getOperation().getBody()->getOperations();
  std::vector<Operation *> ops;
  llvm::for_each(body, [&](Operation &op) { ops.push_back(&op); });

  mlir::ParallelDiagnosticHandler diagHandler(&getContext());
  llvm::parallelForEachN(0, ops.size(), [&](auto index) {
    // Notify the handler the op index and then perform lowering.
    diagHandler.setOrderIDForThread(index);
    TypeLoweringVisitor(&getContext()).lowerModule(ops[index]);
    diagHandler.eraseOrderIDForThread();
  });
}

void LowerTypesPass::runSync() {
  auto circuit = getOperation();
  for (auto &op : circuit.getBody()->getOperations()) {
    TypeLoweringVisitor(&getContext()).lowerModule(&op);
  }
}

// This is the main entrypoint for the lowering pass.
void LowerTypesPass::runOnOperation() {
  if (getContext().isMultithreadingEnabled()) {
    runAsync();
  } else {
    runSync();
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLTypesPass() {
  return std::make_unique<LowerTypesPass>();
}
