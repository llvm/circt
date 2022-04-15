//===- Dedup.cpp - FIRRTL module deduping -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FIRRTL module deduplication.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SHA256.h"

using namespace circt;
using namespace firrtl;
using hw::InnerRefAttr;

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

llvm::raw_ostream &printHex(llvm::raw_ostream &stream,
                            ArrayRef<uint8_t> bytes) {
  // Print the hash on a single line.
  return stream << format_bytes(bytes, llvm::None, 32) << "\n";
}

llvm::raw_ostream &printHash(llvm::raw_ostream &stream, llvm::SHA256 &data) {
  return printHex(stream, data.result());
}

llvm::raw_ostream &printHash(llvm::raw_ostream &stream, std::string data) {
  ArrayRef bytes(reinterpret_cast<const uint8_t *>(data.c_str()),
                 data.length());
  return printHex(stream, bytes);
}

struct StructuralHasher {
  explicit StructuralHasher(MLIRContext *context) {
    portTypesAttr = StringAttr::get(context, "portTypes");
    nonessentialAttributes.insert(StringAttr::get(context, "annotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "name"));
    nonessentialAttributes.insert(StringAttr::get(context, "portAnnotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "portNames"));
    nonessentialAttributes.insert(StringAttr::get(context, "portSyms"));
    nonessentialAttributes.insert(StringAttr::get(context, "sym_name"));
    nonessentialAttributes.insert(StringAttr::get(context, "inner_sym"));
  };

  std::array<uint8_t, 32> hash(FModuleLike module) {
    update(&(*module));
    auto hash = sha.final();
    reset();
    return hash;
  }

private:
  void reset() {
    currentIndex = 0;
    indexes.clear();
    sha.init();
  }

  void update(const void *pointer) {
    auto *addr = reinterpret_cast<const uint8_t *>(&pointer);
    sha.update(ArrayRef(addr, sizeof pointer));
  }

  void update(size_t value) {
    auto *addr = reinterpret_cast<const uint8_t *>(&value);
    sha.update(ArrayRef(addr, sizeof value));
  }

  void update(TypeID typeID) { update(typeID.getAsOpaquePointer()); }

  // NOLINTNEXTLINE(misc-no-recursion)
  void update(BundleType type) {
    update(type.getTypeID());
    for (auto &element : type.getElements()) {
      update(element.isFlip);
      update(element.type);
    }
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void update(Type type) {
    if (auto bundle = type.dyn_cast<BundleType>())
      return update(bundle);
    update(type.getAsOpaquePointer());
  }

  void update(BlockArgument arg) { indexes[arg] = currentIndex++; }

  void update(OpResult result) {
    indexes[result] = currentIndex++;
    update(result.getType());
  }

  void update(OpOperand &operand) {
    // We hash the value's index as it apears in the block.
    auto it = indexes.find(operand.get());
    assert(it != indexes.end() && "op should have been previously hashed");
    update(it->second);
  }

  void update(DictionaryAttr dict) {
    for (auto namedAttr : dict) {
      auto name = namedAttr.getName();
      auto value = namedAttr.getValue();
      // Skip names and annotations.
      if (nonessentialAttributes.contains(name))
        continue;
      // Hash the port types.
      if (name == portTypesAttr) {
        auto portTypes = value.cast<ArrayAttr>().getAsValueRange<TypeAttr>();
        for (auto type : portTypes)
          update(type);
        continue;
      }
      // Hash the interned pointer.
      update(name.getAsOpaquePointer());
      update(value.getAsOpaquePointer());
    }
  }

  void update(Block &block) {
    // Hash the block arguments.
    for (auto arg : block.getArguments())
      update(arg);
    // Hash the operations in the block.
    for (auto &op : block)
      update(&op);
  }

  void update(mlir::OperationName name) {
    // Operation names are interned.
    update(name.getAsOpaquePointer());
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void update(Operation *op) {
    update(op->getName());
    update(op->getAttrDictionary());
    // Hash the operands.
    for (auto &operand : op->getOpOperands())
      update(operand);
    // Hash the regions. We need to make sure an empty region doesn't hash the
    // same as no region, so we include the number of regions.
    update(op->getNumRegions());
    for (auto &region : op->getRegions())
      for (auto &block : region.getBlocks())
        update(block);
    // Record any op results.
    for (auto result : op->getResults())
      update(result);
  }

  // Every value is assigned a unique id based on their order of appearance.
  unsigned currentIndex = 0;
  DenseMap<Value, unsigned> indexes;

  // This is a set of every attribute we should ignore.
  DenseSet<Attribute> nonessentialAttributes;
  // This is a cached "portTypes" string attr.
  StringAttr portTypesAttr;

  // This is the actual running hash calculation. This is a stateful element
  // that should be reinitialized after each hash is produced.
  llvm::SHA256 sha;
};

//===----------------------------------------------------------------------===//
// Equivalence
//===----------------------------------------------------------------------===//

/// This class is for reporting differences between two modules which should
/// have been deduplicated.
struct Equivalence {
  Equivalence(MLIRContext *context, InstanceGraph &instanceGraph)
      : instanceGraph(instanceGraph) {
    noDedupClass =
        StringAttr::get(context, "firrtl.transforms.NoDedupAnnotation");
    portTypesAttr = StringAttr::get(context, "portTypes");
    nonessentialAttributes.insert(StringAttr::get(context, "annotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "name"));
    nonessentialAttributes.insert(StringAttr::get(context, "portAnnotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "portNames"));
    nonessentialAttributes.insert(StringAttr::get(context, "portTypes"));
    nonessentialAttributes.insert(StringAttr::get(context, "portSyms"));
    nonessentialAttributes.insert(StringAttr::get(context, "sym_name"));
    nonessentialAttributes.insert(StringAttr::get(context, "inner_sym"));
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult check(InFlightDiagnostic &diag, const Twine &message,
                      Operation *a, BundleType aType, Operation *b,
                      BundleType bType) {
    if (aType.getNumElements() != bType.getNumElements()) {
      diag.attachNote(a->getLoc())
          << message << " bundle type has different number of elements";
      diag.attachNote(b->getLoc()) << "second operation here";
      return failure();
    }

    for (auto elementPair :
         llvm::zip(aType.getElements(), bType.getElements())) {
      auto aElement = std::get<0>(elementPair);
      auto bElement = std::get<1>(elementPair);
      if (aElement.isFlip != bElement.isFlip) {
        diag.attachNote(a->getLoc()) << message << " bundle element "
                                     << aElement.name << " flip does not match";
        diag.attachNote(b->getLoc()) << "second operation here";
        return failure();
      }

      if (failed(check(diag,
                       "bundle element \'" + aElement.name.getValue() + "'", a,
                       aElement.type, b, bElement.type)))
        return failure();
    }
    return success();
  }

  LogicalResult check(InFlightDiagnostic &diag, const Twine &message,
                      Operation *a, Type aType, Operation *b, Type bType) {
    if (aType == bType)
      return success();
    if (aType.isa<BundleType>() && bType.isa<BundleType>())
      return check(diag, message, a, aType.cast<BundleType>(), b,
                   bType.cast<BundleType>());
    diag.attachNote(a->getLoc())
        << message << " types don't match, first type is " << aType;
    diag.attachNote(b->getLoc()) << "second type is " << bType;
    return failure();
  }

  LogicalResult check(InFlightDiagnostic &diag, BlockAndValueMapping &map,
                      Operation *a, Block &aBlock, Operation *b,
                      Block &bBlock) {

    // Block Arguments.
    if (aBlock.getNumArguments() != bBlock.getNumArguments()) {
      diag.attachNote(a->getLoc())
          << "modules have a different number of ports";
      diag.attachNote(b->getLoc()) << "second module here";
      return failure();
    }

    // Block argument types.
    auto portNames = a->getAttrOfType<ArrayAttr>("portNames");
    auto portNo = 0;
    for (auto argPair :
         llvm::zip(aBlock.getArguments(), bBlock.getArguments())) {
      auto &aArg = std::get<0>(argPair);
      auto &bArg = std::get<1>(argPair);
      // TODO: we should print the port number if there are no port names, but
      // there are always port names ;).
      StringRef portName;
      if (portNames) {
        if (auto portNameAttr = portNames[portNo].dyn_cast<StringAttr>())
          portName = portNameAttr.getValue();
      }
      // Assumption here that block arguments correspond to ports.
      if (failed(check(diag, "module port '" + portName + "'", a,
                       aArg.getType(), b, bArg.getType())))
        return failure();
      map.map(aArg, bArg);
      portNo++;
    }

    // Blocks operations.
    auto aIt = aBlock.begin();
    auto aEnd = aBlock.end();
    auto bIt = bBlock.begin();
    auto bEnd = bBlock.end();
    while (aIt != aEnd && bIt != bEnd)
      if (failed(check(diag, map, &*aIt++, &*bIt++)))
        return failure();
    if (aIt != aEnd) {
      diag.attachNote(aIt->getLoc()) << "first block has more operations";
      diag.attachNote(b->getLoc()) << "second block here";
      return failure();
    }
    if (bIt != bEnd) {
      diag.attachNote(bIt->getLoc()) << "second block has more operations";
      diag.attachNote(a->getLoc()) << "first block here";
      return failure();
    }
    return success();
  }

  LogicalResult check(InFlightDiagnostic &diag, BlockAndValueMapping &map,
                      Operation *a, Region &aRegion, Operation *b,
                      Region &bRegion) {
    auto aIt = aRegion.begin();
    auto aEnd = aRegion.end();
    auto bIt = bRegion.begin();
    auto bEnd = bRegion.end();

    // Region blocks.
    while (aIt != aEnd && bIt != bEnd)
      if (failed(check(diag, map, a, *aIt++, b, *bIt++)))
        return failure();
    if (aIt != aEnd || bIt != bEnd) {
      diag.attachNote(a->getLoc())
          << "operation regions have different number of blocks";
      diag.attachNote(b->getLoc()) << "second operation here";
      return failure();
    }
    return success();
  }

  LogicalResult check(InFlightDiagnostic &diag, Operation *a, IntegerAttr aAttr,
                      Operation *b, IntegerAttr bAttr) {
    if (aAttr == bAttr)
      return success();
    auto aDirections = direction::unpackAttribute(aAttr);
    auto bDirections = direction::unpackAttribute(bAttr);
    auto portNames = a->getAttrOfType<ArrayAttr>("portNames");
    for (unsigned i = 0, e = aDirections.size(); i < e; ++i) {
      auto aDirection = aDirections[i];
      auto bDirection = bDirections[i];
      if (aDirection != bDirection) {
        auto &note = diag.attachNote(a->getLoc()) << "module port ";
        if (portNames)
          note << "'" << portNames[i].cast<StringAttr>().getValue() << "'";
        else
          note << i;
        note << " directions don't match, first direction is '"
             << direction::toString(aDirection) << "'";
        diag.attachNote(b->getLoc()) << "second direction is '"
                                     << direction::toString(bDirection) << "'";
        return failure();
      }
    }
    return success();
  }

  LogicalResult check(InFlightDiagnostic &diag, BlockAndValueMapping &map,
                      Operation *a, DictionaryAttr aDict, Operation *b,
                      DictionaryAttr bDict) {
    // Fast path.
    if (aDict == bDict)
      return success();

    DenseSet<Attribute> seenAttrs;
    for (auto namedAttr : aDict) {
      auto attrName = namedAttr.getName();
      if (nonessentialAttributes.contains(attrName))
        continue;

      auto aAttr = namedAttr.getValue();
      auto bAttr = bDict.get(attrName);
      if (!bAttr) {
        diag.attachNote(a->getLoc())
            << "second operation is missing attribute " << attrName;
        diag.attachNote(b->getLoc()) << "second operation here";
        return diag;
      }

      if (attrName == "portDirections") {
        // Special handling for the port directions attribute for better
        // error messages.
        if (failed(check(diag, a, aAttr.cast<IntegerAttr>(), b,
                         bAttr.cast<IntegerAttr>())))
          return failure();
      } else if (aAttr != bAttr) {
        diag.attachNote(a->getLoc())
            << "first operation has attribute '" << attrName.getValue()
            << "' with value " << aAttr;
        diag.attachNote(b->getLoc()) << "second operation has value " << bAttr;
        return failure();
      }
      seenAttrs.insert(attrName);
    }
    if (aDict.getValue().size() != bDict.getValue().size()) {
      for (auto namedAttr : bDict) {
        auto attrName = namedAttr.getName();
        // Skip the attribute if we don't care about this particular one or it
        // is one that is known to be in both dictionaries.
        if (nonessentialAttributes.contains(attrName) ||
            seenAttrs.contains(attrName))
          continue;
        // We have found an attribute that is only in the second operation.
        diag.attachNote(a->getLoc())
            << "first operation is missing attribute " << attrName;
        diag.attachNote(b->getLoc()) << "second operation here";
        return failure();
      }
    }
    return success();
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult check(InFlightDiagnostic &diag, InstanceOp a, InstanceOp b) {
    auto aName = a.moduleNameAttr().getAttr();
    auto bName = b.moduleNameAttr().getAttr();
    // If the modules instantiate are different we will want to know why the
    // sub module did not dedupliate. This code recursively checks the child
    // module.
    if (aName != bName) {
      diag.attachNote(a->getLoc()) << "first instance targets module " << aName;
      diag.attachNote(b->getLoc())
          << "second instance targets module " << bName;
      diag.report();
      auto aModule = instanceGraph.getReferencedModule(a);
      auto bModule = instanceGraph.getReferencedModule(b);
      // Create a new error for the submodule.
      auto newDiag = emitError(aModule->getLoc())
                     << "module " << aName << " not deduplicated with "
                     << bName;
      check(newDiag, aModule, bModule);
      return failure();
    }
    return success();
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult check(InFlightDiagnostic &diag, BlockAndValueMapping &map,
                      Operation *a, Operation *b) {
    // Operation name.
    if (a->getName() != b->getName()) {
      diag.attachNote(a->getLoc()) << "first operation is a " << a->getName();
      diag.attachNote(b->getLoc()) << "second operation is a " << b->getName();
      return failure();
    }

    // If its an instance operaiton, perform some checking and possibly
    // recurse.
    if (auto aInst = dyn_cast<InstanceOp>(a)) {
      auto bInst = cast<InstanceOp>(b);
      if (failed(check(diag, aInst, bInst)))
        return failure();
    }

    // Operation results.
    if (a->getNumResults() != b->getNumResults()) {
      diag.attachNote(a->getLoc())
          << "operations have different number of results";
      diag.attachNote(b->getLoc()) << "second operation here";
      return failure();
    }
    for (auto resultPair : llvm::zip(a->getResults(), b->getResults())) {
      auto &aValue = std::get<0>(resultPair);
      auto &bValue = std::get<1>(resultPair);
      if (failed(check(diag, "operation result", a, aValue.getType(), b,
                       bValue.getType())))
        return failure();
      map.map(aValue, bValue);
    }

    // Operations operands.
    if (a->getNumOperands() != b->getNumOperands()) {
      diag.attachNote(a->getLoc())
          << "operations have different number of operands";
      diag.attachNote(b->getLoc()) << "second operation here";
      return failure();
    }
    for (auto operandPair : llvm::zip(a->getOperands(), b->getOperands())) {
      auto &aValue = std::get<0>(operandPair);
      auto &bValue = std::get<1>(operandPair);
      if (bValue != map.lookup(aValue)) {
        diag.attachNote(a->getLoc())
            << "operations use different operands, first operand is '"
            << getFieldName(getFieldRefFromValue(aValue)) << "'";
        diag.attachNote(b->getLoc())
            << "second operand is '"
            << getFieldName(getFieldRefFromValue(bValue))
            << "', but should have been '"
            << getFieldName(getFieldRefFromValue(map.lookup(aValue))) << "'";
        return failure();
      }
    }

    // Operation regions.
    if (a->getNumRegions() != b->getNumRegions()) {
      diag.attachNote(a->getLoc())
          << "operations have different number of regions";
      diag.attachNote(b->getLoc()) << "second operation here";
      return failure();
    }
    for (auto regionPair : llvm::zip(a->getRegions(), b->getRegions())) {
      auto &aRegion = std::get<0>(regionPair);
      auto &bRegion = std::get<1>(regionPair);
      if (failed(check(diag, map, a, aRegion, b, bRegion)))
        return failure();
    }

    // Operation attributes.
    if (failed(check(diag, map, a, a->getAttrDictionary(), b,
                     b->getAttrDictionary())))
      return failure();
    return success();
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void check(InFlightDiagnostic &diag, Operation *a, Operation *b) {
    BlockAndValueMapping map;
    if (AnnotationSet(a).hasAnnotation(noDedupClass)) {
      diag.attachNote(a->getLoc()) << "module marked NoDedup";
      return;
    }
    if (AnnotationSet(b).hasAnnotation(noDedupClass)) {
      diag.attachNote(b->getLoc()) << "module marked NoDedup";
      return;
    }
    if (failed(check(diag, map, a, b)))
      return;
    diag.attachNote(a->getLoc()) << "first module here";
    diag.attachNote(b->getLoc()) << "second module here";
  }

  // This is a cached "portTypes" string attr.
  StringAttr portTypesAttr;
  // This is a cached "NoDedup" annotation class string attr.
  StringAttr noDedupClass;
  // This is a set of every attribute we should ignore.
  DenseSet<Attribute> nonessentialAttributes;
  InstanceGraph &instanceGraph;
};

//===----------------------------------------------------------------------===//
// Deduplication
//===----------------------------------------------------------------------===//

struct Deduper {

  using RenameMap = DenseMap<StringAttr, StringAttr>;

  Deduper(InstanceGraph &instanceGraph, SymbolTable &symbolTable,
          NLATable *nlaTable, CircuitOp circuit)
      : context(circuit->getContext()), instanceGraph(instanceGraph),
        symbolTable(symbolTable), nlaTable(nlaTable),
        nlaBlock(circuit.getBody()),
        nonLocalString(StringAttr::get(context, "circt.nonlocal")),
        classString(StringAttr::get(context, "class")) {}

  /// Remove the "fromModule", and replace all references to it with the
  /// "toModule".  Modules should be deduplicated in a bottom-up order.  Any
  /// module which is not deduplicated needs to be recorded with the `record`
  /// call.
  void dedup(FModuleLike toModule, FModuleLike fromModule) {
    // A map of operation (e.g. wires, nodes) names which are changed, which is
    // used to update NLAs that reference the "fromModule".
    RenameMap renameMap;

    // Merge the two modules.
    mergeOps(renameMap, toModule, toModule, fromModule, fromModule);

    // Rewrite NLAs pathing through these modules to refer to the to module.
    if (auto to = dyn_cast<FModuleOp>(*toModule))
      rewriteModuleNLAs(renameMap, to, cast<FModuleOp>(*fromModule));
    else
      rewriteExtModuleNLAs(renameMap, toModule.moduleNameAttr(),
                           fromModule.moduleNameAttr());
    replaceInstances(toModule, fromModule);
  }

  /// Record the usages of any NLA's in this module, so that we may update the
  /// annotation if the parent module is deduped with another module.
  void record(FModuleLike module) {
    // Record any annotations on the module.
    recordAnnotations(module);
    // Record port annotations.
    for (auto pair : llvm::enumerate(module.getPortAnnotations()))
      for (auto anno : AnnotationSet(pair.value().cast<ArrayAttr>()))
        if (auto nlaRef = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
          targetMap[nlaRef.getAttr()] = PortAnnoTarget(module, pair.index());
    // Record any annotations in the module body.
    module->walk([&](Operation *op) { recordAnnotations(op); });
  }

private:
  /// Get a cached namespace for a module.
  ModuleNamespace &getNamespace(Operation *module) {
    auto [it, inserted] = moduleNamespaces.try_emplace(module, module);
    return it->second;
  }

  /// Record all targets which use an NLA.
  void recordAnnotations(Operation *op) {
    for (auto anno : AnnotationSet(op)) {
      if (auto nlaRef = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        // Don't record instance breadcrumbs.  We're only looking for the final
        // target of an NLA.
        if (anno.getClassAttr() == nonLocalString)
          continue;
        targetMap[nlaRef.getAttr()] = OpAnnoTarget(op);
      }
    }

    // Record port annotations if this is a mem operation.
    auto mem = dyn_cast<MemOp>(op);
    if (!mem)
      return;
    // Breadcrumbs don't appear on port annotations, so we can skip the
    // class check that we have above.
    for (auto pair : llvm::enumerate(mem.portAnnotations()))
      for (auto anno : AnnotationSet(pair.value().cast<ArrayAttr>()))
        if (auto nlaRef = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
          targetMap[nlaRef.getAttr()] = PortAnnoTarget(mem, pair.index());
  }

  /// This deletes and replaces all instances of the "fromModule" with instances
  /// of the "toModule".
  void replaceInstances(FModuleLike toModule, Operation *fromModule) {
    // Replace all instances of the other module.
    auto *fromNode = instanceGraph[fromModule];
    auto *toNode = instanceGraph[::cast<hw::HWModuleLike>(*toModule)];
    auto toModuleRef = FlatSymbolRefAttr::get(toModule.moduleNameAttr());
    for (auto *oldInstRec : llvm::make_early_inc_range(fromNode->uses())) {
      auto inst = ::cast<InstanceOp>(*oldInstRec->getInstance());
      inst.moduleNameAttr(toModuleRef);
      inst.portNamesAttr(toModule.getPortNamesAttr());
      oldInstRec->getParent()->addInstance(inst, toNode);
      oldInstRec->erase();
    }
    instanceGraph.erase(fromNode);
    fromModule->erase();
  }

  /// Look up the instantiations of this module and create an NLA for each
  /// one, appending the baseNamepath to each NLA. This is used to add more
  /// context to an already existing NLA.
  SmallVector<FlatSymbolRefAttr> createNLAs(StringAttr toModuleName,
                                            AnnoTarget to,
                                            Operation *fromModule,
                                            ArrayRef<Attribute> baseNamepath) {
    // Create an attribute array with a placeholder in the first element, where
    // the root refence of the NLA will be inserted.
    SmallVector<Attribute> namepath = {nullptr};
    namepath.append(baseNamepath.begin(), baseNamepath.end());

    auto loc = fromModule->getLoc();
    SmallVector<FlatSymbolRefAttr> nlas;
    for (auto *instanceRecord : instanceGraph[fromModule]->uses()) {
      auto parent = cast<FModuleOp>(*instanceRecord->getParent()->getModule());
      auto inst = instanceRecord->getInstance();
      namepath[0] = OpAnnoTarget(inst).getNLAReference(getNamespace(parent));
      auto arrayAttr = ArrayAttr::get(context, namepath);
      auto nla = OpBuilder::atBlockBegin(nlaBlock).create<NonLocalAnchor>(
          loc, "nla", arrayAttr);
      // Insert it into the symbol table to get a unique name.
      symbolTable.insert(nla);
      auto nlaName = nla.getNameAttr();
      auto nlaRef = FlatSymbolRefAttr::get(nlaName);
      nlas.push_back(nlaRef);
      nlaTable->insert(nla);
      targetMap[nlaName] = to;
      // Update the instance breadcrumbs.
      auto nonLocalClass = NamedAttribute(classString, nonLocalString);
      auto dict = DictionaryAttr::get(
          context, {{nonLocalString, nlaRef}, nonLocalClass});
      // Add the breadcrumb on the first instance.
      AnnotationSet instAnnos(inst);
      instAnnos.addAnnotations(dict);
      instAnnos.applyToOperation(inst);

      // Set the breadcrumb on following instances. Ignore the first element
      // which was already handled above, and the last element which does not
      // need to be breadcrumbed.
      for (auto attr : nla.namepath().getValue().drop_front().drop_back()) {
        auto innerRef = attr.cast<InnerRefAttr>();
        auto *node = instanceGraph.lookup(innerRef.getModule());
        // Find the instance referenced by the NLA.
        auto targetInstanceName = innerRef.getName();
        auto it = llvm::find_if(*node, [&](InstanceRecord *record) {
          return cast<InstanceOp>(*record->getInstance()).inner_symAttr() ==
                 targetInstanceName;
        });
        assert(it != node->end() &&
               "Instance referenced by NLA does not exist in module");
        // Commit the annotation update.
        auto inst = (*it)->getInstance();
        AnnotationSet instAnnos(inst);
        instAnnos.addAnnotations(dict);
        instAnnos.applyToOperation(inst);
      }
    }
    return nlas;
  }

  /// Look up the instantiations of this module and create an NLA for each one.
  /// This returns an array of symbol references which can be used to reference
  /// the NLAs.
  SmallVector<FlatSymbolRefAttr> createNLAs(FModuleLike toModule,
                                            StringAttr toModuleName,
                                            AnnoTarget to,
                                            FModuleLike fromModule) {
    return createNLAs(toModuleName, to, fromModule,
                      to.getNLAReference(getNamespace(toModule)));
  }

  /// Clone the annotation for each NLA in a list.
  void cloneAnnotation(SmallVector<FlatSymbolRefAttr> &nlas, Annotation anno,
                       ArrayRef<NamedAttribute> attributes,
                       unsigned nonLocalIndex,
                       SmallVector<Annotation> &newAnnotations) {
    SmallVector<NamedAttribute> mutableAttributes(attributes.begin(),
                                                  attributes.end());
    for (auto &nla : nlas) {
      // Add the new annotation.
      mutableAttributes[nonLocalIndex].setValue(nla);
      auto dict = DictionaryAttr::getWithSorted(context, mutableAttributes);
      anno.setDict(dict);
      newAnnotations.push_back(anno);
    }
  }

  /// This erases the NLA op, all breadcrumb trails, and removes the NLA from
  /// every module's NLA map, but it does not delete the NLA reference from
  /// the target operation's annotations.
  void eraseNLA(NonLocalAnchor nla) {
    auto nlaRef = FlatSymbolRefAttr::get(nla.getNameAttr());
    auto nonLocalClass = NamedAttribute(classString, nonLocalString);
    auto dict =
        DictionaryAttr::get(context, {{nonLocalString, nlaRef}, nonLocalClass});
    auto namepath = nla.namepath().getValue();
    for (auto attr : namepath.drop_back()) {
      auto innerRef = attr.cast<InnerRefAttr>();
      auto moduleName = innerRef.getModule();
      // Find the instance referenced by the NLA.
      auto *node = instanceGraph.lookup(moduleName);
      auto targetInstanceName = innerRef.getName();
      auto it = llvm::find_if(*node, [&](InstanceRecord *record) {
        return cast<InstanceOp>(*record->getInstance()).inner_symAttr() ==
               targetInstanceName;
      });
      assert(it != node->end() &&
             "Instance referenced by NLA does not exist in module");
      // Commit the annotation update.
      auto inst = (*it)->getInstance();
      AnnotationSet instAnnos(inst);
      instAnnos.removeAnnotation(dict);
      instAnnos.applyToOperation(inst);
    }
    // Erase the NLA from the leaf module's nlaMap.
    targetMap.erase(nla.getNameAttr());
    nlaTable->erase(nla);
    symbolTable.erase(nla);
  }

  /// Process all NLAs referencing the "from" module to point to the "to"
  /// module. This is used after merging two modules together.
  void addAnnotationContext(RenameMap &renameMap, FModuleOp toModule,
                            FModuleOp fromModule) {
    auto toName = toModule.getNameAttr();
    auto fromName = fromModule.getNameAttr();
    // Create a copy of the current NLAs. We will be pushing and removing
    // NLAs from this op as we go.
    auto nlas = nlaTable->lookup(fromModule.getNameAttr()).vec();
    // Change the NLA to target the toModule.
    nlaTable->renameModuleAndInnerRef(toName, fromName, renameMap);
    for (auto nla : nlas) {
      auto elements = nla.namepath().getValue();
      // If we don't need to add more context, we're done here.
      if (nla.root() != toName)
        continue;
      // We need to clone the annotation for each new NLA.
      auto target = targetMap[nla.sym_nameAttr()];
      assert(target && "Target of NLA never encountered.  All modules should "
                       "be reachable from the top module.");
      SmallVector<Attribute> namepath(elements.begin(), elements.end());
      SmallVector<Annotation> newAnnotations;
      auto nlas = createNLAs(toName, target, fromModule, namepath);
      for (auto anno : target.getAnnotations()) {
        // Find the non-local field of the annotation.
        auto [it, found] = mlir::impl::findAttrSorted(anno.begin(), anno.end(),
                                                      nonLocalString);
        if (!found || it->getValue().cast<FlatSymbolRefAttr>().getAttr() !=
                          nla.sym_nameAttr()) {
          newAnnotations.push_back(anno);
          continue;
        }
        auto nonLocalIndex = std::distance(anno.begin(), it);
        // We have to clone all the annotations referencing this op
        // SmallVector<NamedAttribute> attributes(anno.begin(), anno.end());
        cloneAnnotation(nlas, anno, ArrayRef(anno.begin(), anno.end()),
                        nonLocalIndex, newAnnotations);
      }
      AnnotationSet annotations(newAnnotations, context);
      target.setAnnotations(annotations);

      // Erase the old NLA and remove it from all breadcrumbs.
      eraseNLA(nla);
    }
  }

  /// Process all the NLAs that the two modules participate in, replacing
  /// references to the "from" module with references to the "to" module, and
  /// adding more context if necessary.
  void rewriteModuleNLAs(RenameMap &renameMap, FModuleOp toModule,
                         FModuleOp fromModule) {
    addAnnotationContext(renameMap, toModule, toModule);
    addAnnotationContext(renameMap, toModule, fromModule);
  }

  // Update all NLAs which the "from" external module participates in to the
  // "toName".
  void rewriteExtModuleNLAs(RenameMap &renameMap, StringAttr toName,
                            StringAttr fromName) {
    nlaTable->renameModuleAndInnerRef(toName, fromName, renameMap);
  }

  /// Take an annotation, and update it to be a non-local annotation.  If the
  /// annotation is already non-local and has enough context, it will be skipped
  /// for now.
  void makeAnnotationNonLocal(SmallVector<FlatSymbolRefAttr> &nlas,
                              FModuleLike toModule, StringAttr toModuleName,
                              AnnoTarget to, FModuleLike fromModule,
                              AnnoTarget from, Annotation anno,
                              SmallVector<Annotation> &newAnnotations) {
    // Start constructing a new annotation, pushing a "circt.nonLocal" field
    // into the correct spot if its not already a non-local annotation.
    SmallVector<NamedAttribute> attributes;
    int nonLocalIndex = -1;
    for (auto val : llvm::enumerate(anno)) {
      auto attr = val.value();
      // Is this field "circt.nonlocal"?
      auto compare = attr.getName().compare(nonLocalString);
      if (compare == 0) {
        // This annotation is already a non-local annotation. Record that this
        // operation uses that NLA and stop processing this annotation.
        auto nlaName = attr.getValue().cast<FlatSymbolRefAttr>().getAttr();
        targetMap[nlaName] = from;
        newAnnotations.push_back(anno);
        return;
      }
      if (compare == 1) {
        // Push an empty place holder for the non-local annotation.
        nonLocalIndex = val.index();
        attributes.push_back(NamedAttribute(nonLocalString, nonLocalString));
        break;
      }
      attributes.push_back(attr);
    }
    if (nonLocalIndex == -1) {
      // Push the "circt.nonlocal" to the last slot.
      nonLocalIndex = attributes.size();
      attributes.push_back(NamedAttribute(nonLocalString, nonLocalString));
    } else {
      // Copy the remaining annotation fields in.
      attributes.append(anno.begin() + nonLocalIndex, anno.end());
    }

    // Construct the NLAs if we don't have any yet.
    if (nlas.empty())
      nlas = createNLAs(toModule, toModuleName, to, fromModule);

    // Clone the annotation for each new NLA.
    cloneAnnotation(nlas, anno, attributes, nonLocalIndex, newAnnotations);
  }

  /// Merge the annotations of a specific target, either a operation or a port
  /// on an operation.
  void mergeAnnotations(FModuleLike toModule, AnnoTarget to,
                        AnnotationSet toAnnos, FModuleLike fromModule,
                        AnnoTarget from, AnnotationSet fromAnnos) {
    // We want to make sure all NLAs are put right above the first module.
    SmallVector<Annotation> newAnnotations;
    SmallVector<unsigned> alreadyHandled;
    // This is a lazily constructed set of NLAs used to turn a local
    // annotation into non-local annotations.
    SmallVector<FlatSymbolRefAttr> fromNLAs;
    for (auto anno : fromAnnos) {
      // If this is a breadcrumb, copy it over with no changes.
      if (anno.getClassAttr() == nonLocalString) {
        newAnnotations.push_back(anno);
        continue;
      }

      // If the ops have the same annotation, we don't have to turn it into a
      // non-local annotation.
      auto found = llvm::find(toAnnos, anno);
      if (found != toAnnos.end()) {
        alreadyHandled.push_back(std::distance(toAnnos.begin(), found));
        newAnnotations.push_back(anno);
        continue;
      }
      makeAnnotationNonLocal(fromNLAs, toModule, toModule.moduleNameAttr(), to,
                             fromModule, from, anno, newAnnotations);
    }

    // This is a helper to skip already handled annotations.
    auto *it = alreadyHandled.begin();
    auto *end = alreadyHandled.end();
    auto getNextHandledIndex = [&]() -> unsigned {
      if (it == end)
        return -1;
      return *(it++);
    };
    auto index = getNextHandledIndex();

    // Merge annotations from the other op, skipping the ones already handled.
    SmallVector<FlatSymbolRefAttr> toNLAs;
    for (auto pair : llvm::enumerate(toAnnos)) {
      // If its already handled, skip it.
      if (pair.index() == index) {
        index = getNextHandledIndex();
        continue;
      }
      // If this is a breadcrumb, copy it over with no changes.
      auto anno = pair.value();
      if (anno.getClassAttr() == nonLocalString) {
        newAnnotations.push_back(anno);
        continue;
      }
      makeAnnotationNonLocal(toNLAs, toModule, toModule.moduleNameAttr(), to,
                             toModule, to, anno, newAnnotations);
    }

    // Copy over all the new annotations.
    if (!newAnnotations.empty())
      to.setAnnotations(AnnotationSet(newAnnotations, context));
  }

  /// Merge all annotations and port annotations on two operations.
  void mergeAnnotations(FModuleLike toModule, Operation *to,
                        FModuleLike fromModule, Operation *from) {
    // Merge op annotations.
    mergeAnnotations(toModule, OpAnnoTarget(to), AnnotationSet(to), fromModule,
                     OpAnnoTarget(to), AnnotationSet(from));

    // Merge port annotations.
    if (toModule == to) {
      // Merge module port annotations.
      for (unsigned i = 0, e = toModule.getNumPorts(); i < e; ++i)
        mergeAnnotations(toModule, PortAnnoTarget(toModule, i),
                         AnnotationSet::forPort(toModule, i), fromModule,
                         PortAnnoTarget(fromModule, i),
                         AnnotationSet::forPort(fromModule, i));
    } else if (auto toMem = dyn_cast<MemOp>(to)) {
      // Merge memory port annotations.
      auto fromMem = cast<MemOp>(from);
      for (unsigned i = 0, e = toMem.getNumResults(); i < e; ++i)
        mergeAnnotations(toModule, PortAnnoTarget(toMem, i),
                         AnnotationSet::forPort(toMem, i), fromModule,
                         PortAnnoTarget(fromMem, i),
                         AnnotationSet::forPort(fromMem, i));
    }
  }

  // Record the symbol name change of the operation or any of its ports when
  // merging two operations.  The renamed symbols are used to update the
  // target of any NLAs.  This will add symbols to the "to" operation if needed.
  void recordSymRenames(RenameMap &renameMap, FModuleLike toModule,
                        Operation *to, FModuleLike fromModule,
                        Operation *from) {
    // If the "from" operation has an inner_sym, we need to make sure the
    // "to" operation also has an `inner_sym` and then record the renaming.
    if (auto fromSym = from->getAttrOfType<StringAttr>("inner_sym")) {
      auto toSym = OpAnnoTarget(to).getInnerSym(getNamespace(toModule));
      renameMap[fromSym] = toSym;
    }

    // If there are no port symbols on the "from" operation, we are done here.
    auto fromPortSyms = from->getAttrOfType<ArrayAttr>("portSyms");
    if (!fromPortSyms || fromPortSyms.empty())
      return;
    // We have to map each "fromPort" to each "toPort".
    auto &moduleNamespace = getNamespace(toModule);
    auto portCount = fromPortSyms.size();
    auto portNames = to->getAttrOfType<ArrayAttr>("portNames");
    auto toPortSyms = to->getAttrOfType<ArrayAttr>("portSyms");

    // Create an array of new port symbols for the "to" operation, copy in the
    // old symbols if it has any, create an empty symbol array if it doesn't.
    SmallVector<Attribute> newPortSyms;
    auto emptyString = StringAttr::get(context, "");
    if (toPortSyms.empty())
      newPortSyms.assign(portCount, emptyString);
    else
      newPortSyms.assign(toPortSyms.begin(), toPortSyms.end());

    for (unsigned portNo = 0; portNo < portCount; ++portNo) {
      // If this fromPort doesn't have a symbol, move on to the next one.
      auto fromSym = fromPortSyms[portNo].cast<StringAttr>();
      if (fromSym.getValue().empty())
        continue;

      // If this toPort doesn't have a symbol, assign one.
      auto toSym = newPortSyms[portNo].cast<StringAttr>();
      if (toSym == emptyString) {
        // Get a reasonable base name for the port.
        StringRef symName = "inner_sym";
        if (portNames)
          symName = portNames[portNo].cast<StringAttr>().getValue();
        // Create the symbol and store it into the array.
        toSym = StringAttr::get(context, moduleNamespace.newName(symName));
        newPortSyms[portNo] = toSym;
      }

      // Record the renaming.
      renameMap[fromSym] = toSym;
    }

    // Commit the new symbol attribute.
    to->setAttr("portSyms", ArrayAttr::get(context, newPortSyms));
  }

  /// Recursively merge two operations.
  // NOLINTNEXTLINE(misc-no-recursion)
  void mergeOps(RenameMap &renameMap, FModuleLike toModule, Operation *to,
                FModuleLike fromModule, Operation *from) {
    // Merge the operation locations.
    to->setLoc(FusedLoc::get(context, {to->getLoc(), from->getLoc()}));

    // Recurse into any regions.
    for (auto regions : llvm::zip(to->getRegions(), from->getRegions()))
      mergeRegions(renameMap, toModule, std::get<0>(regions), fromModule,
                   std::get<1>(regions));

    // Record any inner_sym renamings that happened.
    if (to != from)
      recordSymRenames(renameMap, toModule, to, fromModule, from);

    // Merge the annotations.
    mergeAnnotations(toModule, to, fromModule, from);
  }

  /// Recursively merge two blocks.
  void mergeBlocks(RenameMap &renameMap, FModuleLike toModule, Block &toBlock,
                   FModuleLike fromModule, Block &fromBlock) {
    for (auto ops : llvm::zip(toBlock, fromBlock))
      mergeOps(renameMap, toModule, &std::get<0>(ops), fromModule,
               &std::get<1>(ops));
  }

  // Recursively merge two regions.
  void mergeRegions(RenameMap &renameMap, FModuleLike toModule,
                    Region &toRegion, FModuleLike fromModule,
                    Region &fromRegion) {
    for (auto blocks : llvm::zip(toRegion, fromRegion))
      mergeBlocks(renameMap, toModule, std::get<0>(blocks), fromModule,
                  std::get<1>(blocks));
  }

  MLIRContext *context;
  InstanceGraph &instanceGraph;
  SymbolTable &symbolTable;

  /// Cached nla table analysis.
  NLATable *nlaTable = nullptr;

  /// We insert all NLAs to the beginning of this block.
  Block *nlaBlock;

  // This maps an NLA to the operation or port that uses it. Since NLAs include
  // the name of the leaf element, its only possible for the NLA to be used by a
  // single op or port.
  DenseMap<Attribute, AnnoTarget> targetMap;

  // Cached attributes for faster comparisons and attribute building.
  StringAttr nonLocalString;
  StringAttr classString;

  /// A module namespace cache.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;
};

//===----------------------------------------------------------------------===//
// Fixup
//===----------------------------------------------------------------------===//

/// This fixes up connects when the field names of a bundle type changes.  It
/// finds all fields which were previously bulk connected and legalizes it
/// into a connect for each field.
template <typename T>
void fixupConnect(ImplicitLocOpBuilder &builder, Value dst, Value src) {
  // If the types already match we can emit a connect.
  auto dstType = dst.getType();
  auto srcType = src.getType();
  if (dstType == srcType) {
    builder.create<T>(dst, src);
    return;
  }
  // It must be a bundle type and the field name has changed. We have to
  // manually decompose the bulk connect into a connect for each field.
  auto dstBundle = dstType.cast<BundleType>();
  auto srcBundle = srcType.cast<BundleType>();
  for (unsigned i = 0; i < dstBundle.getNumElements(); ++i) {
    auto dstField = builder.create<SubfieldOp>(dst, i);
    auto srcField = builder.create<SubfieldOp>(src, i);
    if (dstBundle.getElement(i).isFlip) {
      std::swap(srcBundle, dstBundle);
      std::swap(srcField, dstField);
    }
    fixupConnect<T>(builder, dstField, srcField);
  }
}

/// Replaces a ConnectOp or StrictConnectOp with new bundle types.
template <typename T>
void fixupConnect(T connect) {
  ImplicitLocOpBuilder builder(connect.getLoc(), connect);
  fixupConnect<T>(builder, connect.dest(), connect.src());
  connect->erase();
}

/// When we replace a bundle type with a similar bundle with different field
/// names, we have to rewrite all the code to use the new field names. This
/// mostly affects subfield result types and any bulk connects.
void fixupReferences(Value oldValue, Type newType) {
  SmallVector<std::pair<Value, Type>> workList;
  workList.emplace_back(oldValue, newType);
  while (!workList.empty()) {
    auto [oldValue, newType] = workList.pop_back_val();
    auto oldType = oldValue.getType();
    // If the two types are identical, we don't need to do anything, otherwise
    // update the type in place.
    if (oldType == newType)
      continue;
    oldValue.setType(newType);
    for (auto *op : llvm::make_early_inc_range(oldValue.getUsers())) {
      if (auto subfield = dyn_cast<SubfieldOp>(op)) {
        // Rewrite a subfield op to return the correct type.
        auto index = subfield.fieldIndex();
        auto result = subfield.getResult();
        auto newResultType = newType.cast<BundleType>().getElementType(index);
        workList.emplace_back(result, newResultType);
        continue;
      }
      if (auto connect = dyn_cast<ConnectOp>(op)) {
        fixupConnect<ConnectOp>(connect);
        continue;
      }
      if (auto strict = dyn_cast<StrictConnectOp>(op)) {
        fixupConnect<StrictConnectOp>(strict);
        continue;
      }
    }
  }
}

/// This is the root method to fixup module references when a module changes.
/// It matches all the results of "to" module with the results of the "from"
/// module.
void fixupAllModules(InstanceGraph &instanceGraph) {
  for (auto *node : instanceGraph) {
    auto module = cast<FModuleLike>(*node->getModule());
    for (auto *instRec : node->uses()) {
      auto inst = instRec->getInstance();
      for (unsigned i = 0, e = module.getNumPorts(); i < e; ++i)
        fixupReferences(inst->getResult(i), module.getPortType(i));
    }
  }
}

/// A DenseMapInfo implementation for llvm::SHA256 hashes, which are represented
/// as std::array<uint8_t, 32>. This allows us to create DenseMaps with SHA256
/// hashes as keys.
struct SHA256HashDenseMapInfo {
  static inline std::array<uint8_t, 32> getEmptyKey() {
    std::array<uint8_t, 32> key;
    std::fill(key.begin(), key.end(), ~0);
    return key;
  }
  static inline std::array<uint8_t, 32> getTombstoneKey() {
    std::array<uint8_t, 32> key;
    std::fill(key.begin(), key.end(), ~0 - 1);
    return key;
  }

  static unsigned getHashValue(const std::array<uint8_t, 32> &val) {
    // We assume SHA256 is already a good hash and just truncate down to the
    // number of bytes we need for DenseMap.
    unsigned hash;
    std::memcpy(&hash, val.data(), sizeof(unsigned));
    return hash;
  }

  static bool isEqual(const std::array<uint8_t, 32> &lhs,
                      const std::array<uint8_t, 32> &rhs) {
    return lhs == rhs;
  }
};

//===----------------------------------------------------------------------===//
// DedupPass
//===----------------------------------------------------------------------===//

namespace {
class DedupPass : public DedupBase<DedupPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto circuit = getOperation();
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    auto *nlaTable = &getAnalysis<NLATable>();
    SymbolTable symbolTable(circuit);
    Deduper deduper(instanceGraph, symbolTable, nlaTable, circuit);
    StructuralHasher hasher(&getContext());
    Equivalence equiv(context, instanceGraph);
    auto anythingChanged = false;

    // Modules annotated with this should not be considered for deduplication.
    auto noDedupClass =
        StringAttr::get(context, "firrtl.transforms.NoDedupAnnotation");

    // A map of all the module hashes that we have calculated so far.
    llvm::DenseMap<std::array<uint8_t, 32>, Operation *, SHA256HashDenseMapInfo>
        moduleHashes;

    // We track the name of the module that each module is deduped into, so that
    // we can make sure all modules which are marked "must dedup" with each
    // other were all deduped to the same module.
    DenseMap<Attribute, StringAttr> dedupMap;

    // We must iterate the modules from the bottom up so that we can properly
    // deduplicate the modules. We have to store the visit order first so that
    // we can safely delete nodes as we go from the instance graph.
    for (auto *node : llvm::post_order(&instanceGraph)) {
      auto module = cast<FModuleLike>(*node->getModule());
      auto moduleName = module.moduleNameAttr();
      // If the module is marked with NoDedup, just skip it.
      if (AnnotationSet(module).hasAnnotation(noDedupClass)) {
        // We record it in the dedup map to help detect errors when the user
        // marks the module as both NoDedup and MustDedup. We do not record this
        // module in the hasher to make sure no other module dedups "into" this
        // one.
        dedupMap[moduleName] = moduleName;
        continue;
      }
      // Calculate the hash of the module.
      auto h = hasher.hash(module);
      // Check if there a module with the same hash.
      auto it = moduleHashes.find(h);
      if (it != moduleHashes.end()) {
        auto original = cast<FModuleLike>(it->second);
        // Record the group ID of the other module.
        dedupMap[moduleName] = original.moduleNameAttr();
        deduper.dedup(original, module);
        erasedModules++;
        anythingChanged = true;
        continue;
      }
      // Any module not deduplicated must be recorded.
      deduper.record(module);
      // Add the module to a new dedup group.
      dedupMap[moduleName] = moduleName;
      // Record the module's hash.
      moduleHashes[h] = module;
    }

    // This part verifies that all modules marked by "MustDedup" have been
    // properly deduped with each other. For this check to succeed, all modules
    // have to been deduped to the same module. It is possible that a module was
    // deduped with the wrong thing.

    auto failed = false;
    // This parses the module name out of a target string.
    auto parseModule = [&](Attribute path) -> StringAttr {
      // Each module is listed as a target "~Circuit|Module" which we have to
      // parse.
      auto [_, rhs] = path.cast<StringAttr>().getValue().split('|');
      return StringAttr::get(context, rhs);
    };
    // This gets the name of the module which the current module was deduped
    // with. If the named module isn't in the map, then we didn't encounter it
    // in the circuit.
    auto getLead = [&](StringAttr module) -> StringAttr {
      auto it = dedupMap.find(module);
      if (it == dedupMap.end()) {
        auto diag = emitError(circuit.getLoc(),
                              "MustDeduplicateAnnotation references module ")
                    << module << " which does not exist";
        failed = true;
        return 0;
      }
      return it->second;
    };

    AnnotationSet::removeAnnotations(circuit, [&](Annotation annotation) {
      // If we have already failed, don't process any more annotations.
      if (failed)
        return false;
      if (!annotation.isClass("firrtl.transforms.MustDeduplicateAnnotation"))
        return false;
      auto modules = annotation.getMember<ArrayAttr>("modules");
      if (!modules) {
        emitError(circuit.getLoc(),
                  "MustDeduplicateAnnotation missing \"modules\" member");
        failed = true;
        return false;
      }
      // Empty module list has nothing to process.
      if (modules.size() == 0)
        return true;
      // Get the first element.
      auto firstModule = parseModule(modules[0]);
      auto firstLead = getLead(firstModule);
      if (failed)
        return false;
      // Verify that the remaining elements are all the same as the first.
      for (auto attr : modules.getValue().drop_front()) {
        auto nextModule = parseModule(attr);
        auto nextLead = getLead(nextModule);
        if (failed)
          return false;
        if (firstLead != nextLead) {
          auto diag = emitError(circuit.getLoc(), "module ")
                      << nextModule << " not deduplicated with " << firstModule;
          auto a = instanceGraph.lookup(firstLead)->getModule();
          auto b = instanceGraph.lookup(nextLead)->getModule();
          equiv.check(diag, a, b);
          failed = true;
          return false;
        }
      }
      return true;
    });
    if (failed)
      return signalPassFailure();

    // Walk all the modules and fixup the instance operation to return the
    // correct type. We delay this fixup until the end because doing it early
    // can block the deduplication of the parent modules.
    fixupAllModules(instanceGraph);

    if (!anythingChanged)
      markAllAnalysesPreserved();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createDedupPass() {
  return std::make_unique<DedupPass>();
}
