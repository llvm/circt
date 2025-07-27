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

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SHA256.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_DEDUP
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using hw::InnerRefAttr;

//===----------------------------------------------------------------------===//
// Utility function for classifying a Symbol's dedup-ability.
//===----------------------------------------------------------------------===//

/// Returns true if the module can be removed.
static bool canRemoveModule(mlir::SymbolOpInterface symbol) {
  // If the symbol is not private, it cannot be removed.
  if (!symbol.isPrivate())
    return false;
  // Classes may be referenced in object types, so can not normally be removed
  // if we can't find any symbol uses. Since we know that dedup will update the
  // types of instances appropriately, we can ignore that and return true here.
  if (isa<ClassLike>(*symbol))
    return true;
  // If module can not be removed even if no uses can be found, we can not
  // delete it. The implication is that there are hidden symbol uses that dedup
  // will not properly update.
  if (!symbol.canDiscardOnUseEmpty())
    return false;
  // The module can be deleted.
  return true;
}

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

llvm::raw_ostream &printHex(llvm::raw_ostream &stream,
                            ArrayRef<uint8_t> bytes) {
  // Print the hash on a single line.
  return stream << format_bytes(bytes, std::nullopt, 32) << "\n";
}

llvm::raw_ostream &printHash(llvm::raw_ostream &stream, llvm::SHA256 &data) {
  return printHex(stream, data.result());
}

llvm::raw_ostream &printHash(llvm::raw_ostream &stream, std::string data) {
  ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(data.c_str()),
                          data.length());
  return printHex(stream, bytes);
}

// This struct contains information to determine module module uniqueness. A
// first element is a structural hash of the module, and the second element is
// an array which tracks module names encountered in the walk. Since module
// names could be replaced during dedup, it's necessary to keep names up-to-date
// before actually combining them into structural hashes.
struct ModuleInfo {
  // SHA256 hash.
  std::array<uint8_t, 32> structuralHash;
  // Module names referred by instance op in the module.
  std::vector<StringAttr> referredModuleNames;
};

static bool operator==(const ModuleInfo &lhs, const ModuleInfo &rhs) {
  return lhs.structuralHash == rhs.structuralHash &&
         lhs.referredModuleNames == rhs.referredModuleNames;
}

/// This struct contains constant string attributes shared across different
/// threads.
struct StructuralHasherSharedConstants {
  explicit StructuralHasherSharedConstants(MLIRContext *context) {
    portTypesAttr = StringAttr::get(context, "portTypes");
    moduleNameAttr = StringAttr::get(context, "moduleName");
    portNamesAttr = StringAttr::get(context, "portNames");
    nonessentialAttributes.insert(StringAttr::get(context, "annotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "convention"));
    nonessentialAttributes.insert(StringAttr::get(context, "inner_sym"));
    nonessentialAttributes.insert(StringAttr::get(context, "name"));
    nonessentialAttributes.insert(StringAttr::get(context, "portAnnotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "portLocations"));
    nonessentialAttributes.insert(StringAttr::get(context, "portNames"));
    nonessentialAttributes.insert(StringAttr::get(context, "portSymbols"));
    nonessentialAttributes.insert(StringAttr::get(context, "sym_name"));
    nonessentialAttributes.insert(StringAttr::get(context, "sym_visibility"));
  };

  // This is a cached "portTypes" string attr.
  StringAttr portTypesAttr;

  // This is a cached "moduleName" string attr.
  StringAttr moduleNameAttr;

  // This is a cached "portNames" string attr.
  StringAttr portNamesAttr;

  // This is a set of every attribute we should ignore.
  DenseSet<Attribute> nonessentialAttributes;
};

struct StructuralHasher {
  explicit StructuralHasher(const StructuralHasherSharedConstants &constants)
      : constants(constants) {}

  ModuleInfo getModuleInfo(FModuleLike module) {
    populateInnerSymIDTable(module);
    update(&(*module));
    return {sha.final(), std::move(referredModuleNames)};
  }

private:
  /// Find all the ports and operations which may define an inner symbol
  /// operations and give each a unique id.  If the port/operation does define
  /// an inner symbol, map the symbol name to a pair of the id and the symbol's
  /// field id. When we hash (local) references to this inner symbol, we will
  /// hash in the id and the the field id.
  void populateInnerSymIDTable(FModuleLike module) {
    // Add port symbols. If no port has a symbol defined, the port symbol array
    // will be totally empty.
    for (auto [index, innerSym] : llvm::enumerate(module.getPortSymbols())) {
      for (auto prop : cast<hw::InnerSymAttr>(innerSym))
        innerSymIDTable[prop.getName()] = std::pair(index, prop.getFieldID());
    }
    // Add operation symbols.
    size_t index = module.getNumPorts();
    module.walk([&](hw::InnerSymbolOpInterface innerSymOp) {
      if (auto innerSym = innerSymOp.getInnerSymAttr()) {
        for (auto prop : innerSym)
          innerSymIDTable[prop.getName()] = std::pair(index, prop.getFieldID());
      }
      ++index;
    });
  }

  // Get the identifier for an object. The identifier is assigned on first use.
  unsigned getID(void *object) {
    auto [it, inserted] = idTable.try_emplace(object, nextID);
    if (inserted)
      ++nextID;
    return it->second;
  }

  // Get the identifier for an IR object. Free the ID, too.
  unsigned finalizeID(void *object) {
    auto it = idTable.find(object);
    if (it == idTable.end())
      return nextID++;
    auto id = it->second;
    idTable.erase(it);
    return id;
  }

  std::pair<size_t, size_t> getInnerSymID(StringAttr name) {
    return innerSymIDTable.at(name);
  }

  void update(OpOperand &operand) {
    auto value = operand.get();
    if (auto result = dyn_cast<OpResult>(value)) {
      auto *op = result.getOwner();
      update(getID(op));
      update(result.getResultNumber());
      return;
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto *block = argument.getOwner();
      update(getID(block));
      update(argument.getArgNumber());
      return;
    }
    llvm_unreachable("Unknown value type");
  }

  void update(const void *pointer) {
    auto *addr = reinterpret_cast<const uint8_t *>(&pointer);
    sha.update(ArrayRef<uint8_t>(addr, sizeof pointer));
  }

  void update(size_t value) {
    auto *addr = reinterpret_cast<const uint8_t *>(&value);
    sha.update(ArrayRef<uint8_t>(addr, sizeof value));
  }

  template <typename T, typename U>
  void update(const std::pair<T, U> &pair) {
    update(pair.first);
    update(pair.second);
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
    if (auto bundle = type_dyn_cast<BundleType>(type))
      return update(bundle);
    update(type.getAsOpaquePointer());
  }

  void update(OpResult result) {
    // Like instance ops, don't use object ops' result types since they might be
    // replaced by dedup. Record the class names and lazily combine their hashes
    // using the same mechanism as instances and modules.
    if (auto objectOp = dyn_cast<ObjectOp>(result.getOwner())) {
      referredModuleNames.push_back(objectOp.getType().getNameAttr().getAttr());
      return;
    }

    update(result.getType());
  }

  /// Hash the top level attribute dictionary of the operation.  This function
  /// has special handling for inner symbols, ports, and referenced modules.
  void update(Operation *op, DictionaryAttr dict) {
    for (auto namedAttr : dict) {
      auto name = namedAttr.getName();
      auto value = namedAttr.getValue();
      // Skip names and annotations, except in certain cases.
      // Names of ports are load bearing for classes, so we do hash those.
      bool isClassPortNames =
          isa<ClassLike>(op) && name == constants.portNamesAttr;
      if (constants.nonessentialAttributes.contains(name) && !isClassPortNames)
        continue;

      // Hash the attribute name (an interned pointer).
      update(name.getAsOpaquePointer());

      // Hash the port types.
      if (name == constants.portTypesAttr) {
        auto portTypes = cast<ArrayAttr>(value).getAsValueRange<TypeAttr>();
        for (auto type : portTypes)
          update(type);
        continue;
      }

      // For instance op, don't use `moduleName` attributes since they might be
      // replaced by dedup. Record the names and lazily combine their hashes.
      // It is assumed that module names are hashed only through instance ops;
      // it could cause suboptimal results if there was other operation that
      // refers to module names through essential attributes.
      if (isa<InstanceOp>(op) && name == constants.moduleNameAttr) {
        referredModuleNames.push_back(cast<FlatSymbolRefAttr>(value).getAttr());
        continue;
      }

      // TODO: properly handle DistinctAttr, including its use in paths.
      // See https://github.com/llvm/circt/issues/6583.
      if (isa<DistinctAttr>(value))
        continue;

      // If this is an symbol reference, we need to perform name erasure.
      if (auto innerRef = dyn_cast<hw::InnerRefAttr>(value)) {
        update(getInnerSymID(innerRef.getName()));
        continue;
      }

      // We don't need to handle this attribute specially, so hash its unique
      // address.
      update(value.getAsOpaquePointer());
    }
  }

  void update(mlir::OperationName name) {
    // Operation names are interned.
    update(name.getAsOpaquePointer());
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void update(Block *block) {
    for (auto &op : llvm::reverse(*block))
      update(&op);
    for (auto type : block->getArgumentTypes())
      update(type);
    update(finalizeID(block));
    update(position);
    ++position;
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void update(Region *region) {
    for (auto &block : llvm::reverse(region->getBlocks()))
      update(&block);
    update(position);
    ++position;
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void update(Operation *op) {
    // Hash the regions. We need to make sure an empty region doesn't hash the
    // same as no region, so we include the number of regions.
    update(op->getNumRegions());
    for (auto &region : reverse(op->getRegions()))
      update(&region);

    update(op->getName());

    // Record the uses for later hashing.
    for (auto &operand : op->getOpOperands())
      update(operand);

    // This happens after the numbering above, as it uses blockarg numbering
    // for inner symbols.
    update(op, op->getAttrDictionary());

    // Record any op results (types).
    for (auto result : op->getResults())
      update(result);

    // Incorporate the hash of uses we have already built.
    update(finalizeID(op));
    update(position);
    ++position;
  }

  // A map from an operation/block, to its identifier.
  DenseMap<void *, unsigned> idTable;
  unsigned nextID = 0;

  // A map from an inner symbol, to its identifier.
  DenseMap<StringAttr, std::pair<size_t, size_t>> innerSymIDTable;

  // This keeps track of module names in the order of the appearance.
  std::vector<StringAttr> referredModuleNames;

  // String constants.
  const StructuralHasherSharedConstants &constants;

  // This is the actual running hash calculation. This is a stateful element
  // that should be reinitialized after each hash is produced.
  llvm::SHA256 sha;

  // The index of the current op. Increment after handling each op.
  size_t position = 0;
};

//===----------------------------------------------------------------------===//
// Equivalence
//===----------------------------------------------------------------------===//

/// This class is for reporting differences between two modules which should
/// have been deduplicated.
struct Equivalence {
  Equivalence(MLIRContext *context, InstanceGraph &instanceGraph)
      : instanceGraph(instanceGraph) {
    noDedupClass = StringAttr::get(context, noDedupAnnoClass);
    dedupGroupAttrName = StringAttr::get(context, "firrtl.dedup_group");
    portDirectionsAttr = StringAttr::get(context, "portDirections");
    nonessentialAttributes.insert(StringAttr::get(context, "annotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "name"));
    nonessentialAttributes.insert(StringAttr::get(context, "portAnnotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "portNames"));
    nonessentialAttributes.insert(StringAttr::get(context, "portTypes"));
    nonessentialAttributes.insert(StringAttr::get(context, "portSymbols"));
    nonessentialAttributes.insert(StringAttr::get(context, "portLocations"));
    nonessentialAttributes.insert(StringAttr::get(context, "sym_name"));
    nonessentialAttributes.insert(StringAttr::get(context, "inner_sym"));
  }

  struct ModuleData {
    ModuleData(const hw::InnerSymbolTable &a, const hw::InnerSymbolTable &b)
        : a(a), b(b) {}
    IRMapping map;
    const hw::InnerSymbolTable &a;
    const hw::InnerSymbolTable &b;
  };

  std::string prettyPrint(Attribute attr) {
    SmallString<64> buffer;
    llvm::raw_svector_ostream os(buffer);
    if (auto integerAttr = dyn_cast<IntegerAttr>(attr)) {
      os << "0x";
      if (integerAttr.getType().isSignlessInteger())
        integerAttr.getValue().toStringUnsigned(buffer, /*radix=*/16);
      else
        integerAttr.getAPSInt().toString(buffer, /*radix=*/16);

    } else
      os << attr;
    return std::string(buffer);
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
    if (auto aBundleType = type_dyn_cast<BundleType>(aType))
      if (auto bBundleType = type_dyn_cast<BundleType>(bType))
        return check(diag, message, a, aBundleType, b, bBundleType);
    if (type_isa<RefType>(aType) && type_isa<RefType>(bType) &&
        aType != bType) {
      diag.attachNote(a->getLoc())
          << message << ", has a RefType with a different base type "
          << type_cast<RefType>(aType).getType()
          << " in the same position of the two modules marked as 'must dedup'. "
             "(This may be due to Grand Central Taps or Views being different "
             "between the two modules.)";
      diag.attachNote(b->getLoc())
          << "the second module has a different base type "
          << type_cast<RefType>(bType).getType();
      return failure();
    }
    diag.attachNote(a->getLoc())
        << message << " types don't match, first type is " << aType;
    diag.attachNote(b->getLoc()) << "second type is " << bType;
    return failure();
  }

  LogicalResult check(InFlightDiagnostic &diag, ModuleData &data, Operation *a,
                      Block &aBlock, Operation *b, Block &bBlock) {

    // Block argument types.
    auto portNames = a->getAttrOfType<ArrayAttr>("portNames");
    auto portNo = 0;
    auto emitMissingPort = [&](Value existsVal, Operation *opExists,
                               Operation *opDoesNotExist) {
      StringRef portName;
      auto portNames = opExists->getAttrOfType<ArrayAttr>("portNames");
      if (portNames)
        if (auto portNameAttr = dyn_cast<StringAttr>(portNames[portNo]))
          portName = portNameAttr.getValue();
      if (type_isa<RefType>(existsVal.getType())) {
        diag.attachNote(opExists->getLoc())
            << " contains a RefType port named '" + portName +
                   "' that only exists in one of the modules (can be due to "
                   "difference in Grand Central Tap or View of two modules "
                   "marked with must dedup)";
        diag.attachNote(opDoesNotExist->getLoc())
            << "second module to be deduped that does not have the RefType "
               "port";
      } else {
        diag.attachNote(opExists->getLoc())
            << "port '" + portName + "' only exists in one of the modules";
        diag.attachNote(opDoesNotExist->getLoc())
            << "second module to be deduped that does not have the port";
      }
      return failure();
    };

    for (auto argPair :
         llvm::zip_longest(aBlock.getArguments(), bBlock.getArguments())) {
      auto &aArg = std::get<0>(argPair);
      auto &bArg = std::get<1>(argPair);
      if (aArg.has_value() && bArg.has_value()) {
        // TODO: we should print the port number if there are no port names, but
        // there are always port names ;).
        StringRef portName;
        if (portNames) {
          if (auto portNameAttr = dyn_cast<StringAttr>(portNames[portNo]))
            portName = portNameAttr.getValue();
        }
        // Assumption here that block arguments correspond to ports.
        if (failed(check(diag, "module port '" + portName + "'", a,
                         aArg->getType(), b, bArg->getType())))
          return failure();
        data.map.map(aArg.value(), bArg.value());
        portNo++;
        continue;
      }
      if (!aArg.has_value())
        std::swap(a, b);
      return emitMissingPort(aArg.has_value() ? aArg.value() : bArg.value(), a,
                             b);
    }

    // Blocks operations.
    auto aIt = aBlock.begin();
    auto aEnd = aBlock.end();
    auto bIt = bBlock.begin();
    auto bEnd = bBlock.end();
    while (aIt != aEnd && bIt != bEnd)
      if (failed(check(diag, data, &*aIt++, &*bIt++)))
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

  LogicalResult check(InFlightDiagnostic &diag, ModuleData &data, Operation *a,
                      Region &aRegion, Operation *b, Region &bRegion) {
    auto aIt = aRegion.begin();
    auto aEnd = aRegion.end();
    auto bIt = bRegion.begin();
    auto bEnd = bRegion.end();

    // Region blocks.
    while (aIt != aEnd && bIt != bEnd)
      if (failed(check(diag, data, a, *aIt++, b, *bIt++)))
        return failure();
    if (aIt != aEnd || bIt != bEnd) {
      diag.attachNote(a->getLoc())
          << "operation regions have different number of blocks";
      diag.attachNote(b->getLoc()) << "second operation here";
      return failure();
    }
    return success();
  }

  LogicalResult check(InFlightDiagnostic &diag, Operation *a,
                      mlir::DenseBoolArrayAttr aAttr, Operation *b,
                      mlir::DenseBoolArrayAttr bAttr) {
    if (aAttr == bAttr)
      return success();
    auto portNames = a->getAttrOfType<ArrayAttr>("portNames");
    for (unsigned i = 0, e = aAttr.size(); i < e; ++i) {
      auto aDirection = aAttr[i];
      auto bDirection = bAttr[i];
      if (aDirection != bDirection) {
        auto &note = diag.attachNote(a->getLoc()) << "module port ";
        if (portNames)
          note << "'" << cast<StringAttr>(portNames[i]).getValue() << "'";
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

  LogicalResult check(InFlightDiagnostic &diag, ModuleData &data, Operation *a,
                      DictionaryAttr aDict, Operation *b,
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

      if (isa<hw::InnerRefAttr>(aAttr) && isa<hw::InnerRefAttr>(bAttr)) {
        auto bRef = cast<hw::InnerRefAttr>(bAttr);
        auto aRef = cast<hw::InnerRefAttr>(aAttr);
        // See if they are pointing at the same operation or port.
        auto aTarget = data.a.lookup(aRef.getName());
        auto bTarget = data.b.lookup(bRef.getName());
        if (!aTarget || !bTarget)
          diag.attachNote(a->getLoc())
              << "malformed ir, possibly violating use-before-def";
        auto error = [&]() {
          diag.attachNote(a->getLoc())
              << "operations have different targets, first operation has "
              << aTarget;
          diag.attachNote(b->getLoc()) << "second operation has " << bTarget;
          return failure();
        };
        if (aTarget.isPort()) {
          // If they are targeting ports, make sure its the same port number.
          if (!bTarget.isPort() || aTarget.getPort() != bTarget.getPort())
            return error();
        } else {
          // Otherwise make sure that they are targeting the same operation.
          if (!bTarget.isOpOnly() ||
              aTarget.getOp() != data.map.lookup(bTarget.getOp()))
            return error();
        }
        if (aTarget.getField() != bTarget.getField())
          return error();
      } else if (attrName == portDirectionsAttr) {
        // Special handling for the port directions attribute for better
        // error messages.
        if (failed(check(diag, a, cast<mlir::DenseBoolArrayAttr>(aAttr), b,
                         cast<mlir::DenseBoolArrayAttr>(bAttr))))
          return failure();
      } else if (isa<DistinctAttr>(aAttr) && isa<DistinctAttr>(bAttr)) {
        // TODO: properly handle DistinctAttr, including its use in paths.
        // See https://github.com/llvm/circt/issues/6583
      } else if (aAttr != bAttr) {
        diag.attachNote(a->getLoc())
            << "first operation has attribute '" << attrName.getValue()
            << "' with value " << prettyPrint(aAttr);
        diag.attachNote(b->getLoc())
            << "second operation has value " << prettyPrint(bAttr);
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
  LogicalResult check(InFlightDiagnostic &diag, FInstanceLike a,
                      FInstanceLike b) {
    auto aName = a.getReferencedModuleNameAttr();
    auto bName = b.getReferencedModuleNameAttr();
    if (aName == bName)
      return success();

    // If the modules instantiate are different we will want to know why the
    // sub module did not dedupliate. This code recursively checks the child
    // module.
    auto aModule = instanceGraph.lookup(aName)->getModule();
    auto bModule = instanceGraph.lookup(bName)->getModule();
    // Create a new error for the submodule.
    diag.attachNote(std::nullopt)
        << "in instance " << a.getInstanceNameAttr() << " of " << aName
        << ", and instance " << b.getInstanceNameAttr() << " of " << bName;
    check(diag, aModule, bModule);
    return failure();
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult check(InFlightDiagnostic &diag, ModuleData &data, Operation *a,
                      Operation *b) {
    // Operation name.
    if (a->getName() != b->getName()) {
      diag.attachNote(a->getLoc()) << "first operation is a " << a->getName();
      diag.attachNote(b->getLoc()) << "second operation is a " << b->getName();
      return failure();
    }

    // If its an instance operaiton, perform some checking and possibly
    // recurse.
    if (auto aInst = dyn_cast<FInstanceLike>(a)) {
      auto bInst = cast<FInstanceLike>(b);
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
      data.map.map(aValue, bValue);
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
      if (bValue != data.map.lookup(aValue)) {
        diag.attachNote(a->getLoc())
            << "operations use different operands, first operand is '"
            << getFieldName(
                   getFieldRefFromValue(aValue, /*lookThroughCasts=*/true))
                   .first
            << "'";
        diag.attachNote(b->getLoc())
            << "second operand is '"
            << getFieldName(
                   getFieldRefFromValue(bValue, /*lookThroughCasts=*/true))
                   .first
            << "', but should have been '"
            << getFieldName(getFieldRefFromValue(data.map.lookup(aValue),
                                                 /*lookThroughCasts=*/true))
                   .first
            << "'";
        return failure();
      }
    }
    data.map.map(a, b);

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
      if (failed(check(diag, data, a, aRegion, b, bRegion)))
        return failure();
    }

    // Operation attributes.
    if (failed(check(diag, data, a, a->getAttrDictionary(), b,
                     b->getAttrDictionary())))
      return failure();
    return success();
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void check(InFlightDiagnostic &diag, Operation *a, Operation *b) {
    hw::InnerSymbolTable aTable(a);
    hw::InnerSymbolTable bTable(b);
    ModuleData data(aTable, bTable);
    if (AnnotationSet::hasAnnotation(a, noDedupClass)) {
      diag.attachNote(a->getLoc()) << "module marked NoDedup";
      return;
    }
    if (AnnotationSet::hasAnnotation(b, noDedupClass)) {
      diag.attachNote(b->getLoc()) << "module marked NoDedup";
      return;
    }
    auto aSymbol = cast<mlir::SymbolOpInterface>(a);
    auto bSymbol = cast<mlir::SymbolOpInterface>(b);
    if (!canRemoveModule(aSymbol) && !canRemoveModule(bSymbol)) {
      diag.attachNote(a->getLoc())
          << "module is "
          << (aSymbol.isPrivate() ? "private but not discardable" : "public");
      diag.attachNote(b->getLoc())
          << "module is "
          << (bSymbol.isPrivate() ? "private but not discardable" : "public");
      return;
    }
    auto aGroup =
        dyn_cast_or_null<StringAttr>(a->getDiscardableAttr(dedupGroupAttrName));
    auto bGroup = dyn_cast_or_null<StringAttr>(
        b->getAttrOfType<StringAttr>(dedupGroupAttrName));
    if (aGroup != bGroup) {
      if (bGroup) {
        diag.attachNote(b->getLoc())
            << "module is in dedup group '" << bGroup.str() << "'";
      } else {
        diag.attachNote(b->getLoc()) << "module is not part of a dedup group";
      }
      if (aGroup) {
        diag.attachNote(a->getLoc())
            << "module is in dedup group '" << aGroup.str() << "'";
      } else {
        diag.attachNote(a->getLoc()) << "module is not part of a dedup group";
      }
      return;
    }
    if (failed(check(diag, data, a, b)))
      return;
    diag.attachNote(a->getLoc()) << "first module here";
    diag.attachNote(b->getLoc()) << "second module here";
  }

  // This is a cached "portDirections" string attr.
  StringAttr portDirectionsAttr;
  // This is a cached "NoDedup" annotation class string attr.
  StringAttr noDedupClass;
  // This is a cached string attr for the dedup group attribute.
  StringAttr dedupGroupAttrName;

  // This is a set of every attribute we should ignore.
  DenseSet<Attribute> nonessentialAttributes;
  InstanceGraph &instanceGraph;
};

//===----------------------------------------------------------------------===//
// Deduplication
//===----------------------------------------------------------------------===//

// Custom location merging.  This only keeps track of 8 annotations from ".fir"
// files, and however many annotations come from "real" sources.  When
// deduplicating, modules tend not to have scala source locators, so we wind
// up fusing source locators for a module from every copy being deduped.  There
// is little value in this (all the modules are identical by definition).
static Location mergeLoc(MLIRContext *context, Location to, Location from) {
  // Unique the set of locations to be fused.
  llvm::SmallSetVector<Location, 4> decomposedLocs;
  // only track 8 "fir" locations
  unsigned seenFIR = 0;
  for (auto loc : {to, from}) {
    // If the location is a fused location we decompose it if it has no
    // metadata or the metadata is the same as the top level metadata.
    if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
      // UnknownLoc's have already been removed from FusedLocs so we can
      // simply add all of the internal locations.
      for (auto loc : fusedLoc.getLocations()) {
        if (FileLineColLoc fileLoc = dyn_cast<FileLineColLoc>(loc)) {
          if (fileLoc.getFilename().strref().ends_with(".fir")) {
            ++seenFIR;
            if (seenFIR > 8)
              continue;
          }
        }
        decomposedLocs.insert(loc);
      }
      continue;
    }

    // Might need to skip this fir.
    if (FileLineColLoc fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      if (fileLoc.getFilename().strref().ends_with(".fir")) {
        ++seenFIR;
        if (seenFIR > 8)
          continue;
      }
    }
    // Otherwise, only add known locations to the set.
    if (!isa<UnknownLoc>(loc))
      decomposedLocs.insert(loc);
  }

  auto locs = decomposedLocs.getArrayRef();

  // Handle the simple cases of less than two locations. Ensure the metadata (if
  // provided) is not dropped.
  if (locs.empty())
    return UnknownLoc::get(context);
  if (locs.size() == 1)
    return locs.front();

  return FusedLoc::get(context, locs);
}

struct Deduper {

  using RenameMap = DenseMap<StringAttr, StringAttr>;

  Deduper(InstanceGraph &instanceGraph, SymbolTable &symbolTable,
          NLATable *nlaTable, CircuitOp circuit)
      : context(circuit->getContext()), instanceGraph(instanceGraph),
        symbolTable(symbolTable), nlaTable(nlaTable),
        nlaBlock(circuit.getBodyBlock()),
        nonLocalString(StringAttr::get(context, "circt.nonlocal")),
        classString(StringAttr::get(context, "class")) {
    // Populate the NLA cache.
    for (auto nla : circuit.getOps<hw::HierPathOp>())
      nlaCache[nla.getNamepathAttr()] = nla.getSymNameAttr();
  }

  /// Remove the "fromModule", and replace all references to it with the
  /// "toModule".  Modules should be deduplicated in a bottom-up order.  Any
  /// module which is not deduplicated needs to be recorded with the `record`
  /// call.
  void dedup(FModuleLike toModule, FModuleLike fromModule) {
    // A map of operation (e.g. wires, nodes) names which are changed, which is
    // used to update NLAs that reference the "fromModule".
    RenameMap renameMap;

    // Merge the port locations.
    SmallVector<Attribute> newLocs;
    for (auto [toLoc, fromLoc] : llvm::zip(toModule.getPortLocations(),
                                           fromModule.getPortLocations())) {
      if (toLoc == fromLoc)
        newLocs.push_back(toLoc);
      else
        newLocs.push_back(mergeLoc(context, cast<LocationAttr>(toLoc),
                                   cast<LocationAttr>(fromLoc)));
    }
    toModule->setAttr("portLocations", ArrayAttr::get(context, newLocs));

    // Merge the two modules.
    mergeOps(renameMap, toModule, toModule, fromModule, fromModule);

    // Rewrite NLAs pathing through these modules to refer to the to module. It
    // is safe to do this at this point because NLAs cannot be one element long.
    // This means that all NLAs which require more context cannot be targetting
    // something in the module it self.
    if (auto to = dyn_cast<FModuleOp>(*toModule))
      rewriteModuleNLAs(renameMap, to, cast<FModuleOp>(*fromModule));
    else
      rewriteExtModuleNLAs(renameMap, toModule.getModuleNameAttr(),
                           fromModule.getModuleNameAttr());

    replaceInstances(toModule, fromModule);
  }

  /// Record the usages of any NLA's in this module, so that we may update the
  /// annotation if the parent module is deduped with another module.
  void record(FModuleLike module) {
    // Record any annotations on the module.
    recordAnnotations(module);
    // Record port annotations.
    for (unsigned i = 0, e = getNumPorts(module); i < e; ++i)
      recordAnnotations(PortAnnoTarget(module, i));
    // Record any annotations in the module body.
    module->walk([&](Operation *op) { recordAnnotations(op); });
  }

private:
  /// Get a cached namespace for a module.
  hw::InnerSymbolNamespace &getNamespace(Operation *module) {
    return moduleNamespaces.try_emplace(module, cast<FModuleLike>(module))
        .first->second;
  }

  /// For a specific annotation target, record all the unique NLAs which
  /// target it in the `targetMap`.
  void recordAnnotations(AnnoTarget target) {
    for (auto anno : target.getAnnotations())
      if (auto nlaRef = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
        targetMap[nlaRef.getAttr()].insert(target);
  }

  /// Record all targets which use an NLA.
  void recordAnnotations(Operation *op) {
    // Record annotations.
    recordAnnotations(OpAnnoTarget(op));

    // Record port annotations only if this is a mem operation.
    auto mem = dyn_cast<MemOp>(op);
    if (!mem)
      return;

    // Record port annotations.
    for (unsigned i = 0, e = mem->getNumResults(); i < e; ++i)
      recordAnnotations(PortAnnoTarget(mem, i));
  }

  /// This deletes and replaces all instances of the "fromModule" with instances
  /// of the "toModule".
  void replaceInstances(FModuleLike toModule, Operation *fromModule) {
    // Replace all instances of the other module.
    auto *fromNode =
        instanceGraph[::cast<igraph::ModuleOpInterface>(fromModule)];
    auto *toNode = instanceGraph[toModule];
    auto toModuleRef = FlatSymbolRefAttr::get(toModule.getModuleNameAttr());
    for (auto *oldInstRec : llvm::make_early_inc_range(fromNode->uses())) {
      auto inst = oldInstRec->getInstance();
      if (auto instOp = dyn_cast<InstanceOp>(*inst)) {
        instOp.setModuleNameAttr(toModuleRef);
        instOp.setPortNamesAttr(toModule.getPortNamesAttr());
      } else if (auto objectOp = dyn_cast<ObjectOp>(*inst)) {
        auto classLike = cast<ClassLike>(*toNode->getModule());
        ClassType classType = detail::getInstanceTypeForClassLike(classLike);
        objectOp.getResult().setType(classType);
      }
      oldInstRec->getParent()->addInstance(inst, toNode);
      oldInstRec->erase();
    }
    instanceGraph.erase(fromNode);
    fromModule->erase();
  }

  /// Look up the instantiations of the `from` module and create an NLA for each
  /// one, appending the baseNamepath to each NLA. This is used to add more
  /// context to an already existing NLA. The `fromModule` is used to indicate
  /// which module the annotation is coming from before the merge, and will be
  /// used to create the namepaths.
  SmallVector<FlatSymbolRefAttr>
  createNLAs(Operation *fromModule, ArrayRef<Attribute> baseNamepath,
             SymbolTable::Visibility vis = SymbolTable::Visibility::Private) {
    // Create an attribute array with a placeholder in the first element, where
    // the root refence of the NLA will be inserted.
    SmallVector<Attribute> namepath = {nullptr};
    namepath.append(baseNamepath.begin(), baseNamepath.end());

    auto loc = fromModule->getLoc();
    auto *fromNode = instanceGraph[cast<igraph::ModuleOpInterface>(fromModule)];
    SmallVector<FlatSymbolRefAttr> nlas;
    for (auto *instanceRecord : fromNode->uses()) {
      auto parent = cast<FModuleOp>(*instanceRecord->getParent()->getModule());
      auto inst = instanceRecord->getInstance();
      namepath[0] = OpAnnoTarget(inst).getNLAReference(getNamespace(parent));
      auto arrayAttr = ArrayAttr::get(context, namepath);
      // Check the NLA cache to see if we already have this NLA.
      auto &cacheEntry = nlaCache[arrayAttr];
      if (!cacheEntry) {
        auto builder = OpBuilder::atBlockBegin(nlaBlock);
        auto nla = hw::HierPathOp::create(builder, loc, "nla", arrayAttr);
        // Insert it into the symbol table to get a unique name.
        symbolTable.insert(nla);
        // Store it in the cache.
        cacheEntry = nla.getNameAttr();
        nla.setVisibility(vis);
        nlaTable->addNLA(nla);
      }
      auto nlaRef = FlatSymbolRefAttr::get(cast<StringAttr>(cacheEntry));
      nlas.push_back(nlaRef);
    }
    return nlas;
  }

  /// Look up the instantiations of this module and create an NLA for each one.
  /// This returns an array of symbol references which can be used to reference
  /// the NLAs.
  SmallVector<FlatSymbolRefAttr>
  createNLAs(StringAttr toModuleName, FModuleLike fromModule,
             SymbolTable::Visibility vis = SymbolTable::Visibility::Private) {
    return createNLAs(fromModule, FlatSymbolRefAttr::get(toModuleName), vis);
  }

  /// Clone the annotation for each NLA in a list. The attribute list should
  /// have a placeholder for the "circt.nonlocal" field, and `nonLocalIndex`
  /// should be the index of this field.
  void cloneAnnotation(SmallVectorImpl<FlatSymbolRefAttr> &nlas,
                       Annotation anno, ArrayRef<NamedAttribute> attributes,
                       unsigned nonLocalIndex,
                       SmallVectorImpl<Annotation> &newAnnotations) {
    SmallVector<NamedAttribute> mutableAttributes(attributes.begin(),
                                                  attributes.end());
    for (auto &nla : nlas) {
      // Add the new annotation.
      mutableAttributes[nonLocalIndex].setValue(nla);
      auto dict = DictionaryAttr::getWithSorted(context, mutableAttributes);
      // The original annotation records if its a subannotation.
      anno.setDict(dict);
      newAnnotations.push_back(anno);
    }
  }

  /// This erases the NLA op, and removes the NLA from every module's NLA map,
  /// but it does not delete the NLA reference from the target operation's
  /// annotations.
  void eraseNLA(hw::HierPathOp nla) {
    // Erase the NLA from the leaf module's nlaMap.
    targetMap.erase(nla.getNameAttr());
    nlaTable->erase(nla);
    nlaCache.erase(nla.getNamepathAttr());
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
    auto moduleNLAs = nlaTable->lookup(fromModule.getNameAttr()).vec();
    // Change the NLA to target the toModule.
    nlaTable->renameModuleAndInnerRef(toName, fromName, renameMap);
    // Now we walk the NLA searching for ones that require more context to be
    // added.
    for (auto nla : moduleNLAs) {
      auto elements = nla.getNamepath().getValue();
      // If we don't need to add more context, we're done here.
      if (nla.root() != toName)
        continue;
      // Create the replacement NLAs.
      SmallVector<Attribute> namepath(elements.begin(), elements.end());
      auto nlaRefs = createNLAs(fromModule, namepath, nla.getVisibility());
      // Copy out the targets, because we will be updating the map.
      auto &set = targetMap[nla.getSymNameAttr()];
      SmallVector<AnnoTarget> targets(set.begin(), set.end());
      // Replace the uses of the old NLA with the new NLAs.
      for (auto target : targets) {
        // We have to clone any annotation which uses the old NLA for each new
        // NLA. This array collects the new set of annotations.
        SmallVector<Annotation> newAnnotations;
        for (auto anno : target.getAnnotations()) {
          // Find the non-local field of the annotation.
          auto [it, found] = mlir::impl::findAttrSorted(
              anno.begin(), anno.end(), nonLocalString);
          // If this annotation doesn't use the target NLA, copy it with no
          // changes.
          if (!found || cast<FlatSymbolRefAttr>(it->getValue()).getAttr() !=
                            nla.getSymNameAttr()) {
            newAnnotations.push_back(anno);
            continue;
          }
          auto nonLocalIndex = std::distance(anno.begin(), it);
          // Clone the annotation and add it to the list of new annotations.
          cloneAnnotation(nlaRefs, anno,
                          ArrayRef<NamedAttribute>(anno.begin(), anno.end()),
                          nonLocalIndex, newAnnotations);
        }

        // Apply the new annotations to the operation.
        AnnotationSet annotations(newAnnotations, context);
        target.setAnnotations(annotations);
        // Record that target uses the NLA.
        for (auto nla : nlaRefs)
          targetMap[nla.getAttr()].insert(target);
      }

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
  /// for now.  Return true if the annotation was made non-local.
  bool makeAnnotationNonLocal(StringAttr toModuleName, AnnoTarget to,
                              FModuleLike fromModule, Annotation anno,
                              SmallVectorImpl<Annotation> &newAnnotations) {
    // Start constructing a new annotation, pushing a "circt.nonLocal" field
    // into the correct spot if its not already a non-local annotation.
    SmallVector<NamedAttribute> attributes;
    int nonLocalIndex = -1;
    for (const auto &val : llvm::enumerate(anno)) {
      auto attr = val.value();
      // Is this field "circt.nonlocal"?
      auto compare = attr.getName().compare(nonLocalString);
      assert(compare != 0 && "should not pass non-local annotations here");
      if (compare == 1) {
        // This annotation definitely does not have "circt.nonlocal" field. Push
        // an empty place holder for the non-local annotation.
        nonLocalIndex = val.index();
        attributes.push_back(NamedAttribute(nonLocalString, nonLocalString));
        break;
      }
      // Otherwise push the current attribute and keep searching for the
      // "circt.nonlocal" field.
      attributes.push_back(attr);
    }
    if (nonLocalIndex == -1) {
      // Push an empty "circt.nonlocal" field to the last slot.
      nonLocalIndex = attributes.size();
      attributes.push_back(NamedAttribute(nonLocalString, nonLocalString));
    } else {
      // Copy the remaining annotation fields in.
      attributes.append(anno.begin() + nonLocalIndex, anno.end());
    }

    // Construct the NLAs if we don't have any yet.
    auto nlaRefs = createNLAs(toModuleName, fromModule);
    for (auto nla : nlaRefs)
      targetMap[nla.getAttr()].insert(to);

    // Clone the annotation for each new NLA.
    cloneAnnotation(nlaRefs, anno, attributes, nonLocalIndex, newAnnotations);
    return true;
  }

  void copyAnnotations(FModuleLike toModule, AnnoTarget to,
                       FModuleLike fromModule, AnnotationSet annos,
                       SmallVectorImpl<Annotation> &newAnnotations,
                       SmallPtrSetImpl<Attribute> &dontTouches) {
    for (auto anno : annos) {
      if (anno.isClass(dontTouchAnnoClass)) {
        // Remove the nonlocal field of the annotation if it has one, since this
        // is a sticky annotation.
        anno.removeMember("circt.nonlocal");
        auto [it, inserted] = dontTouches.insert(anno.getAttr());
        if (inserted)
          newAnnotations.push_back(anno);
        continue;
      }
      // If the annotation is already non-local, we add it as is.  It is already
      // added to the target map.
      if (auto nla = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        newAnnotations.push_back(anno);
        targetMap[nla.getAttr()].insert(to);
        continue;
      }
      // Otherwise make the annotation non-local and add it to the set.
      makeAnnotationNonLocal(toModule.getModuleNameAttr(), to, fromModule, anno,
                             newAnnotations);
    }
  }

  /// Merge the annotations of a specific target, either a operation or a port
  /// on an operation.
  void mergeAnnotations(FModuleLike toModule, AnnoTarget to,
                        AnnotationSet toAnnos, FModuleLike fromModule,
                        AnnoTarget from, AnnotationSet fromAnnos) {
    // This is a list of all the annotations which will be added to `to`.
    SmallVector<Annotation> newAnnotations;

    // We have special case handling of DontTouch to prevent it from being
    // turned into a non-local annotation, and to remove duplicates.
    llvm::SmallPtrSet<Attribute, 4> dontTouches;

    // Iterate the annotations, transforming most annotations into non-local
    // ones.
    copyAnnotations(toModule, to, toModule, toAnnos, newAnnotations,
                    dontTouches);
    copyAnnotations(toModule, to, fromModule, fromAnnos, newAnnotations,
                    dontTouches);

    // Copy over all the new annotations.
    if (!newAnnotations.empty())
      to.setAnnotations(AnnotationSet(newAnnotations, context));
  }

  /// Merge all annotations and port annotations on two operations.
  void mergeAnnotations(FModuleLike toModule, Operation *to,
                        FModuleLike fromModule, Operation *from) {
    // Merge op annotations.
    mergeAnnotations(toModule, OpAnnoTarget(to), AnnotationSet(to), fromModule,
                     OpAnnoTarget(from), AnnotationSet(from));

    // Merge port annotations.
    if (toModule == to) {
      // Merge module port annotations.
      for (unsigned i = 0, e = getNumPorts(toModule); i < e; ++i)
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

  hw::InnerSymAttr mergeInnerSymbols(RenameMap &renameMap, FModuleLike toModule,
                                     hw::InnerSymAttr toSym,
                                     hw::InnerSymAttr fromSym) {
    if (fromSym && !fromSym.getProps().empty()) {
      auto &isn = getNamespace(toModule);
      // The properties for the new inner symbol..
      SmallVector<hw::InnerSymPropertiesAttr> newProps;
      // If the "to" op already has an inner symbol, copy all its properties.
      if (toSym)
        llvm::append_range(newProps, toSym);
      // Add each property from the fromSym to the toSym.
      for (auto fromProp : fromSym) {
        hw::InnerSymPropertiesAttr newProp;
        auto *it = llvm::find_if(newProps, [&](auto p) {
          return p.getFieldID() == fromProp.getFieldID();
        });
        if (it != newProps.end()) {
          // If we already have an inner sym with the same field id, use
          // that.
          newProp = *it;
          // If the old symbol is public, we need to make the new one public.
          if (fromProp.getSymVisibility().getValue() == "public" &&
              newProp.getSymVisibility().getValue() != "public") {
            *it = hw::InnerSymPropertiesAttr::get(context, newProp.getName(),
                                                  newProp.getFieldID(),
                                                  fromProp.getSymVisibility());
          }
        } else {
          // We need to add a new property to the inner symbol for this field.
          auto newName = isn.newName(fromProp.getName().getValue());
          newProp = hw::InnerSymPropertiesAttr::get(
              context, StringAttr::get(context, newName), fromProp.getFieldID(),
              fromProp.getSymVisibility());
          newProps.push_back(newProp);
        }
        renameMap[fromProp.getName()] = newProp.getName();
      }
      // Sort the fields by field id.
      llvm::sort(newProps, [](auto &p, auto &q) {
        return p.getFieldID() < q.getFieldID();
      });
      // Return the merged inner symbol.
      return hw::InnerSymAttr::get(context, newProps);
    }
    return hw::InnerSymAttr();
  }

  // Record the symbol name change of the operation or any of its ports when
  // merging two operations.  The renamed symbols are used to update the
  // target of any NLAs.  This will add symbols to the "to" operation if needed.
  void recordSymRenames(RenameMap &renameMap, FModuleLike toModule,
                        Operation *to, FModuleLike fromModule,
                        Operation *from) {
    // If the "from" operation has an inner_sym, we need to make sure the
    // "to" operation also has an `inner_sym` and then record the renaming.
    if (auto fromInnerSym = dyn_cast<hw::InnerSymbolOpInterface>(from)) {
      auto toInnerSym = cast<hw::InnerSymbolOpInterface>(to);
      if (auto newSymAttr = mergeInnerSymbols(renameMap, toModule,
                                              toInnerSym.getInnerSymAttr(),
                                              fromInnerSym.getInnerSymAttr()))
        toInnerSym.setInnerSymbolAttr(newSymAttr);
    }

    // If there are no port symbols on the "from" operation, we are done here.
    auto fromPortSyms = from->getAttrOfType<ArrayAttr>("portSymbols");
    if (!fromPortSyms || fromPortSyms.empty())
      return;
    // We have to map each "fromPort" to each "toPort".
    auto portCount = fromPortSyms.size();
    auto toPortSyms = to->getAttrOfType<ArrayAttr>("portSymbols");

    // Create an array of new port symbols for the "to" operation, copy in the
    // old symbols if it has any, create an empty symbol array if it doesn't.
    SmallVector<Attribute> newPortSyms;
    if (toPortSyms.empty())
      newPortSyms.assign(portCount, hw::InnerSymAttr());
    else
      newPortSyms.assign(toPortSyms.begin(), toPortSyms.end());

    for (unsigned portNo = 0; portNo < portCount; ++portNo) {
      if (auto newPortSym = mergeInnerSymbols(
              renameMap, toModule,
              llvm::cast_if_present<hw::InnerSymAttr>(newPortSyms[portNo]),
              cast<hw::InnerSymAttr>(fromPortSyms[portNo]))) {
        newPortSyms[portNo] = newPortSym;
      }
    }

    // Commit the new symbol attribute.
    FModuleLike::fixupPortSymsArray(newPortSyms, toModule.getContext());
    cast<FModuleLike>(to).setPortSymbols(newPortSyms);
  }

  /// Recursively merge two operations.
  // NOLINTNEXTLINE(misc-no-recursion)
  void mergeOps(RenameMap &renameMap, FModuleLike toModule, Operation *to,
                FModuleLike fromModule, Operation *from) {
    // Merge the operation locations.
    if (to->getLoc() != from->getLoc())
      to->setLoc(mergeLoc(context, to->getLoc(), from->getLoc()));

    // Recurse into any regions.
    for (auto regions : llvm::zip(to->getRegions(), from->getRegions()))
      mergeRegions(renameMap, toModule, std::get<0>(regions), fromModule,
                   std::get<1>(regions));

    // Record any inner_sym renamings that happened.
    recordSymRenames(renameMap, toModule, to, fromModule, from);

    // Merge the annotations.
    mergeAnnotations(toModule, to, fromModule, from);
  }

  /// Recursively merge two blocks.
  void mergeBlocks(RenameMap &renameMap, FModuleLike toModule, Block &toBlock,
                   FModuleLike fromModule, Block &fromBlock) {
    // Merge the block locations.
    for (auto [toArg, fromArg] :
         llvm::zip(toBlock.getArguments(), fromBlock.getArguments()))
      if (toArg.getLoc() != fromArg.getLoc())
        toArg.setLoc(mergeLoc(context, toArg.getLoc(), fromArg.getLoc()));

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

  // This maps an NLA  to the operations and ports that uses it.
  DenseMap<Attribute, llvm::SmallDenseSet<AnnoTarget>> targetMap;

  // This is a cache to avoid creating duplicate NLAs.  This maps the ArrayAtr
  // of the NLA's path to the name of the NLA which contains it.
  DenseMap<Attribute, Attribute> nlaCache;

  // Cached attributes for faster comparisons and attribute building.
  StringAttr nonLocalString;
  StringAttr classString;

  /// A module namespace cache.
  DenseMap<Operation *, hw::InnerSymbolNamespace> moduleNamespaces;
};

//===----------------------------------------------------------------------===//
// Fixup
//===----------------------------------------------------------------------===//

/// This fixes up ClassLikes with ClassType ports, when the classes have
/// deduped. For each ClassType port, if the object reference being assigned is
/// a different type, update the port type. Returns true if the ClassOp was
/// updated and the associated ObjectOps should be updated.
bool fixupClassOp(ClassOp classOp) {
  // New port type attributes, if necessary.
  SmallVector<Attribute> newPortTypes;
  bool anyDifferences = false;

  // Check each port.
  for (size_t i = 0, e = classOp.getNumPorts(); i < e; ++i) {
    // Check if this port is a ClassType. If not, save the original type
    // attribute in case we need to update port types.
    auto portClassType = dyn_cast<ClassType>(classOp.getPortType(i));
    if (!portClassType) {
      newPortTypes.push_back(classOp.getPortTypeAttr(i));
      continue;
    }

    // Check if this port is assigned a reference of a different ClassType.
    Type newPortClassType;
    BlockArgument portArg = classOp.getArgument(i);
    for (auto &use : portArg.getUses()) {
      if (auto propassign = dyn_cast<PropAssignOp>(use.getOwner())) {
        Type sourceType = propassign.getSrc().getType();
        if (propassign.getDest() == use.get() && sourceType != portClassType) {
          // Double check that all references are the same new type.
          if (newPortClassType) {
            assert(newPortClassType == sourceType &&
                   "expected all references to be of the same type");
            continue;
          }

          newPortClassType = sourceType;
        }
      }
    }

    // If there was no difference, save the original type attribute in case we
    // need to update port types and move along.
    if (!newPortClassType) {
      newPortTypes.push_back(classOp.getPortTypeAttr(i));
      continue;
    }

    // The port type changed, so update the block argument, save the new port
    // type attribute, and indicate there was a difference.
    classOp.getArgument(i).setType(newPortClassType);
    newPortTypes.push_back(TypeAttr::get(newPortClassType));
    anyDifferences = true;
  }

  // If necessary, update port types.
  if (anyDifferences)
    classOp.setPortTypes(newPortTypes);

  return anyDifferences;
}

/// This fixes up ObjectOps when the signature of their ClassOp changes. This
/// amounts to updating the ObjectOp result type to match the newly updated
/// ClassOp type.
void fixupObjectOp(ObjectOp objectOp, ClassType newClassType) {
  objectOp.getResult().setType(newClassType);
}

/// This fixes up connects when the field names of a bundle type changes.  It
/// finds all fields which were previously bulk connected and legalizes it
/// into a connect for each field.
void fixupConnect(ImplicitLocOpBuilder &builder, Value dst, Value src) {
  // If the types already match we can emit a connect.
  auto dstType = dst.getType();
  auto srcType = src.getType();
  if (dstType == srcType) {
    emitConnect(builder, dst, src);
    return;
  }
  // It must be a bundle type and the field name has changed. We have to
  // manually decompose the bulk connect into a connect for each field.
  auto dstBundle = type_cast<BundleType>(dstType);
  auto srcBundle = type_cast<BundleType>(srcType);
  for (unsigned i = 0; i < dstBundle.getNumElements(); ++i) {
    auto dstField = SubfieldOp::create(builder, dst, i);
    auto srcField = SubfieldOp::create(builder, src, i);
    if (dstBundle.getElement(i).isFlip) {
      std::swap(srcBundle, dstBundle);
      std::swap(srcField, dstField);
    }
    fixupConnect(builder, dstField, srcField);
  }
}

/// This is the root method to fixup module references when a module changes.
/// It matches all the results of "to" module with the results of the "from"
/// module.
void fixupAllModules(InstanceGraph &instanceGraph) {
  for (auto *node : instanceGraph) {
    auto module = cast<FModuleLike>(*node->getModule());

    // Handle class declarations here.
    bool shouldFixupObjects = false;
    auto classOp = dyn_cast<ClassOp>(module.getOperation());
    if (classOp)
      shouldFixupObjects = fixupClassOp(classOp);

    for (auto *instRec : node->uses()) {
      // Handle object instantiations here.
      if (classOp) {
        if (shouldFixupObjects) {
          fixupObjectOp(instRec->getInstance<ObjectOp>(),
                        classOp.getInstanceType());
        }
        continue;
      }

      auto inst = instRec->getInstance<InstanceOp>();
      // Only handle module instantiations here.
      if (!inst)
        continue;
      ImplicitLocOpBuilder builder(inst.getLoc(), inst->getContext());
      builder.setInsertionPointAfter(inst);
      for (size_t i = 0, e = getNumPorts(module); i < e; ++i) {
        auto result = inst.getResult(i);
        auto newType = module.getPortType(i);
        auto oldType = result.getType();
        // If the type has not changed, we don't have to fix up anything.
        if (newType == oldType)
          continue;
        // If the type changed we transform it back to the old type with an
        // intermediate wire.
        auto wire =
            WireOp::create(builder, oldType, inst.getPortName(i)).getResult();
        result.replaceAllUsesWith(wire);
        result.setType(newType);
        if (inst.getPortDirection(i) == Direction::Out)
          fixupConnect(builder, wire, result);
        else
          fixupConnect(builder, result, wire);
      }
    }
  }
}

namespace llvm {
/// A DenseMapInfo implementation for `ModuleInfo` that is a pair of
/// llvm::SHA256 hashes, which are represented as std::array<uint8_t, 32>, and
/// an array of string attributes. This allows us to create a DenseMap with
/// `ModuleInfo` as keys.
template <>
struct DenseMapInfo<ModuleInfo> {
  static inline ModuleInfo getEmptyKey() {
    std::array<uint8_t, 32> key;
    std::fill(key.begin(), key.end(), ~0);
    return {key, {}};
  }

  static inline ModuleInfo getTombstoneKey() {
    std::array<uint8_t, 32> key;
    std::fill(key.begin(), key.end(), ~0 - 1);
    return {key, {}};
  }

  static unsigned getHashValue(const ModuleInfo &val) {
    // We assume SHA256 is already a good hash and just truncate down to the
    // number of bytes we need for DenseMap.
    unsigned hash;
    std::memcpy(&hash, val.structuralHash.data(), sizeof(unsigned));

    // Combine module names.
    return llvm::hash_combine(
        hash, llvm::hash_combine_range(val.referredModuleNames.begin(),
                                       val.referredModuleNames.end()));
  }

  static bool isEqual(const ModuleInfo &lhs, const ModuleInfo &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// DedupPass
//===----------------------------------------------------------------------===//

namespace {
class DedupPass : public circt::firrtl::impl::DedupBase<DedupPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto circuit = getOperation();
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    auto *nlaTable = &getAnalysis<NLATable>();
    auto &symbolTable = getAnalysis<SymbolTable>();
    Deduper deduper(instanceGraph, symbolTable, nlaTable, circuit);
    Equivalence equiv(context, instanceGraph);
    auto anythingChanged = false;

    // Modules annotated with this should not be considered for deduplication.
    auto noDedupClass = StringAttr::get(context, noDedupAnnoClass);

    // Only modules within the same group may be deduplicated.
    auto dedupGroupClass = StringAttr::get(context, dedupGroupAnnoClass);

    // A map of all the module moduleInfo that we have calculated so far.
    llvm::DenseMap<ModuleInfo, Operation *> moduleInfoToModule;

    // We track the name of the module that each module is deduped into, so that
    // we can make sure all modules which are marked "must dedup" with each
    // other were all deduped to the same module.
    DenseMap<Attribute, StringAttr> dedupMap;

    // We must iterate the modules from the bottom up so that we can properly
    // deduplicate the modules. We copy the list of modules into a vector first
    // to avoid iterator invalidation while we mutate the instance graph.
    SmallVector<FModuleLike, 0> modules(
        llvm::map_range(llvm::post_order(&instanceGraph), [](auto *node) {
          return cast<FModuleLike>(*node->getModule());
        }));

    SmallVector<std::optional<ModuleInfo>> moduleInfos(modules.size());
    StructuralHasherSharedConstants hasherConstants(&getContext());

    // Attribute name used to store dedup_group for this pass.
    auto dedupGroupAttrName = StringAttr::get(context, "firrtl.dedup_group");

    // Move dedup group annotations to attributes on the module.
    // This results in the desired behavior (included in hash),
    // and avoids unnecessary processing of these as annotations
    // that need to be tracked, made non-local, so on.
    for (auto module : modules) {
      llvm::SmallSetVector<StringAttr, 1> groups;
      AnnotationSet::removeAnnotations(
          module, [&groups, dedupGroupClass](Annotation annotation) {
            if (annotation.getClassAttr() != dedupGroupClass)
              return false;
            groups.insert(annotation.getMember<StringAttr>("group"));
            return true;
          });
      if (groups.size() > 1) {
        module.emitError("module belongs to multiple dedup groups: ") << groups;
        return signalPassFailure();
      }
      assert(!module->hasAttr(dedupGroupAttrName) &&
             "unexpected existing use of temporary dedup group attribute");
      if (!groups.empty())
        module->setDiscardableAttr(dedupGroupAttrName, groups.front());
    }

    // Calculate module information parallelly.
    auto result = mlir::failableParallelForEach(
        context, llvm::seq(modules.size()), [&](unsigned idx) {
          auto module = modules[idx];
          // If the module is marked with NoDedup, just skip it.
          if (AnnotationSet::hasAnnotation(module, noDedupClass))
            return success();

          // Only dedup extmodule's with defname.
          if (auto ext = dyn_cast<FExtModuleOp>(*module);
              ext && !ext.getDefname().has_value())
            return success();

          StructuralHasher hasher(hasherConstants);
          // Calculate the hash of the module and referred module names.
          moduleInfos[idx] = hasher.getModuleInfo(module);
          return success();
        });

    if (result.failed())
      return signalPassFailure();

    for (auto [i, module] : llvm::enumerate(modules)) {
      auto moduleName = module.getModuleNameAttr();
      auto &maybeModuleInfo = moduleInfos[i];
      // If the hash was not calculated, we need to skip it.
      if (!maybeModuleInfo) {
        // We record it in the dedup map to help detect errors when the user
        // marks the module as both NoDedup and MustDedup. We do not record this
        // module in the hasher to make sure no other module dedups "into" this
        // one.
        dedupMap[moduleName] = moduleName;
        continue;
      }

      auto &moduleInfo = maybeModuleInfo.value();

      // Replace module names referred in the module with new names.
      for (auto &referredModule : moduleInfo.referredModuleNames)
        referredModule = dedupMap[referredModule];

      // Check if there a module with the same hash.
      auto it = moduleInfoToModule.find(moduleInfo);
      if (it != moduleInfoToModule.end()) {
        auto original = cast<FModuleLike>(it->second);
        auto originalName = original.getModuleNameAttr();

        // If the current module is public, and the original is private, we
        // want to dedup the private module into the public one.
        if (!canRemoveModule(module)) {
          // If both modules are public, then we can't dedup anything.
          if (!canRemoveModule(original))
            continue;
          // Swap the canonical module in the dedup map.
          for (auto &[originalName, dedupedName] : dedupMap)
            if (dedupedName == originalName)
              dedupedName = moduleName;
          // Update the module hash table to point to the new original, so all
          // future modules dedup with the new canonical module.
          it->second = module;
          // Swap the locals.
          std::swap(originalName, moduleName);
          std::swap(original, module);
        }

        // Record the group ID of the other module.
        dedupMap[moduleName] = originalName;
        deduper.dedup(original, module);
        ++erasedModules;
        anythingChanged = true;
        continue;
      }
      // Any module not deduplicated must be recorded.
      deduper.record(module);
      // Add the module to a new dedup group.
      dedupMap[moduleName] = moduleName;
      // Record the module info.
      moduleInfoToModule[std::move(moduleInfo)] = module;
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
      auto [_, rhs] = cast<StringAttr>(path).getValue().split('|');
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
        return nullptr;
      }
      return it->second;
    };

    AnnotationSet::removeAnnotations(circuit, [&](Annotation annotation) {
      if (!annotation.isClass(mustDedupAnnoClass))
        return false;
      auto modules = annotation.getMember<ArrayAttr>("modules");
      if (!modules) {
        emitError(circuit.getLoc(),
                  "MustDeduplicateAnnotation missing \"modules\" member");
        failed = true;
        return false;
      }
      // Empty module list has nothing to process.
      if (modules.empty())
        return true;
      // Get the first element.
      auto firstModule = parseModule(modules[0]);
      auto firstLead = getLead(firstModule);
      if (!firstLead)
        return false;
      // Verify that the remaining elements are all the same as the first.
      for (auto attr : modules.getValue().drop_front()) {
        auto nextModule = parseModule(attr);
        auto nextLead = getLead(nextModule);
        if (!nextLead)
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

    // Remove all dedup group attributes, they only exist during this pass.
    for (auto module : circuit.getOps<FModuleLike>())
      module->removeDiscardableAttr(dedupGroupAttrName);

    // Walk all the modules and fixup the instance operation to return the
    // correct type. We delay this fixup until the end because doing it early
    // can block the deduplication of the parent modules.
    fixupAllModules(instanceGraph);

    markAnalysesPreserved<NLATable>();
    if (!anythingChanged)
      markAllAnalysesPreserved();
  }
};
} // end anonymous namespace
