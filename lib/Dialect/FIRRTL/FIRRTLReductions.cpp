//===- FIRRTLReductions.cpp - Reduction patterns for the FIRRTL dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLReductions.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/LayerSet.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Reduce/ReductionUtils.h"
#include "circt/Support/Namespace.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-reductions"

using namespace mlir;
using namespace circt;
using namespace firrtl;
using llvm::MapVector;
using llvm::SmallDenseSet;
using llvm::SmallSetVector;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace detail {
/// A utility doing lazy construction of `SymbolTable`s and `SymbolUserMap`s,
/// which is handy for reductions that need to look up a lot of symbols.
struct SymbolCache {
  SymbolCache() : tables(std::make_unique<SymbolTableCollection>()) {}

  SymbolTable &getSymbolTable(Operation *op) {
    return tables->getSymbolTable(op);
  }
  SymbolTable &getNearestSymbolTable(Operation *op) {
    return getSymbolTable(SymbolTable::getNearestSymbolTable(op));
  }

  SymbolUserMap &getSymbolUserMap(Operation *op) {
    auto it = userMaps.find(op);
    if (it != userMaps.end())
      return it->second;
    return userMaps.insert({op, SymbolUserMap(*tables, op)}).first->second;
  }
  SymbolUserMap &getNearestSymbolUserMap(Operation *op) {
    return getSymbolUserMap(SymbolTable::getNearestSymbolTable(op));
  }

  void clear() {
    tables = std::make_unique<SymbolTableCollection>();
    userMaps.clear();
  }

private:
  std::unique_ptr<SymbolTableCollection> tables;
  SmallDenseMap<Operation *, SymbolUserMap, 2> userMaps;
};
} // namespace detail

/// Utility to easily get the instantiated firrtl::FModuleOp or an empty
/// optional in case another type of module is instantiated.
static std::optional<firrtl::FModuleOp>
findInstantiatedModule(firrtl::InstanceOp instOp,
                       ::detail::SymbolCache &symbols) {
  auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
  auto moduleOp = dyn_cast<firrtl::FModuleOp>(
      instOp.getReferencedOperation(symbols.getSymbolTable(tableOp)));
  return moduleOp ? std::optional(moduleOp) : std::nullopt;
}

/// Utility to track the transitive size of modules.
struct ModuleSizeCache {
  void clear() { moduleSizes.clear(); }

  uint64_t getModuleSize(Operation *module, ::detail::SymbolCache &symbols) {
    if (auto it = moduleSizes.find(module); it != moduleSizes.end())
      return it->second;
    uint64_t size = 1;
    module->walk([&](Operation *op) {
      size += 1;
      if (auto instOp = dyn_cast<firrtl::InstanceOp>(op))
        if (auto instModule = findInstantiatedModule(instOp, symbols))
          size += getModuleSize(*instModule, symbols);
    });
    moduleSizes.insert({module, size});
    return size;
  }

private:
  llvm::DenseMap<Operation *, uint64_t> moduleSizes;
};

/// Check that all connections to a value are invalids.
static bool onlyInvalidated(Value arg) {
  return llvm::all_of(arg.getUses(), [](OpOperand &use) {
    auto *op = use.getOwner();
    if (!isa<firrtl::ConnectOp, firrtl::MatchingConnectOp>(op))
      return false;
    if (use.getOperandNumber() != 0)
      return false;
    if (!op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>())
      return false;
    return true;
  });
}

/// A tracker for track NLAs affected by a reduction. Performs the necessary
/// cleanup steps in order to maintain IR validity after the reduction has
/// applied. For example, removing an instance that forms part of an NLA path
/// requires that NLA to be removed as well.
struct NLARemover {
  /// Clear the set of marked NLAs. Call this before attempting a reduction.
  void clear() { nlasToRemove.clear(); }

  /// Remove all marked annotations. Call this after applying a reduction in
  /// order to validate the IR.
  void remove(mlir::ModuleOp module) {
    unsigned numRemoved = 0;
    (void)numRemoved;
    SymbolTableCollection symbolTables;
    for (Operation &rootOp : *module.getBody()) {
      if (!isa<firrtl::CircuitOp>(&rootOp))
        continue;
      SymbolUserMap symbolUserMap(symbolTables, &rootOp);
      auto &symbolTable = symbolTables.getSymbolTable(&rootOp);
      for (auto sym : nlasToRemove) {
        if (auto *op = symbolTable.lookup(sym)) {
          if (symbolUserMap.useEmpty(op)) {
            ++numRemoved;
            op->erase();
          }
        }
      }
    }
    LLVM_DEBUG({
      unsigned numLost = nlasToRemove.size() - numRemoved;
      if (numRemoved > 0 || numLost > 0) {
        llvm::dbgs() << "Removed " << numRemoved << " NLAs";
        if (numLost > 0)
          llvm::dbgs() << " (" << numLost << " no longer there)";
        llvm::dbgs() << "\n";
      }
    });
  }

  /// Mark all NLAs referenced in the given annotation as to be removed. This
  /// can be an entire array or dictionary of annotations, and the function will
  /// descend into child annotations appropriately.
  void markNLAsInAnnotation(Attribute anno) {
    if (auto dict = dyn_cast<DictionaryAttr>(anno)) {
      if (auto field = dict.getAs<FlatSymbolRefAttr>("circt.nonlocal"))
        nlasToRemove.insert(field.getAttr());
      for (auto namedAttr : dict)
        markNLAsInAnnotation(namedAttr.getValue());
    } else if (auto array = dyn_cast<ArrayAttr>(anno)) {
      for (auto attr : array)
        markNLAsInAnnotation(attr);
    }
  }

  /// Mark all NLAs referenced in an operation. Also traverses all nested
  /// operations. Call this before removing an operation, to mark any associated
  /// NLAs as to be removed as well.
  void markNLAsInOperation(Operation *op) {
    op->walk([&](Operation *op) {
      if (auto annos = op->getAttrOfType<ArrayAttr>("annotations"))
        markNLAsInAnnotation(annos);
    });
  }

  /// The set of NLAs to remove, identified by their symbol.
  llvm::DenseSet<StringAttr> nlasToRemove;
};

//===----------------------------------------------------------------------===//
// Reduction patterns
//===----------------------------------------------------------------------===//

namespace {

/// A sample reduction pattern that maps `firrtl.module` to `firrtl.extmodule`.
struct FIRRTLModuleExternalizer : public OpReduction<FModuleOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    nlaRemover.clear();
    symbols.clear();
    moduleSizes.clear();
    innerSymUses = reduce::InnerSymbolUses(op);
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(FModuleOp module) override {
    if (innerSymUses.hasInnerRef(module))
      return 0;
    return moduleSizes.getModuleSize(module, symbols);
  }

  LogicalResult rewrite(FModuleOp module) override {
    // Hack up a list of known layers.
    LayerSet layers;
    layers.insert_range(module.getLayersAttr().getAsRange<SymbolRefAttr>());
    for (auto attr : module.getPortTypes()) {
      auto type = cast<TypeAttr>(attr).getValue();
      if (auto refType = type_dyn_cast<RefType>(type))
        if (auto layer = refType.getLayer())
          layers.insert(layer);
    }
    SmallVector<Attribute, 4> layersArray;
    layersArray.reserve(layers.size());
    for (auto layer : layers)
      layersArray.push_back(layer);

    nlaRemover.markNLAsInOperation(module);
    OpBuilder builder(module);
    auto extmodule = FExtModuleOp::create(
        builder, module->getLoc(),
        module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        module.getConventionAttr(), module.getPorts(),
        builder.getArrayAttr(layersArray), StringRef(),
        module.getAnnotationsAttr());
    SymbolTable::setSymbolVisibility(extmodule,
                                     SymbolTable::getSymbolVisibility(module));
    module->erase();
    return success();
  }

  std::string getName() const override { return "firrtl-module-externalizer"; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  reduce::InnerSymbolUses innerSymUses;
  ModuleSizeCache moduleSizes;
};

/// Invalidate all the leaf fields of a value with a given flippedness by
/// connecting an invalid value to them. This function handles different FIRRTL
/// types appropriately:
/// - Ref types (probes): Creates wire infrastructure with ref.send/ref.define
///   and invalidates the underlying wire.
/// - Bundle/Vector types: Recursively descends into elements.
/// - Base types: Creates InvalidValueOp and connects it.
/// - Property types: Creates UnknownValueOp and assigns it.
///
/// This is useful for ensuring that all output ports of an instance or memory
/// (including those nested in bundles) are properly invalidated.
static void invalidateOutputs(ImplicitLocOpBuilder &builder, Value value,
                              TieOffCache &tieOffCache, bool flip = false) {
  auto type = type_dyn_cast<FIRRTLType>(value.getType());
  if (!type)
    return;

  // Handle ref types (probes) by creating wires and defining them properly.
  if (auto refType = type_dyn_cast<RefType>(type)) {
    // Input probes are illegal in FIRRTL.
    assert(!flip && "input probes are not allowed");

    auto underlyingType = refType.getType();

    if (!refType.getForceable()) {
      // For probe types: create underlying wire, ref.send, ref.define, and
      // invalidate.
      auto targetWire = WireOp::create(builder, underlyingType);
      auto refSend = builder.create<RefSendOp>(targetWire.getResult());
      builder.create<RefDefineOp>(value, refSend.getResult());

      // Invalidate the underlying wire.
      auto invalid = tieOffCache.getInvalid(underlyingType);
      MatchingConnectOp::create(builder, targetWire.getResult(), invalid);
      return;
    }

    // For rwprobe types: create forceable wire, ref.define, and invalidate.
    auto forceableWire =
        WireOp::create(builder, underlyingType,
                       /*name=*/"", NameKindEnum::DroppableName,
                       /*annotations=*/ArrayRef<Attribute>{},
                       /*innerSym=*/StringAttr{},
                       /*forceable=*/true);

    // The forceable wire returns both the wire and the rwprobe.
    auto targetWire = forceableWire.getResult();
    auto forceableRef = forceableWire.getDataRef();

    builder.create<RefDefineOp>(value, forceableRef);

    // Invalidate the underlying wire.
    auto invalid = tieOffCache.getInvalid(underlyingType);
    MatchingConnectOp::create(builder, targetWire, invalid);
    return;
  }

  // Descend into bundles by creating subfield ops.
  if (auto bundleType = type_dyn_cast<BundleType>(type)) {
    for (auto element : llvm::enumerate(bundleType.getElements())) {
      auto subfield = builder.createOrFold<SubfieldOp>(value, element.index());
      invalidateOutputs(builder, subfield, tieOffCache,
                        flip ^ element.value().isFlip);
      if (subfield.use_empty())
        subfield.getDefiningOp()->erase();
    }
    return;
  }

  // Descend into vectors by creating subindex ops.
  if (auto vectorType = type_dyn_cast<FVectorType>(type)) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i) {
      auto subindex = builder.createOrFold<SubindexOp>(value, i);
      invalidateOutputs(builder, subindex, tieOffCache, flip);
      if (subindex.use_empty())
        subindex.getDefiningOp()->erase();
    }
    return;
  }

  // Only drive outputs.
  if (flip)
    return;

  // Create InvalidValueOp for FIRRTLBaseType.
  if (auto baseType = type_dyn_cast<FIRRTLBaseType>(type)) {
    auto invalid = tieOffCache.getInvalid(baseType);
    ConnectOp::create(builder, value, invalid);
    return;
  }

  // For property types, use UnknownValueOp to tie off the connection.
  if (auto propType = type_dyn_cast<PropertyType>(type)) {
    auto unknown = tieOffCache.getUnknown(propType);
    builder.create<PropAssignOp>(value, unknown);
  }
}

/// Connect a value to every leave of a destination value.
static void connectToLeafs(ImplicitLocOpBuilder &builder, Value dest,
                           Value value) {
  auto type = dyn_cast<firrtl::FIRRTLBaseType>(dest.getType());
  if (!type)
    return;
  if (auto bundleType = dyn_cast<firrtl::BundleType>(type)) {
    for (auto element : llvm::enumerate(bundleType.getElements()))
      connectToLeafs(builder,
                     firrtl::SubfieldOp::create(builder, dest, element.index()),
                     value);
    return;
  }
  if (auto vectorType = dyn_cast<firrtl::FVectorType>(type)) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i)
      connectToLeafs(builder, firrtl::SubindexOp::create(builder, dest, i),
                     value);
    return;
  }
  auto valueType = dyn_cast<firrtl::FIRRTLBaseType>(value.getType());
  if (!valueType)
    return;
  auto destWidth = type.getBitWidthOrSentinel();
  auto valueWidth = valueType ? valueType.getBitWidthOrSentinel() : -1;
  if (destWidth >= 0 && valueWidth >= 0 && destWidth < valueWidth)
    value = firrtl::HeadPrimOp::create(builder, value, destWidth);
  if (!isa<firrtl::UIntType>(type)) {
    if (isa<firrtl::SIntType>(type))
      value = firrtl::AsSIntPrimOp::create(builder, value);
    else
      return;
  }
  firrtl::ConnectOp::create(builder, dest, value);
}

/// Reduce all leaf fields of a value through an XOR tree.
static void reduceXor(ImplicitLocOpBuilder &builder, Value &into, Value value) {
  auto type = dyn_cast<firrtl::FIRRTLType>(value.getType());
  if (!type)
    return;
  if (auto bundleType = dyn_cast<firrtl::BundleType>(type)) {
    for (auto element : llvm::enumerate(bundleType.getElements()))
      reduceXor(
          builder, into,
          builder.createOrFold<firrtl::SubfieldOp>(value, element.index()));
    return;
  }
  if (auto vectorType = dyn_cast<firrtl::FVectorType>(type)) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i)
      reduceXor(builder, into,
                builder.createOrFold<firrtl::SubindexOp>(value, i));
    return;
  }
  if (!isa<firrtl::UIntType>(type)) {
    if (isa<firrtl::SIntType>(type))
      value = firrtl::AsUIntPrimOp::create(builder, value);
    else
      return;
  }
  into = into ? builder.createOrFold<firrtl::XorPrimOp>(into, value) : value;
}

/// A sample reduction pattern that maps `firrtl.instance` to a set of
/// invalidated wires. This often shortcuts a long iterative process of connect
/// invalidation, module externalization, and wire stripping
struct InstanceStubber : public OpReduction<firrtl::InstanceOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    erasedInsts.clear();
    erasedModules.clear();
    symbols.clear();
    nlaRemover.clear();
    moduleSizes.clear();
  }
  void afterReduction(mlir::ModuleOp op) override {
    // Look into deleted modules to find additional instances that are no longer
    // instantiated anywhere.
    SmallVector<Operation *> worklist;
    auto deadInsts = erasedInsts;
    for (auto *op : erasedModules)
      worklist.push_back(op);
    while (!worklist.empty()) {
      auto *op = worklist.pop_back_val();
      auto *tableOp = SymbolTable::getNearestSymbolTable(op);
      op->walk([&](firrtl::InstanceOp instOp) {
        auto moduleOp = cast<firrtl::FModuleLike>(
            instOp.getReferencedOperation(symbols.getSymbolTable(tableOp)));
        deadInsts.insert(instOp);
        if (llvm::all_of(
                symbols.getSymbolUserMap(tableOp).getUsers(moduleOp),
                [&](Operation *user) { return deadInsts.contains(user); })) {
          LLVM_DEBUG(llvm::dbgs() << "- Removing transitively unused module `"
                                  << moduleOp.getModuleName() << "`\n");
          erasedModules.insert(moduleOp);
          worklist.push_back(moduleOp);
        }
      });
    }

    for (auto *op : erasedInsts)
      op->erase();
    for (auto *op : erasedModules)
      op->erase();
    nlaRemover.remove(op);
  }

  uint64_t match(firrtl::InstanceOp instOp) override {
    if (auto fmoduleOp = findInstantiatedModule(instOp, symbols))
      return moduleSizes.getModuleSize(*fmoduleOp, symbols);
    return 0;
  }

  LogicalResult rewrite(firrtl::InstanceOp instOp) override {
    LLVM_DEBUG(llvm::dbgs()
               << "Stubbing instance `" << instOp.getName() << "`\n");
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    TieOffCache tieOffCache(builder);
    for (unsigned i = 0, e = instOp.getNumResults(); i != e; ++i) {
      auto result = instOp.getResult(i);
      auto name = builder.getStringAttr(Twine(instOp.getName()) + "_" +
                                        instOp.getPortName(i));
      auto wire =
          firrtl::WireOp::create(builder, result.getType(), name,
                                 firrtl::NameKindEnum::DroppableName,
                                 instOp.getPortAnnotation(i), StringAttr{})
              .getResult();
      invalidateOutputs(builder, wire, tieOffCache,
                        instOp.getPortDirection(i) == firrtl::Direction::In);
      result.replaceAllUsesWith(wire);
    }
    auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = cast<firrtl::FModuleLike>(
        instOp.getReferencedOperation(symbols.getSymbolTable(tableOp)));
    nlaRemover.markNLAsInOperation(instOp);
    erasedInsts.insert(instOp);
    if (llvm::all_of(
            symbols.getSymbolUserMap(tableOp).getUsers(moduleOp),
            [&](Operation *user) { return erasedInsts.contains(user); })) {
      LLVM_DEBUG(llvm::dbgs() << "- Removing now unused module `"
                              << moduleOp.getModuleName() << "`\n");
      erasedModules.insert(moduleOp);
    }
    return success();
  }

  std::string getName() const override { return "instance-stubber"; }
  bool acceptSizeIncrease() const override { return true; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  llvm::DenseSet<Operation *> erasedInsts;
  llvm::DenseSet<Operation *> erasedModules;
  ModuleSizeCache moduleSizes;
};

/// A sample reduction pattern that maps `firrtl.mem` to a set of invalidated
/// wires.
struct MemoryStubber : public OpReduction<firrtl::MemOp> {
  void beforeReduction(mlir::ModuleOp op) override { nlaRemover.clear(); }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }
  LogicalResult rewrite(firrtl::MemOp memOp) override {
    LLVM_DEBUG(llvm::dbgs() << "Stubbing memory `" << memOp.getName() << "`\n");
    ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
    TieOffCache tieOffCache(builder);
    Value xorInputs;
    SmallVector<Value> outputs;
    for (unsigned i = 0, e = memOp.getNumResults(); i != e; ++i) {
      auto result = memOp.getResult(i);
      auto name = builder.getStringAttr(Twine(memOp.getName()) + "_" +
                                        memOp.getPortName(i));
      auto wire =
          firrtl::WireOp::create(builder, result.getType(), name,
                                 firrtl::NameKindEnum::DroppableName,
                                 memOp.getPortAnnotation(i), StringAttr{})
              .getResult();
      invalidateOutputs(builder, wire, tieOffCache, true);
      result.replaceAllUsesWith(wire);

      // Isolate the input and output data fields of the port.
      Value input, output;
      switch (memOp.getPortKind(i)) {
      case firrtl::MemOp::PortKind::Read:
        output = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::Write:
        input = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::ReadWrite:
        input = builder.createOrFold<firrtl::SubfieldOp>(wire, 5);
        output = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::Debug:
        output = wire;
        break;
      }

      if (!isa<firrtl::RefType>(result.getType())) {
        // Reduce all input ports to a single one through an XOR tree.
        unsigned numFields =
            cast<firrtl::BundleType>(wire.getType()).getNumElements();
        for (unsigned i = 0; i != numFields; ++i) {
          if (i != 2 && i != 3 && i != 5)
            reduceXor(builder, xorInputs,
                      builder.createOrFold<firrtl::SubfieldOp>(wire, i));
        }
        if (input)
          reduceXor(builder, xorInputs, input);
      }

      // Track the output port to hook it up to the XORd input later.
      if (output)
        outputs.push_back(output);
    }

    // Hook up the outputs.
    for (auto output : outputs)
      connectToLeafs(builder, output, xorInputs);

    nlaRemover.markNLAsInOperation(memOp);
    memOp->erase();
    return success();
  }
  std::string getName() const override { return "memory-stubber"; }
  bool acceptSizeIncrease() const override { return true; }
  NLARemover nlaRemover;
};

/// Check whether an operation interacts with flows in any way, which can make
/// replacement and operand forwarding harder in some cases.
static bool isFlowSensitiveOp(Operation *op) {
  return isa<WireOp, RegOp, RegResetOp, InstanceOp, SubfieldOp, SubindexOp,
             SubaccessOp, ObjectSubfieldOp>(op);
}

/// A sample reduction pattern that replaces all uses of an operation with one
/// of its operands. This can help pruning large parts of the expression tree
/// rapidly.
template <unsigned OpNum>
struct FIRRTLOperandForwarder : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() != 1 || OpNum >= op->getNumOperands())
      return 0;
    if (isFlowSensitiveOp(op))
      return 0;
    auto resultTy =
        dyn_cast<firrtl::FIRRTLBaseType>(op->getResult(0).getType());
    auto opTy =
        dyn_cast<firrtl::FIRRTLBaseType>(op->getOperand(OpNum).getType());
    return resultTy && opTy &&
           resultTy.getWidthlessType() == opTy.getWidthlessType() &&
           (resultTy.getBitWidthOrSentinel() == -1) ==
               (opTy.getBitWidthOrSentinel() == -1) &&
           isa<firrtl::UIntType, firrtl::SIntType>(resultTy);
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    ImplicitLocOpBuilder builder(op->getLoc(), op);
    auto result = op->getResult(0);
    auto operand = op->getOperand(OpNum);
    auto resultTy = cast<firrtl::FIRRTLBaseType>(result.getType());
    auto operandTy = cast<firrtl::FIRRTLBaseType>(operand.getType());
    auto resultWidth = resultTy.getBitWidthOrSentinel();
    auto operandWidth = operandTy.getBitWidthOrSentinel();
    Value newOp;
    if (resultWidth < operandWidth)
      newOp =
          builder.createOrFold<firrtl::BitsPrimOp>(operand, resultWidth - 1, 0);
    else if (resultWidth > operandWidth)
      newOp = builder.createOrFold<firrtl::PadPrimOp>(operand, resultWidth);
    else
      newOp = operand;
    LLVM_DEBUG(llvm::dbgs() << "Forwarding " << newOp << " in " << *op << "\n");
    result.replaceAllUsesWith(newOp);
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override {
    return ("firrtl-operand" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that replaces FIRRTL operations with a constant
/// zero of their type.
struct Constantifier : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();

    // Find valid dummy classes that we can use for anyref casts.
    anyrefCastDummy.clear();
    op.walk<WalkOrder::PreOrder>([&](CircuitOp circuitOp) {
      for (auto classOp : circuitOp.getOps<ClassOp>()) {
        if (classOp.getArguments().empty() && classOp.getBodyBlock()->empty()) {
          anyrefCastDummy.insert({circuitOp, classOp});
          anyrefCastDummyNames[circuitOp].insert(classOp.getNameAttr());
        }
      }
      return WalkResult::skip();
    });
  }

  uint64_t match(Operation *op) override {
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      Attribute attr;
      if (!matchPattern(op, m_Constant(&attr)))
        return 0;
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        if (intAttr.getValue().isZero())
          return 0;
      if (auto strAttr = dyn_cast<StringAttr>(attr))
        if (strAttr.empty())
          return 0;
      if (auto floatAttr = dyn_cast<FloatAttr>(attr))
        if (floatAttr.getValue().isZero())
          return 0;
    }
    if (auto listOp = dyn_cast<ListCreateOp>(op))
      if (listOp.getElements().empty())
        return 0;
    if (auto pathOp = dyn_cast<UnresolvedPathOp>(op))
      if (pathOp.getTarget().empty())
        return 0;

    // Don't replace anyref casts that already target a dummy class.
    if (auto anyrefCastOp = dyn_cast<ObjectAnyRefCastOp>(op)) {
      auto circuitOp = anyrefCastOp->getParentOfType<CircuitOp>();
      auto className =
          anyrefCastOp.getInput().getType().getNameAttr().getAttr();
      if (anyrefCastDummyNames[circuitOp].contains(className))
        return 0;
    }

    if (op->getNumResults() != 1)
      return 0;
    if (op->hasAttr("inner_sym"))
      return 0;
    if (isFlowSensitiveOp(op))
      return 0;
    return isa<UIntType, SIntType, StringType, FIntegerType, BoolType,
               DoubleType, ListType, PathType, AnyRefType>(
        op->getResult(0).getType());
  }

  LogicalResult rewrite(Operation *op) override {
    OpBuilder builder(op);
    auto type = op->getResult(0).getType();

    // Handle UInt/SInt types.
    if (isa<UIntType, SIntType>(type)) {
      auto width = cast<FIRRTLBaseType>(type).getBitWidthOrSentinel();
      if (width == -1)
        width = 64;
      auto newOp = ConstantOp::create(builder, op->getLoc(), type,
                                      APSInt(width, isa<UIntType>(type)));
      op->replaceAllUsesWith(newOp);
      reduce::pruneUnusedOps(op, *this);
      return success();
    }

    // Handle property string types.
    if (isa<StringType>(type)) {
      auto attr = builder.getStringAttr("");
      auto newOp = StringConstantOp::create(builder, op->getLoc(), attr);
      op->replaceAllUsesWith(newOp);
      reduce::pruneUnusedOps(op, *this);
      return success();
    }

    // Handle property integer types.
    if (isa<FIntegerType>(type)) {
      auto attr = builder.getIntegerAttr(builder.getIntegerType(64, true), 0);
      auto newOp = FIntegerConstantOp::create(builder, op->getLoc(), attr);
      op->replaceAllUsesWith(newOp);
      reduce::pruneUnusedOps(op, *this);
      return success();
    }

    // Handle property boolean types.
    if (isa<BoolType>(type)) {
      auto attr = builder.getBoolAttr(false);
      auto newOp = BoolConstantOp::create(builder, op->getLoc(), attr);
      op->replaceAllUsesWith(newOp);
      reduce::pruneUnusedOps(op, *this);
      return success();
    }

    // Handle property double types.
    if (isa<DoubleType>(type)) {
      auto attr = builder.getFloatAttr(builder.getF64Type(), 0.0);
      auto newOp = DoubleConstantOp::create(builder, op->getLoc(), attr);
      op->replaceAllUsesWith(newOp);
      reduce::pruneUnusedOps(op, *this);
      return success();
    }

    // Handle property list types.
    if (isa<ListType>(type)) {
      auto newOp =
          ListCreateOp::create(builder, op->getLoc(), type, ValueRange{});
      op->replaceAllUsesWith(newOp);
      reduce::pruneUnusedOps(op, *this);
      return success();
    }

    // Handle property path types.
    if (isa<PathType>(type)) {
      auto newOp = UnresolvedPathOp::create(builder, op->getLoc(), "");
      op->replaceAllUsesWith(newOp);
      reduce::pruneUnusedOps(op, *this);
      return success();
    }

    // Handle anyref types.
    if (isa<AnyRefType>(type)) {
      auto circuitOp = op->getParentOfType<CircuitOp>();
      auto &dummy = anyrefCastDummy[circuitOp];
      if (!dummy) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(circuitOp.getBodyBlock());
        auto &symbolTable = symbols.getNearestSymbolTable(op);
        dummy = ClassOp::create(builder, op->getLoc(), "Dummy", {}, {});
        symbolTable.insert(dummy);
        anyrefCastDummyNames[circuitOp].insert(dummy.getNameAttr());
      }
      auto objectOp = ObjectOp::create(builder, op->getLoc(), dummy, "dummy");
      auto anyrefOp =
          ObjectAnyRefCastOp::create(builder, op->getLoc(), objectOp);
      op->replaceAllUsesWith(anyrefOp);
      reduce::pruneUnusedOps(op, *this);
      return success();
    }

    return failure();
  }

  std::string getName() const override { return "firrtl-constantifier"; }
  bool acceptSizeIncrease() const override { return true; }

  ::detail::SymbolCache symbols;
  SmallDenseMap<CircuitOp, ClassOp, 2> anyrefCastDummy;
  SmallDenseMap<CircuitOp, DenseSet<StringAttr>, 2> anyrefCastDummyNames;
};

/// A sample reduction pattern that replaces the right-hand-side of
/// `firrtl.connect` and `firrtl.matchingconnect` operations with a
/// `firrtl.invalidvalue`. This removes uses from the fanin cone to these
/// connects and creates opportunities for reduction in DCE/CSE.
struct ConnectInvalidator : public Reduction {
  uint64_t match(Operation *op) override {
    if (!isa<FConnectLike>(op))
      return 0;
    if (auto *srcOp = op->getOperand(1).getDefiningOp())
      if (srcOp->hasTrait<OpTrait::ConstantLike>() ||
          isa<InvalidValueOp>(srcOp))
        return 0;
    auto type = dyn_cast<FIRRTLBaseType>(op->getOperand(1).getType());
    return type && type.isPassive();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    auto rhs = op->getOperand(1);
    OpBuilder builder(op);
    auto invOp = InvalidValueOp::create(builder, rhs.getLoc(), rhs.getType());
    auto *rhsOp = rhs.getDefiningOp();
    op->setOperand(1, invOp);
    if (rhsOp)
      reduce::pruneUnusedOps(rhsOp, *this);
    return success();
  }
  std::string getName() const override { return "connect-invalidator"; }
  bool acceptSizeIncrease() const override { return true; }
};

/// A reduction pattern that removes FIRRTL annotations from ports and
/// operations. This generates one match per annotation and port annotation,
/// allowing selective removal of individual annotations.
struct AnnotationRemover : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { nlaRemover.clear(); }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  void matches(Operation *op,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    uint64_t matchId = 0;

    // Generate matches for regular annotations
    if (auto annos = op->getAttrOfType<ArrayAttr>("annotations"))
      for (unsigned i = 0; i < annos.size(); ++i)
        addMatch(1, matchId++);

    // Generate matches for port annotations
    if (auto portAnnos = op->getAttrOfType<ArrayAttr>("portAnnotations"))
      for (auto portAnnoArray : portAnnos)
        if (auto portAnnoArrayAttr = dyn_cast<ArrayAttr>(portAnnoArray))
          for (unsigned i = 0; i < portAnnoArrayAttr.size(); ++i)
            addMatch(1, matchId++);
  }

  LogicalResult rewriteMatches(Operation *op,
                               ArrayRef<uint64_t> matches) override {
    // Convert matches to a set for fast lookup
    llvm::SmallDenseSet<uint64_t, 4> matchesSet(matches.begin(), matches.end());

    // Lambda to process annotations and filter out matched ones
    uint64_t matchId = 0;
    auto processAnnotations =
        [&](ArrayRef<Attribute> annotations) -> ArrayAttr {
      SmallVector<Attribute> newAnnotations;
      for (auto anno : annotations) {
        if (!matchesSet.contains(matchId)) {
          newAnnotations.push_back(anno);
        } else {
          // Mark NLAs in the removed annotation for cleanup
          nlaRemover.markNLAsInAnnotation(anno);
        }
        matchId++;
      }
      return ArrayAttr::get(op->getContext(), newAnnotations);
    };

    // Remove regular annotations
    if (auto annos = op->getAttrOfType<ArrayAttr>("annotations")) {
      op->setAttr("annotations", processAnnotations(annos.getValue()));
    }

    // Remove port annotations
    if (auto portAnnos = op->getAttrOfType<ArrayAttr>("portAnnotations")) {
      SmallVector<Attribute> newPortAnnos;
      for (auto portAnnoArrayAttr : portAnnos.getAsRange<ArrayAttr>()) {
        newPortAnnos.push_back(
            processAnnotations(portAnnoArrayAttr.getValue()));
      }
      op->setAttr("portAnnotations",
                  ArrayAttr::get(op->getContext(), newPortAnnos));
    }

    return success();
  }

  std::string getName() const override { return "annotation-remover"; }
  NLARemover nlaRemover;
};

/// A reduction pattern that replaces ResetType with UInt<1> across an entire
/// circuit. This walks all operations in the circuit and replaces ResetType in
/// results, block arguments, and attributes.
struct SimplifyResets : public OpReduction<CircuitOp> {
  uint64_t match(CircuitOp circuit) override {
    uint64_t numResets = 0;
    AttrTypeWalker walker;
    walker.addWalk([&](ResetType type) { ++numResets; });

    circuit.walk([&](Operation *op) {
      for (auto result : op->getResults())
        walker.walk(result.getType());

      for (auto &region : op->getRegions())
        for (auto &block : region)
          for (auto arg : block.getArguments())
            walker.walk(arg.getType());

      walker.walk(op->getAttrDictionary());
    });

    return numResets;
  }

  LogicalResult rewrite(CircuitOp circuit) override {
    auto uint1Type = UIntType::get(circuit->getContext(), 1, false);
    auto constUint1Type = UIntType::get(circuit->getContext(), 1, true);

    AttrTypeReplacer replacer;
    replacer.addReplacement([&](ResetType type) {
      return type.isConst() ? constUint1Type : uint1Type;
    });
    replacer.recursivelyReplaceElementsIn(circuit, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

    // Remove annotations related to InferResets pass
    circuit.walk([&](Operation *op) {
      // Remove operation annotations
      AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
        return anno.isClass(fullResetAnnoClass, excludeFromFullResetAnnoClass,
                            fullAsyncResetAnnoClass,
                            ignoreFullAsyncResetAnnoClass);
      });

      // Remove port annotations for module-like operations
      if (auto module = dyn_cast<FModuleLike>(op)) {
        AnnotationSet::removePortAnnotations(module, [&](unsigned portIdx,
                                                         Annotation anno) {
          return anno.isClass(fullResetAnnoClass, excludeFromFullResetAnnoClass,
                              fullAsyncResetAnnoClass,
                              ignoreFullAsyncResetAnnoClass);
        });
      }
    });

    return success();
  }

  std::string getName() const override { return "firrtl-simplify-resets"; }
  bool acceptSizeIncrease() const override { return true; }
};

/// A sample reduction pattern that removes ports from the root `firrtl.module`
/// if the port is not used or just invalidated.
struct RootPortPruner : public OpReduction<firrtl::FModuleOp> {
  uint64_t match(firrtl::FModuleOp module) override {
    auto circuit = module->getParentOfType<firrtl::CircuitOp>();
    if (!circuit)
      return 0;
    return circuit.getNameAttr() == module.getNameAttr();
  }
  LogicalResult rewrite(firrtl::FModuleOp module) override {
    assert(match(module));
    size_t numPorts = module.getNumPorts();
    llvm::BitVector dropPorts(numPorts);
    for (unsigned i = 0; i != numPorts; ++i) {
      if (onlyInvalidated(module.getArgument(i))) {
        dropPorts.set(i);
        for (auto *user :
             llvm::make_early_inc_range(module.getArgument(i).getUsers()))
          user->erase();
      }
    }
    module.erasePorts(dropPorts);
    return success();
  }
  std::string getName() const override { return "root-port-pruner"; }
};

/// A reduction pattern that removes all ports from the root `firrtl.extmodule`.
/// Since extmodules have no body, all ports can be safely removed for reduction
/// purposes.
struct RootExtmodulePortPruner : public OpReduction<firrtl::FExtModuleOp> {
  uint64_t match(firrtl::FExtModuleOp module) override {
    auto circuit = module->getParentOfType<firrtl::CircuitOp>();
    if (!circuit || circuit.getNameAttr() != module.getNameAttr())
      return 0;
    // We can remove all ports from the root extmodule
    return module.getNumPorts();
  }

  LogicalResult rewrite(firrtl::FExtModuleOp module) override {
    assert(match(module));
    size_t numPorts = module.getNumPorts();
    if (numPorts == 0)
      return failure();

    llvm::BitVector dropPorts(numPorts);
    dropPorts.set(0, numPorts);
    module.erasePorts(dropPorts);
    return success();
  }

  std::string getName() const override { return "root-extmodule-port-pruner"; }
};

/// A sample reduction pattern that replaces instances of `firrtl.extmodule`
/// with wires.
struct ExtmoduleInstanceRemover : public OpReduction<firrtl::InstanceOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(firrtl::InstanceOp instOp) override {
    return isa<firrtl::FExtModuleOp>(
        instOp.getReferencedOperation(symbols.getNearestSymbolTable(instOp)));
  }
  LogicalResult rewrite(firrtl::InstanceOp instOp) override {
    auto portInfo =
        cast<firrtl::FModuleLike>(instOp.getReferencedOperation(
                                      symbols.getNearestSymbolTable(instOp)))
            .getPorts();
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    TieOffCache tieOffCache(builder);
    SmallVector<Value> replacementWires;
    for (firrtl::PortInfo info : portInfo) {
      auto wire = firrtl::WireOp::create(
                      builder, info.type,
                      (Twine(instOp.getName()) + "_" + info.getName()).str())
                      .getResult();
      if (info.isOutput()) {
        // Tie off output ports using TieOffCache.
        if (auto baseType = dyn_cast<firrtl::FIRRTLBaseType>(info.type)) {
          auto inv = tieOffCache.getInvalid(baseType);
          firrtl::ConnectOp::create(builder, wire, inv);
        } else if (auto propType = dyn_cast<firrtl::PropertyType>(info.type)) {
          auto unknown = tieOffCache.getUnknown(propType);
          builder.create<firrtl::PropAssignOp>(wire, unknown);
        }
      }
      replacementWires.push_back(wire);
    }
    nlaRemover.markNLAsInOperation(instOp);
    instOp.replaceAllUsesWith(std::move(replacementWires));
    instOp->erase();
    return success();
  }
  std::string getName() const override { return "extmodule-instance-remover"; }
  bool acceptSizeIncrease() const override { return true; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
};

/// A reduction pattern that removes unused ports from extmodules and regular
/// modules. This is particularly useful for reducing test cases with many probe
/// ports or other unused ports.
///
/// Shared helper functions for port pruning reductions.
struct PortPrunerHelpers {
  /// Compute which ports are unused across all instances of a module.
  template <typename ModuleOpType>
  static void computeUnusedInstancePorts(ModuleOpType module,
                                         ArrayRef<Operation *> users,
                                         llvm::BitVector &portsToRemove) {
    auto ports = module.getPorts();
    for (size_t portIdx = 0; portIdx < ports.size(); ++portIdx) {
      bool portUsed = false;
      for (auto *user : users) {
        if (auto instOp = dyn_cast<firrtl::InstanceOp>(user)) {
          auto result = instOp.getResult(portIdx);
          if (!result.use_empty()) {
            portUsed = true;
            break;
          }
        }
      }
      if (!portUsed)
        portsToRemove.set(portIdx);
    }
  }

  /// Update all instances of a module to remove the specified ports.
  static void
  updateInstancesAndErasePorts(Operation *module, ArrayRef<Operation *> users,
                               const llvm::BitVector &portsToRemove) {
    // Update all instances to remove the corresponding results
    SmallVector<firrtl::InstanceOp> instancesToUpdate;
    for (auto *user : users) {
      if (auto instOp = dyn_cast<firrtl::InstanceOp>(user))
        instancesToUpdate.push_back(instOp);
    }

    for (auto instOp : instancesToUpdate) {
      auto newInst = instOp.cloneWithErasedPorts(portsToRemove);

      // Manually replace uses, skipping erased ports
      size_t newResultIdx = 0;
      for (size_t oldResultIdx = 0; oldResultIdx < instOp.getNumResults();
           ++oldResultIdx) {
        if (portsToRemove[oldResultIdx]) {
          // This port is being removed, assert it has no uses
          assert(instOp.getResult(oldResultIdx).use_empty() &&
                 "removing port with uses");
        } else {
          // Replace uses of the old result with the new result
          instOp.getResult(oldResultIdx)
              .replaceAllUsesWith(newInst.getResult(newResultIdx));
          ++newResultIdx;
        }
      }

      instOp->erase();
    }
  }
};

/// Reduction to remove unused ports from regular modules.
struct ModulePortPruner : public OpReduction<firrtl::FModuleOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
    portsToRemoveMap.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(firrtl::FModuleOp module) override {
    auto *tableOp = SymbolTable::getNearestSymbolTable(module);
    auto &userMap = symbols.getSymbolUserMap(tableOp);
    auto ports = module.getPorts();
    auto users = userMap.getUsers(module);

    // Compute which ports to remove and cache the result
    llvm::BitVector portsToRemove(ports.size());

    // If the module has no instances, aggressively remove ports that aren't
    // used within the module body itself
    if (users.empty()) {
      for (size_t portIdx = 0; portIdx < ports.size(); ++portIdx) {
        auto arg = module.getArgument(portIdx);
        if (arg.use_empty())
          portsToRemove.set(portIdx);
      }
    } else {
      // For modules with instances, check if ports are unused across all
      // instances
      PortPrunerHelpers::computeUnusedInstancePorts(module, users,
                                                    portsToRemove);
    }

    auto count = portsToRemove.count();
    if (count > 0)
      portsToRemoveMap[module] = std::move(portsToRemove);

    return count;
  }

  LogicalResult rewrite(firrtl::FModuleOp module) override {
    // Use the cached ports to remove from match()
    auto it = portsToRemoveMap.find(module);
    if (it == portsToRemoveMap.end())
      return failure();

    const auto &portsToRemove = it->second;

    // Get users for updating instances
    auto *tableOp = SymbolTable::getNearestSymbolTable(module);
    auto &userMap = symbols.getSymbolUserMap(tableOp);
    auto users = userMap.getUsers(module);

    // Update all instances
    PortPrunerHelpers::updateInstancesAndErasePorts(module, users,
                                                    portsToRemove);

    // Erase uses of port arguments within the module body
    for (size_t portIdx = 0; portIdx < module.getNumPorts(); ++portIdx) {
      if (portsToRemove[portIdx]) {
        auto arg = module.getArgument(portIdx);
        for (auto *user : llvm::make_early_inc_range(arg.getUsers()))
          user->erase();
      }
    }

    // Remove the ports from the module
    module.erasePorts(portsToRemove);

    return success();
  }

  std::string getName() const override { return "module-port-pruner"; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  DenseMap<firrtl::FModuleOp, llvm::BitVector> portsToRemoveMap;
};

/// Reduction to remove unused ports from extmodules.
struct ExtmodulePortPruner : public OpReduction<firrtl::FExtModuleOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
    portsToRemoveMap.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(firrtl::FExtModuleOp module) override {
    auto *tableOp = SymbolTable::getNearestSymbolTable(module);
    auto &userMap = symbols.getSymbolUserMap(tableOp);
    auto ports = module.getPorts();
    auto users = userMap.getUsers(module);

    // Compute which ports to remove and cache the result
    llvm::BitVector portsToRemove(ports.size());

    if (users.empty()) {
      // If the extmodule has no instances, aggressively remove all ports
      portsToRemove.set();
    } else {
      // For extmodules with instances, check if ports are unused across all
      // instances
      PortPrunerHelpers::computeUnusedInstancePorts(module, users,
                                                    portsToRemove);
    }

    auto count = portsToRemove.count();
    if (count > 0)
      portsToRemoveMap[module] = std::move(portsToRemove);

    return count;
  }

  LogicalResult rewrite(firrtl::FExtModuleOp module) override {
    // Use the cached ports to remove from match()
    auto it = portsToRemoveMap.find(module);
    if (it == portsToRemoveMap.end())
      return failure();

    const auto &portsToRemove = it->second;

    // Get users for updating instances
    auto *tableOp = SymbolTable::getNearestSymbolTable(module);
    auto &userMap = symbols.getSymbolUserMap(tableOp);
    auto users = userMap.getUsers(module);

    // Update all instances.
    PortPrunerHelpers::updateInstancesAndErasePorts(module, users,
                                                    portsToRemove);

    // Remove the ports from the module (no body to clean up for extmodules).
    module.erasePorts(portsToRemove);

    return success();
  }

  std::string getName() const override { return "extmodule-port-pruner"; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  DenseMap<firrtl::FExtModuleOp, llvm::BitVector> portsToRemoveMap;
};

/// A sample reduction pattern that pushes connected values through wires.
struct ConnectForwarder : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    domInfo = std::make_unique<DominanceInfo>(op);
  }

  uint64_t match(Operation *op) override {
    if (!isa<firrtl::FConnectLike>(op))
      return 0;
    auto dest = op->getOperand(0);
    auto src = op->getOperand(1);
    auto *destOp = dest.getDefiningOp();
    auto *srcOp = src.getDefiningOp();
    if (dest == src)
      return 0;

    // Ensure that the destination is something we should be able to forward
    // through.
    if (!isa_and_nonnull<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp>(
            destOp))
      return 0;

    // Ensure that the destination is connected to only once, and all uses of
    // the connection occur after the definition of the source.
    unsigned numConnects = 0;
    for (auto &use : dest.getUses()) {
      auto *op = use.getOwner();
      if (use.getOperandNumber() == 0 && isa<firrtl::FConnectLike>(op)) {
        if (++numConnects > 1)
          return 0;
        continue;
      }
      // Check if srcOp properly dominates op, but op is not enclosed in srcOp.
      // This handles cross-block cases (e.g., layerblocks).
      if (srcOp &&
          !domInfo->properlyDominates(srcOp, op, /*enclosingOpOk=*/false))
        return 0;
    }

    return 1;
  }

  LogicalResult rewrite(Operation *op) override {
    auto dst = op->getOperand(0);
    auto src = op->getOperand(1);
    dst.replaceAllUsesWith(src);
    op->erase();
    if (auto *dstOp = dst.getDefiningOp())
      reduce::pruneUnusedOps(dstOp, *this);
    if (auto *srcOp = src.getDefiningOp())
      reduce::pruneUnusedOps(srcOp, *this);
    return success();
  }

  std::string getName() const override { return "connect-forwarder"; }

private:
  std::unique_ptr<DominanceInfo> domInfo;
};

/// A sample reduction pattern that replaces a single-use wire and register with
/// an operand of the source value of the connection.
template <unsigned OpNum>
struct ConnectSourceOperandForwarder : public Reduction {
  uint64_t match(Operation *op) override {
    if (!isa<firrtl::ConnectOp, firrtl::MatchingConnectOp>(op))
      return 0;
    auto dest = op->getOperand(0);
    auto *destOp = dest.getDefiningOp();

    // Ensure that the destination is used only once.
    if (!destOp || !destOp->hasOneUse() ||
        !isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp>(destOp))
      return 0;

    auto *srcOp = op->getOperand(1).getDefiningOp();
    if (!srcOp || OpNum >= srcOp->getNumOperands())
      return 0;

    auto resultTy = dyn_cast<firrtl::FIRRTLBaseType>(dest.getType());
    auto opTy =
        dyn_cast<firrtl::FIRRTLBaseType>(srcOp->getOperand(OpNum).getType());

    return resultTy && opTy &&
           resultTy.getWidthlessType() == opTy.getWidthlessType() &&
           ((resultTy.getBitWidthOrSentinel() == -1) ==
            (opTy.getBitWidthOrSentinel() == -1)) &&
           isa<firrtl::UIntType, firrtl::SIntType>(resultTy);
  }

  LogicalResult rewrite(Operation *op) override {
    auto *destOp = op->getOperand(0).getDefiningOp();
    auto *srcOp = op->getOperand(1).getDefiningOp();
    auto forwardedOperand = srcOp->getOperand(OpNum);
    ImplicitLocOpBuilder builder(destOp->getLoc(), destOp);
    Value newDest;
    if (auto wire = dyn_cast<firrtl::WireOp>(destOp))
      newDest = firrtl::WireOp::create(builder, forwardedOperand.getType(),
                                       wire.getName())
                    .getResult();
    else {
      auto regName = destOp->getAttrOfType<StringAttr>("name");
      // We can promote the register into a wire but we wouldn't do here because
      // the error might be caused by the register.
      auto clock = destOp->getOperand(0);
      newDest = firrtl::RegOp::create(builder, forwardedOperand.getType(),
                                      clock, regName ? regName.str() : "")
                    .getResult();
    }

    // Create new connection between a new wire and the forwarded operand.
    builder.setInsertionPointAfter(op);
    if (isa<firrtl::ConnectOp>(op))
      firrtl::ConnectOp::create(builder, newDest, forwardedOperand);
    else
      firrtl::MatchingConnectOp::create(builder, newDest, forwardedOperand);

    // Remove the old connection and destination. We don't have to replace them
    // because destination has only one use.
    op->erase();
    destOp->erase();
    reduce::pruneUnusedOps(srcOp, *this);

    return success();
  }
  std::string getName() const override {
    return ("connect-source-operand-" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that tries to remove aggregate wires by replacing
/// all subaccesses with new independent wires. This can disentangle large
/// unused wires that are otherwise difficult to collect due to the subaccesses.
struct DetachSubaccesses : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { opsToErase.clear(); }
  void afterReduction(mlir::ModuleOp op) override {
    for (auto *op : opsToErase)
      op->dropAllReferences();
    for (auto *op : opsToErase)
      op->erase();
  }
  uint64_t match(Operation *op) override {
    // Only applies to wires and registers that are purely used in subaccess
    // operations.
    return isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp>(op) &&
           llvm::all_of(op->getUses(), [](auto &use) {
             return use.getOperandNumber() == 0 &&
                    isa<firrtl::SubfieldOp, firrtl::SubindexOp,
                        firrtl::SubaccessOp>(use.getOwner());
           });
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    OpBuilder builder(op);
    bool isWire = isa<firrtl::WireOp>(op);
    Value invalidClock;
    if (!isWire)
      invalidClock = firrtl::InvalidValueOp::create(
          builder, op->getLoc(), firrtl::ClockType::get(op->getContext()));
    for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      builder.setInsertionPoint(user);
      auto type = user->getResult(0).getType();
      Operation *replOp;
      if (isWire)
        replOp = firrtl::WireOp::create(builder, user->getLoc(), type);
      else
        replOp =
            firrtl::RegOp::create(builder, user->getLoc(), type, invalidClock);
      user->replaceAllUsesWith(replOp);
      opsToErase.insert(user);
    }
    opsToErase.insert(op);
    return success();
  }
  std::string getName() const override { return "detach-subaccesses"; }
  llvm::DenseSet<Operation *> opsToErase;
};

/// This reduction removes inner symbols on ops. Name preservation creates a lot
/// of node ops with symbols to keep name information but it also prevents
/// normal canonicalizations.
struct NodeSymbolRemover : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    innerSymUses = reduce::InnerSymbolUses(op);
  }

  uint64_t match(Operation *op) override {
    // Only match ops with an inner symbol.
    auto sym = op->getAttrOfType<hw::InnerSymAttr>("inner_sym");
    if (!sym || sym.empty())
      return 0;

    // Only match ops that have no references to their inner symbol.
    if (innerSymUses.hasInnerRef(op))
      return 0;
    return 1;
  }

  LogicalResult rewrite(Operation *op) override {
    op->removeAttr("inner_sym");
    return success();
  }

  std::string getName() const override { return "node-symbol-remover"; }
  bool acceptSizeIncrease() const override { return true; }

  reduce::InnerSymbolUses innerSymUses;
};

/// Check if inlining the referenced operation into the parent operation would
/// cause inner symbol collisions.
static bool
hasInnerSymbolCollision(Operation *referencedOp, Operation *parentOp,
                        hw::InnerSymbolTableCollection &innerSymTables) {
  // Get the inner symbol tables for both operations
  auto &targetTable = innerSymTables.getInnerSymbolTable(referencedOp);
  auto &parentTable = innerSymTables.getInnerSymbolTable(parentOp);

  // Check if any inner symbol name in the target operation already exists
  // in the parent operation. Return failure() if a collision is found to stop
  // the walk early.
  LogicalResult walkResult = targetTable.walkSymbols(
      [&](StringAttr name, const hw::InnerSymTarget &target) -> LogicalResult {
        // Check if this symbol name exists in the parent operation
        if (parentTable.lookup(name)) {
          // Collision found, return failure to stop the walk
          return failure();
        }
        return success();
      });

  // If the walk failed, it means we found a collision
  return failed(walkResult);
}

/// A sample reduction pattern that eagerly inlines instances.
struct EagerInliner : public OpReduction<InstanceOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
    nlaTables.clear();
    for (auto circuitOp : op.getOps<CircuitOp>())
      nlaTables.insert({circuitOp, std::make_unique<NLATable>(circuitOp)});
    innerSymTables = std::make_unique<hw::InnerSymbolTableCollection>();
  }
  void afterReduction(mlir::ModuleOp op) override {
    nlaRemover.remove(op);
    nlaTables.clear();
    innerSymTables.reset();
  }

  uint64_t match(InstanceOp instOp) override {
    auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto *moduleOp =
        instOp.getReferencedOperation(symbols.getSymbolTable(tableOp));

    // Only inline FModuleOp instances
    if (!isa<FModuleOp>(moduleOp))
      return 0;

    // Skip instances that participate in any NLAs
    auto circuitOp = instOp->getParentOfType<CircuitOp>();
    if (!circuitOp)
      return 0;
    auto it = nlaTables.find(circuitOp);
    if (it == nlaTables.end() || !it->second)
      return 0;
    DenseSet<hw::HierPathOp> nlas;
    it->second->getInstanceNLAs(instOp, nlas);
    if (!nlas.empty())
      return 0;

    // Check for inner symbol collisions between the referenced module and the
    // instance's parent module
    auto parentOp = instOp->getParentOfType<FModuleLike>();
    if (hasInnerSymbolCollision(moduleOp, parentOp, *innerSymTables))
      return 0;

    return 1;
  }

  LogicalResult rewrite(InstanceOp instOp) override {
    auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = cast<FModuleOp>(
        instOp.getReferencedOperation(symbols.getSymbolTable(tableOp)));
    bool isLastUse =
        (symbols.getSymbolUserMap(tableOp).getUsers(moduleOp).size() == 1);
    auto clonedModuleOp = isLastUse ? moduleOp : moduleOp.clone();

    // Create wires to replace the instance results.
    IRRewriter rewriter(instOp);
    SmallVector<Value> argWires;
    for (unsigned i = 0, e = instOp.getNumResults(); i != e; ++i) {
      auto result = instOp.getResult(i);
      auto name = rewriter.getStringAttr(Twine(instOp.getName()) + "_" +
                                         instOp.getPortName(i));
      auto wire = WireOp::create(rewriter, instOp.getLoc(), result.getType(),
                                 name, NameKindEnum::DroppableName,
                                 instOp.getPortAnnotation(i), StringAttr{})
                      .getResult();
      result.replaceAllUsesWith(wire);
      argWires.push_back(wire);
    }

    // Splice in the cloned module body.
    rewriter.inlineBlockBefore(clonedModuleOp.getBodyBlock(), instOp, argWires);

    // Make sure we remove any NLAs that go through this instance, and the
    // module if we're about the delete the module.
    nlaRemover.markNLAsInOperation(instOp);
    if (isLastUse)
      nlaRemover.markNLAsInOperation(moduleOp);

    instOp.erase();
    clonedModuleOp.erase();
    return success();
  }

  std::string getName() const override { return "firrtl-eager-inliner"; }
  bool acceptSizeIncrease() const override { return true; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  DenseMap<CircuitOp, std::unique_ptr<NLATable>> nlaTables;
  std::unique_ptr<hw::InnerSymbolTableCollection> innerSymTables;
};

/// A reduction pattern that eagerly inlines `ObjectOp`s.
struct ObjectInliner : public OpReduction<ObjectOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    blocksToSort.clear();
    symbols.clear();
    nlaRemover.clear();
    innerSymTables = std::make_unique<hw::InnerSymbolTableCollection>();
  }
  void afterReduction(mlir::ModuleOp op) override {
    for (auto *block : blocksToSort)
      mlir::sortTopologically(block);
    blocksToSort.clear();
    nlaRemover.remove(op);
    innerSymTables.reset();
  }

  uint64_t match(ObjectOp objOp) override {
    auto *tableOp = SymbolTable::getNearestSymbolTable(objOp);
    auto *classOp =
        objOp.getReferencedOperation(symbols.getSymbolTable(tableOp));

    // Only inline `ClassOp`s.
    if (!isa<ClassOp>(classOp))
      return 0;

    // Check for inner symbol collisions between the referenced class and the
    // object's parent module.
    auto parentOp = objOp->getParentOfType<FModuleLike>();
    if (hasInnerSymbolCollision(classOp, parentOp, *innerSymTables))
      return 0;

    // Verify all uses are ObjectSubfieldOp.
    for (auto *user : objOp.getResult().getUsers())
      if (!isa<ObjectSubfieldOp>(user))
        return 0;

    return 1;
  }

  LogicalResult rewrite(ObjectOp objOp) override {
    auto *tableOp = SymbolTable::getNearestSymbolTable(objOp);
    auto classOp = cast<ClassOp>(
        objOp.getReferencedOperation(symbols.getSymbolTable(tableOp)));
    auto clonedClassOp = classOp.clone();

    // Create wires to replace the ObjectSubfieldOp results.
    IRRewriter rewriter(objOp);
    SmallVector<Value> portWires;
    auto classType = objOp.getType();

    // Create a wire for each port in the class
    for (unsigned i = 0, e = classType.getNumElements(); i != e; ++i) {
      auto element = classType.getElement(i);
      auto name = rewriter.getStringAttr(Twine(objOp.getName()) + "_" +
                                         element.name.getValue());
      auto wire = WireOp::create(rewriter, objOp.getLoc(), element.type, name,
                                 NameKindEnum::DroppableName,
                                 rewriter.getArrayAttr({}), StringAttr{})
                      .getResult();
      portWires.push_back(wire);
    }

    // Replace all ObjectSubfieldOp uses with corresponding wires
    SmallVector<ObjectSubfieldOp> subfieldOps;
    for (auto *user : objOp.getResult().getUsers()) {
      auto subfieldOp = cast<ObjectSubfieldOp>(user);
      subfieldOps.push_back(subfieldOp);
      auto index = subfieldOp.getIndex();
      subfieldOp.getResult().replaceAllUsesWith(portWires[index]);
    }

    // Splice in the cloned class body.
    rewriter.inlineBlockBefore(clonedClassOp.getBodyBlock(), objOp, portWires);

    // After inlining the class body, we need to eliminate `WireOps` since
    // `ClassOps` cannot contain wires. For each port wire, find its single
    // connect, remove it, and replace all uses of the wire with the assigned
    // value.
    SmallVector<FConnectLike> connectsToErase;
    for (auto portWire : portWires) {
      // Find a single value to replace the wire with, and collect all connects
      // to the wire such that we can erase them later.
      Value value;
      for (auto *user : portWire.getUsers()) {
        if (auto connect = dyn_cast<FConnectLike>(user)) {
          if (connect.getDest() == portWire) {
            value = connect.getSrc();
            connectsToErase.push_back(connect);
          }
        }
      }

      // Be very conservative about deleting these wires. Other reductions may
      // leave class ports unconnected, which means that there isn't always a
      // clean replacement available here. Better to just leave the wires in the
      // IR and let the verifier fail later.
      if (value)
        portWire.replaceAllUsesWith(value);
      for (auto connect : connectsToErase)
        connect.erase();
      if (portWire.use_empty())
        portWire.getDefiningOp()->erase();
      connectsToErase.clear();
    }

    // Make sure we remove any NLAs that go through this object.
    nlaRemover.markNLAsInOperation(objOp);

    // Since the above forwarding of SSA values through wires can create
    // dominance issues, mark the region containing the object to be sorted
    // topologically.
    blocksToSort.insert(objOp->getBlock());

    // Erase the object and cloned class.
    for (auto subfieldOp : subfieldOps)
      subfieldOp.erase();
    objOp.erase();
    clonedClassOp.erase();
    return success();
  }

  std::string getName() const override { return "firrtl-object-inliner"; }
  bool acceptSizeIncrease() const override { return true; }

  SetVector<Block *> blocksToSort;
  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  std::unique_ptr<hw::InnerSymbolTableCollection> innerSymTables;
};

/// Psuedo-reduction that sanitizes the names of things inside modules.  This is
/// not an actual reduction, but often removes extraneous information that has
/// no bearing on the actual reduction (and would likely be removed before
/// sharing the reduction).  This makes the following changes:
///
///   - All wires are renamed to "wire"
///   - All registers are renamed to "reg"
///   - All nodes are renamed to "node"
///   - All memories are renamed to "mem"
///   - All verification messages and labels are dropped
///
struct ModuleInternalNameSanitizer : public Reduction {
  uint64_t match(Operation *op) override {
    // Only match operations with names.
    return isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp,
               firrtl::NodeOp, firrtl::MemOp, chirrtl::CombMemOp,
               chirrtl::SeqMemOp, firrtl::AssertOp, firrtl::AssumeOp,
               firrtl::CoverOp>(op);
  }
  LogicalResult rewrite(Operation *op) override {
    TypeSwitch<Operation *, void>(op)
        .Case<firrtl::WireOp>([](auto op) { op.setName("wire"); })
        .Case<firrtl::RegOp, firrtl::RegResetOp>(
            [](auto op) { op.setName("reg"); })
        .Case<firrtl::NodeOp>([](auto op) { op.setName("node"); })
        .Case<firrtl::MemOp, chirrtl::CombMemOp, chirrtl::SeqMemOp>(
            [](auto op) { op.setName("mem"); })
        .Case<firrtl::AssertOp, firrtl::AssumeOp, firrtl::CoverOp>([](auto op) {
          op->setAttr("message", StringAttr::get(op.getContext(), ""));
          op->setAttr("name", StringAttr::get(op.getContext(), ""));
        });
    return success();
  }

  std::string getName() const override {
    return "module-internal-name-sanitizer";
  }

  bool acceptSizeIncrease() const override { return true; }

  bool isOneShot() const override { return true; }
};

/// Psuedo-reduction that sanitizes module, instance, and port names.  This
/// makes the following changes:
///
///     - All modules are given metasyntactic names ("Foo", "Bar", etc.)
///     - All instances are renamed to match the new module name
///     - All module ports are renamed in the following way:
///         - All clocks are reanemd to "clk"
///         - All resets are renamed to "rst"
///         - All references are renamed to "ref"
///         - Anything else is renamed to "port"
///
struct ModuleNameSanitizer : OpReduction<firrtl::CircuitOp> {

  const char *names[48] = {
      "Foo",    "Bar",    "Baz",    "Qux",      "Quux",   "Quuux",  "Quuuux",
      "Quz",    "Corge",  "Grault", "Bazola",   "Ztesch", "Thud",   "Grunt",
      "Bletch", "Fum",    "Fred",   "Jim",      "Sheila", "Barney", "Flarp",
      "Zxc",    "Spqr",   "Wombat", "Shme",     "Bongo",  "Spam",   "Eggs",
      "Snork",  "Zot",    "Blarg",  "Wibble",   "Toto",   "Titi",   "Tata",
      "Tutu",   "Pippo",  "Pluto",  "Paperino", "Aap",    "Noot",   "Mies",
      "Oogle",  "Foogle", "Boogle", "Zork",     "Gork",   "Bork"};

  size_t nameIndex = 0;

  const char *getName() {
    if (nameIndex >= 48)
      nameIndex = 0;
    return names[nameIndex++];
  };

  size_t portNameIndex = 0;

  char getPortName() {
    if (portNameIndex >= 26)
      portNameIndex = 0;
    return 'a' + portNameIndex++;
  }

  void beforeReduction(mlir::ModuleOp op) override { nameIndex = 0; }

  LogicalResult rewrite(firrtl::CircuitOp circuitOp) override {

    firrtl::InstanceGraph iGraph(circuitOp);

    auto *circuitName = getName();
    iGraph.getTopLevelModule().setName(circuitName);
    circuitOp.setName(circuitName);

    for (auto *node : iGraph) {
      auto module = node->getModule<firrtl::FModuleLike>();

      bool shouldReplacePorts = false;
      SmallVector<Attribute> newNames;
      if (auto fmodule = dyn_cast<firrtl::FModuleOp>(*module)) {
        portNameIndex = 0;
        // TODO: The namespace should be unnecessary. However, some FIRRTL
        // passes expect that port names are unique.
        circt::Namespace ns;
        auto oldPorts = fmodule.getPorts();
        shouldReplacePorts = !oldPorts.empty();
        for (unsigned i = 0, e = fmodule.getNumPorts(); i != e; ++i) {
          auto port = oldPorts[i];
          auto newName = firrtl::FIRRTLTypeSwitch<Type, StringRef>(port.type)
                             .Case<firrtl::ClockType>(
                                 [&](auto a) { return ns.newName("clk"); })
                             .Case<firrtl::ResetType, firrtl::AsyncResetType>(
                                 [&](auto a) { return ns.newName("rst"); })
                             .Case<firrtl::RefType>(
                                 [&](auto a) { return ns.newName("ref"); })
                             .Default([&](auto a) {
                               return ns.newName(Twine(getPortName()));
                             });
          newNames.push_back(StringAttr::get(circuitOp.getContext(), newName));
        }
        fmodule->setAttr("portNames",
                         ArrayAttr::get(fmodule.getContext(), newNames));
      }

      if (module == iGraph.getTopLevelModule())
        continue;
      auto newName = StringAttr::get(circuitOp.getContext(), getName());
      module.setName(newName);
      for (auto *use : node->uses()) {
        auto useOp = use->getInstance();
        if (auto instanceOp = dyn_cast<firrtl::InstanceOp>(*useOp)) {
          instanceOp.setModuleName(newName);
          instanceOp.setName(newName);
          if (shouldReplacePorts)
            instanceOp.setPortNamesAttr(
                ArrayAttr::get(circuitOp.getContext(), newNames));
        } else if (auto objectOp = dyn_cast<firrtl::ObjectOp>(*useOp)) {
          // ObjectOp stores the class name in its result type, so we need to
          // create a new ClassType with the new name and set it on the result.
          auto oldClassType = objectOp.getType();
          auto newClassType = firrtl::ClassType::get(
              circuitOp.getContext(), FlatSymbolRefAttr::get(newName),
              oldClassType.getElements());
          objectOp.getResult().setType(newClassType);
          objectOp.setName(newName);
        }
      }
    }

    return success();
  }

  std::string getName() const override { return "module-name-sanitizer"; }

  bool acceptSizeIncrease() const override { return true; }

  bool isOneShot() const override { return true; }
};

/// A reduction pattern that groups modules by their port signature (types and
/// directions) and replaces instances with the smallest module in each group.
/// This helps reduce the IR by consolidating functionally equivalent modules
/// based on their interface.
///
/// The pattern works by:
/// 1. Grouping all modules by their port signature (port types and directions)
/// 2. For each group with multiple modules, finding the smallest module using
///    the module size cache
/// 3. Replacing all instances of larger modules with instances of the smallest
///    module in the same group
/// 4. Removing the larger modules from the circuit
///
/// This reduction is useful for reducing circuits where multiple modules have
/// the same interface but different implementations, allowing the reducer to
/// try the smallest implementation first.
struct ModuleSwapper : public OpReduction<InstanceOp> {
  // Per-circuit state containing all the information needed for module swapping
  using PortSignature = SmallVector<std::pair<Type, Direction>>;
  struct CircuitState {
    DenseMap<PortSignature, SmallVector<FModuleLike, 4>> moduleTypeGroups;
    DenseMap<StringAttr, FModuleLike> instanceToCanonicalModule;
    std::unique_ptr<NLATable> nlaTable;
  };

  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
    moduleSizes.clear();
    circuitStates.clear();

    // Collect module type groups and NLA tables for all circuits up front
    op.walk<WalkOrder::PreOrder>([&](CircuitOp circuitOp) {
      auto &state = circuitStates[circuitOp];
      state.nlaTable = std::make_unique<NLATable>(circuitOp);
      buildModuleTypeGroups(circuitOp, state);
      return WalkResult::skip();
    });
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  /// Create a vector of port type-direction pairs for the given FIRRTL module.
  /// This ignores port names, allowing modules with the same port types and
  /// directions but different port names to be considered equivalent for
  /// swapping.
  PortSignature getModulePortSignature(FModuleLike module) {
    PortSignature signature;
    signature.reserve(module.getNumPorts());
    for (unsigned i = 0, e = module.getNumPorts(); i < e; ++i)
      signature.emplace_back(module.getPortType(i), module.getPortDirection(i));
    return signature;
  }

  /// Group modules by their port signature and find the smallest in each group.
  void buildModuleTypeGroups(CircuitOp circuitOp, CircuitState &state) {
    // Group modules by their port signature
    for (auto module : circuitOp.getBodyBlock()->getOps<FModuleLike>()) {
      auto signature = getModulePortSignature(module);
      state.moduleTypeGroups[signature].push_back(module);
    }

    // For each group, find the smallest module
    for (auto &[signature, modules] : state.moduleTypeGroups) {
      if (modules.size() <= 1)
        continue;

      FModuleLike smallestModule = nullptr;
      uint64_t smallestSize = std::numeric_limits<uint64_t>::max();

      for (auto module : modules) {
        uint64_t size = moduleSizes.getModuleSize(module, symbols);
        if (size < smallestSize) {
          smallestSize = size;
          smallestModule = module;
        }
      }

      // Map all modules in this group to the smallest one
      for (auto module : modules) {
        if (module != smallestModule) {
          state.instanceToCanonicalModule[module.getModuleNameAttr()] =
              smallestModule;
        }
      }
    }
  }

  uint64_t match(InstanceOp instOp) override {
    // Get the circuit this instance belongs to
    auto circuitOp = instOp->getParentOfType<CircuitOp>();
    assert(circuitOp);
    const auto &state = circuitStates.at(circuitOp);

    // Skip instances that participate in any NLAs
    DenseSet<hw::HierPathOp> nlas;
    state.nlaTable->getInstanceNLAs(instOp, nlas);
    if (!nlas.empty())
      return 0;

    // Check if this instance can be redirected to a smaller module
    auto moduleName = instOp.getModuleNameAttr().getAttr();
    auto canonicalModule = state.instanceToCanonicalModule.lookup(moduleName);
    if (!canonicalModule)
      return 0;

    // Benefit is the size difference
    auto currentModule = cast<FModuleLike>(
        instOp.getReferencedOperation(symbols.getNearestSymbolTable(instOp)));
    uint64_t currentSize = moduleSizes.getModuleSize(currentModule, symbols);
    uint64_t canonicalSize =
        moduleSizes.getModuleSize(canonicalModule, symbols);
    return currentSize > canonicalSize ? currentSize - canonicalSize : 1;
  }

  LogicalResult rewrite(InstanceOp instOp) override {
    // Get the circuit this instance belongs to
    auto circuitOp = instOp->getParentOfType<CircuitOp>();
    assert(circuitOp);
    const auto &state = circuitStates.at(circuitOp);

    // Replace the instantiated module with the canonical module.
    auto canonicalModule = state.instanceToCanonicalModule.at(
        instOp.getModuleNameAttr().getAttr());
    auto canonicalName = canonicalModule.getModuleNameAttr();
    instOp.setModuleNameAttr(FlatSymbolRefAttr::get(canonicalName));

    // Update port names to match the canonical module
    instOp.setPortNamesAttr(canonicalModule.getPortNamesAttr());

    return success();
  }

  std::string getName() const override { return "firrtl-module-swapper"; }
  bool acceptSizeIncrease() const override { return true; }

private:
  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  ModuleSizeCache moduleSizes;

  // Per-circuit state containing all module swapping information
  DenseMap<CircuitOp, CircuitState> circuitStates;
};

/// A reduction pattern that handles MustDedup annotations by replacing all
/// module names in a dedup group with a single module name. This helps reduce
/// the IR by consolidating module references that are required to be identical.
///
/// The pattern works by:
/// 1. Finding all MustDeduplicateAnnotation annotations on the circuit
/// 2. For each dedup group, using the first module as the canonical name
/// 3. Replacing all instance references to other modules in the group with
///    references to the canonical module
/// 4. Removing the non-canonical modules from the circuit
/// 5. Removing the processed MustDedup annotation
///
/// This reduction is particularly useful for reducing large circuits where
/// multiple modules are known to be identical but haven't been deduplicated
/// yet.
struct ForceDedup : public OpReduction<CircuitOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
    modulesToErase.clear();
    moduleSizes.clear();
  }
  void afterReduction(mlir::ModuleOp op) override {
    nlaRemover.remove(op);
    for (auto mod : modulesToErase)
      mod->erase();
  }

  /// Collect all MustDedup annotations and create matches for each dedup group.
  void matches(CircuitOp circuitOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    auto &symbolTable = symbols.getNearestSymbolTable(circuitOp);
    auto annotations = AnnotationSet(circuitOp);
    for (auto [annoIdx, anno] : llvm::enumerate(annotations)) {
      if (!anno.isClass(mustDeduplicateAnnoClass))
        continue;

      auto modulesAttr = anno.getMember<ArrayAttr>("modules");
      if (!modulesAttr || modulesAttr.size() < 2)
        continue;

      // Check that all modules have the same port signature. Malformed inputs
      // may have modules listed in a MustDedup annotation that have distinct
      // port types.
      uint64_t totalSize = 0;
      ArrayAttr portTypes;
      DenseBoolArrayAttr portDirections;
      bool allSame = true;
      for (auto moduleName : modulesAttr.getAsRange<StringAttr>()) {
        auto target = tokenizePath(moduleName);
        if (!target) {
          allSame = false;
          break;
        }
        auto mod = symbolTable.lookup<FModuleLike>(target->module);
        if (!mod) {
          allSame = false;
          break;
        }
        totalSize += moduleSizes.getModuleSize(mod, symbols);
        if (!portTypes) {
          portTypes = mod.getPortTypesAttr();
          portDirections = mod.getPortDirectionsAttr();
        } else if (portTypes != mod.getPortTypesAttr() ||
                   portDirections != mod.getPortDirectionsAttr()) {
          allSame = false;
          break;
        }
      }
      if (!allSame)
        continue;

      // Each dedup group gets its own match with benefit proportional to group
      // size.
      addMatch(totalSize, annoIdx);
    }
  }

  LogicalResult rewriteMatches(CircuitOp circuitOp,
                               ArrayRef<uint64_t> matches) override {
    auto *context = circuitOp->getContext();
    NLATable nlaTable(circuitOp);
    hw::InnerSymbolTableCollection innerSymTables;
    auto annotations = AnnotationSet(circuitOp);
    SmallVector<Annotation> newAnnotations;

    for (auto [annoIdx, anno] : llvm::enumerate(annotations)) {
      // Check if this annotation was selected.
      if (!llvm::is_contained(matches, annoIdx)) {
        newAnnotations.push_back(anno);
        continue;
      }
      auto modulesAttr = anno.getMember<ArrayAttr>("modules");
      assert(anno.isClass(mustDeduplicateAnnoClass) && modulesAttr &&
             modulesAttr.size() >= 2);

      // Extract module names from the dedup group.
      SmallVector<StringAttr> moduleNames;
      for (auto moduleRef : modulesAttr.getAsRange<StringAttr>()) {
        // Parse "~CircuitName|ModuleName" format.
        auto refStr = moduleRef.getValue();
        auto pipePos = refStr.find('|');
        if (pipePos != StringRef::npos && pipePos + 1 < refStr.size()) {
          auto moduleName = refStr.substr(pipePos + 1);
          moduleNames.push_back(StringAttr::get(context, moduleName));
        }
      }

      // Simply drop the annotation if there's only one module.
      if (moduleNames.size() < 2)
        continue;

      // Replace all instances and references to other modules with the
      // first module.
      replaceModuleReferences(circuitOp, moduleNames, nlaTable, innerSymTables);
      nlaRemover.markNLAsInAnnotation(anno.getAttr());
    }
    if (newAnnotations.size() == annotations.size())
      return failure();

    // Update circuit annotations.
    AnnotationSet newAnnoSet(newAnnotations, context);
    newAnnoSet.applyToOperation(circuitOp);
    return success();
  }

  std::string getName() const override { return "firrtl-force-dedup"; }
  bool acceptSizeIncrease() const override { return true; }

private:
  /// Replace all references to modules in the dedup group with the canonical
  /// module name
  void replaceModuleReferences(CircuitOp circuitOp,
                               ArrayRef<StringAttr> moduleNames,
                               NLATable &nlaTable,
                               hw::InnerSymbolTableCollection &innerSymTables) {
    auto *tableOp = SymbolTable::getNearestSymbolTable(circuitOp);
    auto &symbolTable = symbols.getSymbolTable(tableOp);
    auto &symbolUserMap = symbols.getSymbolUserMap(tableOp);
    auto *context = circuitOp->getContext();
    auto innerRefs = hw::InnerRefNamespace{symbolTable, innerSymTables};

    // Collect the modules.
    FModuleLike canonicalModule;
    SmallVector<FModuleLike> modulesToReplace;
    for (auto name : moduleNames) {
      if (auto mod = symbolTable.lookup<FModuleLike>(name)) {
        if (!canonicalModule)
          canonicalModule = mod;
        else
          modulesToReplace.push_back(mod);
      }
    }
    if (modulesToReplace.empty())
      return;

    // Replace all instance references.
    auto canonicalName = canonicalModule.getModuleNameAttr();
    auto canonicalRef = FlatSymbolRefAttr::get(canonicalName);
    for (auto moduleName : moduleNames) {
      if (moduleName == canonicalName)
        continue;
      auto *symbolOp = symbolTable.lookup(moduleName);
      if (!symbolOp)
        continue;
      for (auto *user : symbolUserMap.getUsers(symbolOp)) {
        auto instOp = dyn_cast<InstanceOp>(user);
        if (!instOp || instOp.getModuleNameAttr().getAttr() != moduleName)
          continue;
        instOp.setModuleNameAttr(canonicalRef);
        instOp.setPortNamesAttr(canonicalModule.getPortNamesAttr());
      }
    }

    // Update NLAs to reference the canonical module instead of modules being
    // removed using NLATable for better performance.
    for (auto oldMod : modulesToReplace) {
      SmallVector<hw::HierPathOp> nlaOps(
          nlaTable.lookup(oldMod.getModuleNameAttr()));
      for (auto nlaOp : nlaOps) {
        nlaTable.erase(nlaOp);
        StringAttr oldModName = oldMod.getModuleNameAttr();
        StringAttr newModName = canonicalName;
        SmallVector<Attribute, 4> newPath;
        for (auto nameRef : nlaOp.getNamepath()) {
          if (auto ref = dyn_cast<hw::InnerRefAttr>(nameRef)) {
            if (ref.getModule() == oldModName) {
              auto oldInst = innerRefs.lookupOp<FInstanceLike>(ref);
              ref = hw::InnerRefAttr::get(newModName, ref.getName());
              auto newInst = innerRefs.lookupOp<FInstanceLike>(ref);
              if (oldInst && newInst) {
                // Get the first module name from the list (for
                // InstanceOp/ObjectOp, there's only one)
                auto oldModNames = oldInst.getReferencedModuleNamesAttr();
                auto newModNames = newInst.getReferencedModuleNamesAttr();
                if (!oldModNames.empty() && !newModNames.empty()) {
                  oldModName = cast<StringAttr>(oldModNames[0]);
                  newModName = cast<StringAttr>(newModNames[0]);
                }
              }
            }
            newPath.push_back(ref);
          } else if (cast<FlatSymbolRefAttr>(nameRef).getAttr() == oldModName) {
            newPath.push_back(FlatSymbolRefAttr::get(newModName));
          } else {
            newPath.push_back(nameRef);
          }
        }
        nlaOp.setNamepathAttr(ArrayAttr::get(context, newPath));
        nlaTable.addNLA(nlaOp);
      }
    }

    // Mark NLAs in modules to be removed.
    for (auto module : modulesToReplace) {
      nlaRemover.markNLAsInOperation(module);
      modulesToErase.insert(module);
    }
  }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  SetVector<FModuleLike> modulesToErase;
  ModuleSizeCache moduleSizes;
};

/// A reduction pattern that moves `MustDedup` annotations from a module onto
/// its child modules. This pattern iterates over all MustDedup annotations,
/// collects all `FInstanceLike` ops in each module of the dedup group, and
/// creates new MustDedup annotations for corresponding instances across the
/// modules. Each set of corresponding instances becomes a separate match of the
/// reduction. The reduction also removes the original MustDedup annotation on
/// the parent module.
///
/// The pattern works by:
/// 1. Finding all MustDeduplicateAnnotation annotations on the circuit
/// 2. For each dedup group, collecting all FInstanceLike operations in each
/// module
/// 3. Grouping corresponding instances across modules by their position/name
/// 4. Creating new MustDedup annotations for each group of corresponding
/// instances
/// 5. Removing the original MustDedup annotation from the circuit
struct MustDedupChildren : public OpReduction<CircuitOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  /// Collect all MustDedup annotations and create matches for each instance
  /// group.
  void matches(CircuitOp circuitOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    auto annotations = AnnotationSet(circuitOp);
    uint64_t matchId = 0;

    DenseSet<StringRef> modulesAlreadyInMustDedup;
    for (auto [annoIdx, anno] : llvm::enumerate(annotations))
      if (anno.isClass(mustDeduplicateAnnoClass))
        if (auto modulesAttr = anno.getMember<ArrayAttr>("modules"))
          for (auto moduleRef : modulesAttr.getAsRange<StringAttr>())
            if (auto target = tokenizePath(moduleRef))
              modulesAlreadyInMustDedup.insert(target->module);

    for (auto [annoIdx, anno] : llvm::enumerate(annotations)) {
      if (!anno.isClass(mustDeduplicateAnnoClass))
        continue;

      auto modulesAttr = anno.getMember<ArrayAttr>("modules");
      if (!modulesAttr || modulesAttr.size() < 2)
        continue;

      // Process each group of corresponding instances
      processInstanceGroups(
          circuitOp, modulesAttr, [&](ArrayRef<FInstanceLike> instanceGroup) {
            matchId++;

            // Make sure there are at least two distinct modules.
            SmallDenseSet<StringAttr, 4> moduleTargets;
            for (auto instOp : instanceGroup) {
              auto moduleNames = instOp.getReferencedModuleNamesAttr();
              for (auto moduleName : moduleNames)
                moduleTargets.insert(cast<StringAttr>(moduleName));
            }
            if (moduleTargets.size() < 2)
              return;

            // Make sure none of the modules are not yet in a must dedup
            // annotation.
            if (llvm::any_of(instanceGroup, [&](FInstanceLike inst) {
                  auto moduleNames = inst.getReferencedModuleNames();
                  return llvm::any_of(moduleNames, [&](StringRef moduleName) {
                    return modulesAlreadyInMustDedup.contains(moduleName);
                  });
                }))
              return;

            addMatch(1, matchId - 1);
          });
    }
  }

  LogicalResult rewriteMatches(CircuitOp circuitOp,
                               ArrayRef<uint64_t> matches) override {
    auto *context = circuitOp->getContext();
    auto annotations = AnnotationSet(circuitOp);
    SmallVector<Annotation> newAnnotations;
    uint64_t matchId = 0;

    for (auto [annoIdx, anno] : llvm::enumerate(annotations)) {
      if (!anno.isClass(mustDeduplicateAnnoClass)) {
        newAnnotations.push_back(anno);
        continue;
      }

      auto modulesAttr = anno.getMember<ArrayAttr>("modules");
      if (!modulesAttr || modulesAttr.size() < 2) {
        newAnnotations.push_back(anno);
        continue;
      }

      processInstanceGroups(
          circuitOp, modulesAttr, [&](ArrayRef<FInstanceLike> instanceGroup) {
            // Check if this instance group was selected
            if (!llvm::is_contained(matches, matchId++))
              return;

            // Create the list of modules to put into this new annotation.
            SmallSetVector<StringAttr, 4> moduleTargets;
            for (auto instOp : instanceGroup) {
              auto moduleNames = instOp.getReferencedModuleNames();
              for (auto moduleName : moduleNames) {
                auto target = TokenAnnoTarget();
                target.circuit = circuitOp.getName();
                target.module = moduleName;
                moduleTargets.insert(target.toStringAttr(context));
              }
            }

            // Create a new MustDedup annotation for this list of modules.
            SmallVector<NamedAttribute> newAnnoAttrs;
            newAnnoAttrs.emplace_back(
                StringAttr::get(context, "class"),
                StringAttr::get(context, mustDeduplicateAnnoClass));
            newAnnoAttrs.emplace_back(
                StringAttr::get(context, "modules"),
                ArrayAttr::get(context,
                               SmallVector<Attribute>(moduleTargets.begin(),
                                                      moduleTargets.end())));

            auto newAnnoDict = DictionaryAttr::get(context, newAnnoAttrs);
            newAnnotations.emplace_back(newAnnoDict);
          });

      // Keep the original annotation around.
      newAnnotations.push_back(anno);
    }

    // Update circuit annotations
    AnnotationSet newAnnoSet(newAnnotations, context);
    newAnnoSet.applyToOperation(circuitOp);
    return success();
  }

  std::string getName() const override { return "must-dedup-children"; }
  bool acceptSizeIncrease() const override { return true; }

private:
  /// Helper function to process groups of corresponding instances from a
  /// MustDedup annotation. Calls the provided lambda for each group of
  /// corresponding instances across the modules. Only calls the lambda if there
  /// are at least 2 modules.
  void processInstanceGroups(
      CircuitOp circuitOp, ArrayAttr modulesAttr,
      llvm::function_ref<void(ArrayRef<FInstanceLike>)> callback) {
    auto &symbolTable = symbols.getSymbolTable(circuitOp);

    // Extract module names and get the actual modules
    SmallVector<FModuleLike> modules;
    for (auto moduleRef : modulesAttr.getAsRange<StringAttr>())
      if (auto target = tokenizePath(moduleRef))
        if (auto mod = symbolTable.lookup<FModuleLike>(target->module))
          modules.push_back(mod);

    // Need at least 2 modules for deduplication
    if (modules.size() < 2)
      return;

    // Collect all FInstanceLike operations from each module and group them by
    // name. Instance names are a good key for matching instances across
    // modules. But they may not be unique, so we need to be careful to only
    // match up instances that are uniquely named within every module.
    struct InstanceGroup {
      SmallVector<FInstanceLike> instances;
      bool nameIsUnique = true;
    };
    MapVector<StringAttr, InstanceGroup> instanceGroups;
    for (auto module : modules) {
      SmallDenseMap<StringAttr, unsigned> nameCounts;
      module.walk([&](FInstanceLike instOp) {
        if (isa<ObjectOp>(instOp.getOperation()))
          return;
        auto name = instOp.getInstanceNameAttr();
        auto &group = instanceGroups[name];
        if (nameCounts[name]++ > 1)
          group.nameIsUnique = false;
        group.instances.push_back(instOp);
      });
    }

    // Call the callback for each group of instances that are uniquely named and
    // consist of at least 2 instances.
    for (auto &[name, group] : instanceGroups)
      if (group.nameIsUnique && group.instances.size() >= 2)
        callback(group.instances);
  }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
};

struct LayerDisable : public OpReduction<CircuitOp> {
  LayerDisable(MLIRContext *context) {
    pm = std::make_unique<mlir::PassManager>(
        context, "builtin.module", mlir::OpPassManager::Nesting::Explicit);
    pm->nest<firrtl::CircuitOp>().addPass(firrtl::createSpecializeLayers());
  };

  void beforeReduction(mlir::ModuleOp op) override { symbolRefAttrMap.clear(); }

  void afterReduction(mlir::ModuleOp op) override { (void)pm->run(op); };

  void matches(CircuitOp circuitOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    uint64_t matchId = 0;

    SmallVector<FlatSymbolRefAttr> nestedRefs;
    std::function<void(StringAttr, LayerOp)> addLayer = [&](StringAttr rootRef,
                                                            LayerOp layerOp) {
      if (!rootRef)
        rootRef = layerOp.getSymNameAttr();
      else
        nestedRefs.push_back(FlatSymbolRefAttr::get(layerOp));

      symbolRefAttrMap[matchId] = SymbolRefAttr::get(rootRef, nestedRefs);
      addMatch(1, matchId++);

      for (auto nestedLayerOp : layerOp.getOps<LayerOp>())
        addLayer(rootRef, nestedLayerOp);

      if (!nestedRefs.empty())
        nestedRefs.pop_back();
    };

    for (auto layerOp : circuitOp.getOps<LayerOp>())
      addLayer({}, layerOp);
  }

  LogicalResult rewriteMatches(CircuitOp circuitOp,
                               ArrayRef<uint64_t> matches) override {
    SmallVector<Attribute> disableLayers;
    if (auto existingDisables = circuitOp.getDisableLayersAttr()) {
      auto disableRange = existingDisables.getAsRange<Attribute>();
      disableLayers.append(disableRange.begin(), disableRange.end());
    }
    for (auto match : matches)
      disableLayers.push_back(symbolRefAttrMap.at(match));

    circuitOp.setDisableLayersAttr(
        ArrayAttr::get(circuitOp.getContext(), disableLayers));

    return success();
  }

  std::string getName() const override { return "firrtl-layer-disable"; }

  std::unique_ptr<mlir::PassManager> pm;
  DenseMap<uint64_t, SymbolRefAttr> symbolRefAttrMap;
};

} // namespace

/// A reduction pattern that removes elements from FIRRTL list create
/// operations. This generates one match per element in each list, allowing
/// selective removal of individual elements.
struct ListCreateElementRemover : public OpReduction<ListCreateOp> {
  void matches(ListCreateOp listOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    // Create one match for each element in the list
    auto elements = listOp.getElements();
    for (size_t i = 0; i < elements.size(); ++i)
      addMatch(1, i);
  }

  LogicalResult rewriteMatches(ListCreateOp listOp,
                               ArrayRef<uint64_t> matches) override {
    // Convert matches to a set for fast lookup
    llvm::SmallDenseSet<uint64_t, 4> matchesSet(matches.begin(), matches.end());

    // Collect elements that should be kept (not in matches)
    SmallVector<Value> newElements;
    auto elements = listOp.getElements();
    for (size_t i = 0; i < elements.size(); ++i) {
      if (!matchesSet.contains(i))
        newElements.push_back(elements[i]);
    }

    // Create a new list with the remaining elements
    OpBuilder builder(listOp);
    auto newListOp = ListCreateOp::create(builder, listOp.getLoc(),
                                          listOp.getType(), newElements);
    listOp.getResult().replaceAllUsesWith(newListOp.getResult());
    listOp.erase();

    return success();
  }

  std::string getName() const override {
    return "firrtl-list-create-element-remover";
  }
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

void firrtl::FIRRTLReducePatternDialectInterface::populateReducePatterns(
    circt::ReducePatternSet &patterns) const {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // assigned reasonable benefit indicators (higher benefit patterns are
  // prioritized). For example, things that can knock out entire modules while
  // being cheap should be tried first (and thus have higher benefit), before
  // trying to tweak operands of individual arithmetic ops.
  patterns.add<SimplifyResets, 35>();
  patterns.add<ForceDedup, 34>();
  patterns.add<MustDedupChildren, 33>();
  patterns.add<AnnotationRemover, 32>();
  patterns.add<ModuleSwapper, 31>();
  patterns.add<LayerDisable, 30>(getContext());
  patterns.add<PassReduction, 29>(
      getContext(),
      firrtl::createDropName({/*preserveMode=*/PreserveValues::None}), false,
      true);
  patterns.add<PassReduction, 28>(getContext(),
                                  firrtl::createLowerCHIRRTLPass(), true, true);
  patterns.add<PassReduction, 27>(getContext(), firrtl::createInferWidths(),
                                  true, true);
  patterns.add<PassReduction, 26>(getContext(), firrtl::createInferResets(),
                                  true, true);
  patterns.add<FIRRTLModuleExternalizer, 25>();
  patterns.add<InstanceStubber, 24>();
  patterns.add<MemoryStubber, 23>();
  patterns.add<EagerInliner, 22>();
  patterns.add<ObjectInliner, 22>();
  patterns.add<PassReduction, 21>(getContext(),
                                  firrtl::createLowerFIRRTLTypes(), true, true);
  patterns.add<PassReduction, 20>(getContext(), firrtl::createExpandWhens(),
                                  true, true);
  patterns.add<PassReduction, 19>(getContext(), firrtl::createInliner());
  patterns.add<PassReduction, 18>(getContext(), firrtl::createIMConstProp());
  patterns.add<PassReduction, 17>(
      getContext(),
      firrtl::createRemoveUnusedPorts({/*ignoreDontTouch=*/true}));
  patterns.add<NodeSymbolRemover, 15>();
  patterns.add<ConnectForwarder, 14>();
  patterns.add<ConnectInvalidator, 13>();
  patterns.add<Constantifier, 12>();
  patterns.add<FIRRTLOperandForwarder<0>, 11>();
  patterns.add<FIRRTLOperandForwarder<1>, 10>();
  patterns.add<FIRRTLOperandForwarder<2>, 9>();
  patterns.add<ListCreateElementRemover, 8>();
  patterns.add<DetachSubaccesses, 7>();
  patterns.add<ModulePortPruner, 7>();
  patterns.add<ExtmodulePortPruner, 6>();
  patterns.add<RootPortPruner, 5>();
  patterns.add<RootExtmodulePortPruner, 5>();
  patterns.add<ExtmoduleInstanceRemover, 4>();
  patterns.add<ConnectSourceOperandForwarder<0>, 3>();
  patterns.add<ConnectSourceOperandForwarder<1>, 2>();
  patterns.add<ConnectSourceOperandForwarder<2>, 1>();
  patterns.add<ModuleInternalNameSanitizer, 0>();
  patterns.add<ModuleNameSanitizer, 0>();
}

void firrtl::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, FIRRTLDialect *dialect) {
    dialect->addInterfaces<FIRRTLReducePatternDialectInterface>();
  });
}
