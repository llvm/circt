//===- SpecializeLayers.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <type_traits>

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_SPECIALIZELAYERS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

// TODO: this should be upstreamed.
namespace llvm {
template <>
struct PointerLikeTypeTraits<mlir::ArrayAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::ArrayAttr getFromVoidPointer(void *ptr) {
    return mlir::ArrayAttr::getFromOpaquePointer(ptr);
  }
};
} // namespace llvm

namespace {
/// Removes non-local annotations whose path is no longer viable, due to
/// the removal of module instances.
struct AnnotationCleaner {
  /// Create an AnnotationCleaner which removes any annotation which contains a
  /// reference to a symbol in removedPaths.
  AnnotationCleaner(const DenseSet<StringAttr> &removedPaths)
      : removedPaths(removedPaths) {}

  AnnotationSet cleanAnnotations(AnnotationSet annos) {
    annos.removeAnnotations([&](Annotation anno) {
      if (auto nla = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
        return removedPaths.contains(nla.getAttr());
      return false;
    });
    return annos;
  }

  void cleanAnnotations(Operation *op) {
    AnnotationSet oldAnnotations(op);
    // We want to avoid attaching an empty annotation array on to an op that
    // never had an annotation array in the first place.
    if (!oldAnnotations.empty()) {
      auto newAnnotations = cleanAnnotations(oldAnnotations);
      if (oldAnnotations != newAnnotations)
        newAnnotations.applyToOperation(op);
    }
  }

  void operator()(FModuleLike module) {
    // Clean the regular annotations.
    cleanAnnotations(module);

    // Clean all port annotations.
    for (size_t i = 0, e = module.getNumPorts(); i < e; ++i) {
      auto oldAnnotations = AnnotationSet::forPort(module, i);
      if (!oldAnnotations.empty()) {
        auto newAnnotations = cleanAnnotations(oldAnnotations);
        if (oldAnnotations != newAnnotations)
          newAnnotations.applyToPort(module, i);
      }
    }

    // Clean all annotations in body.
    module->walk([&](Operation *op) {
      // Clean regular annotations.
      cleanAnnotations(op);

      if (auto mem = dyn_cast<MemOp>(op)) {
        // Update all annotations on ports.
        for (size_t i = 0, e = mem.getNumResults(); i < e; ++i) {
          auto oldAnnotations = AnnotationSet::forPort(mem, i);
          if (!oldAnnotations.empty()) {
            auto newAnnotations = cleanAnnotations(oldAnnotations);
            if (oldAnnotations != newAnnotations)
              newAnnotations.applyToPort(mem, i);
          }
        }
      }
    });
  }

  /// A set of symbols of removed paths.
  const DenseSet<StringAttr> &removedPaths;
};

/// Helper to keep track of an insertion point and move operations around.
struct InsertionPoint {
  /// Create an insertion point at the end of a block.
  static InsertionPoint atBlockEnd(Block *block) {
    return InsertionPoint{block, block->end()};
  }

  /// Move the target operation before the current insertion point and update
  /// the insertion point to point to the op.
  void moveOpBefore(Operation *op) {
    op->moveBefore(block, it);
    it = Block::iterator(op);
  }

private:
  InsertionPoint(Block *block, Block::iterator it) : block(block), it(it) {}

  Block *block;
  Block::iterator it;
};

/// A specialized value.  If the value is colored such that it is disabled,
/// it will not contain an underlying value.  Otherwise, contains the
/// specialized version of the value.
template <typename T>
struct Specialized {
  /// Create a disabled specialized value.
  Specialized() : Specialized(LayerSpecialization::Disable, nullptr) {}

  /// Create an enabled specialized value.
  Specialized(T value) : Specialized(LayerSpecialization::Enable, value) {}

  /// Returns true if the value was specialized away.
  bool isDisabled() const {
    return value.getInt() == LayerSpecialization::Disable;
  }

  /// Returns the specialized value if it still exists.
  T getValue() const {
    assert(!isDisabled());
    return value.getPointer();
  }

  operator bool() const { return !isDisabled(); }

private:
  Specialized(LayerSpecialization specialization, T value)
      : value(value, specialization) {}
  llvm::PointerIntPair<T, 1, LayerSpecialization> value;
};

struct SpecializeLayers {
  SpecializeLayers(
      CircuitOp circuit,
      const DenseMap<SymbolRefAttr, LayerSpecialization> &specializations,
      std::optional<LayerSpecialization> defaultSpecialization)
      : context(circuit->getContext()), circuit(circuit),
        specializations(specializations),
        defaultSpecialization(defaultSpecialization) {}

  /// Create a reference to every field in the inner symbol, and record it in
  /// the list of removed symbols.
  static void recordRemovedInnerSym(DenseSet<Attribute> &removedSyms,
                                    StringAttr moduleName,
                                    hw::InnerSymAttr innerSym) {
    for (auto field : innerSym)
      removedSyms.insert(hw::InnerRefAttr::get(moduleName, field.getName()));
  }

  /// Create a reference to every field in the inner symbol, and record it in
  /// the list of removed symbols.
  static void recordRemovedInnerSyms(DenseSet<Attribute> &removedSyms,
                                     StringAttr moduleName, Block *block) {
    block->walk([&](hw::InnerSymbolOpInterface op) {
      if (auto innerSym = op.getInnerSymAttr())
        recordRemovedInnerSym(removedSyms, moduleName, innerSym);
    });
  }

  /// If this layer reference is being specialized, returns the specialization
  /// mode.  Otherwise, it returns disabled value.
  std::optional<LayerSpecialization> getSpecialization(SymbolRefAttr layerRef) {
    auto it = specializations.find(layerRef);
    if (it != specializations.end())
      return it->getSecond();
    return defaultSpecialization;
  }

  /// Forms a symbol reference to a layer using head as the root reference, and
  /// nestedRefs as the path to the specific layer. If this layer reference is
  /// being specialized, returns the specialization mode.  Otherwise, it returns
  /// nullopt.
  std::optional<LayerSpecialization>
  getSpecialization(StringAttr head, ArrayRef<FlatSymbolRefAttr> nestedRefs) {
    return getSpecialization(SymbolRefAttr::get(head, nestedRefs));
  }

  /// Specialize a layer reference by removing enabled layers, mangling the
  /// names of inlined layers, and returning a disabled value if the layer was
  /// disabled. This can return nullptr if all layers were enabled.
  Specialized<SymbolRefAttr> specializeLayerRef(SymbolRefAttr layerRef) {
    // Walk the layer reference from root to leaf, checking if each outer layer
    // was specialized or not.  If an outer layer was disabled, then this
    // specific layer is implicitly disabled as well. If an outer layer was
    // enabled, we need to use its name to mangle the name of inner layers.  If
    // the layer is not specialized, we may need to mangle its name, otherwise
    // we leave it alone.

    auto oldRoot = layerRef.getRootReference();
    SmallVector<FlatSymbolRefAttr> oldNestedRefs;

    // A prefix to be used to track how to mangle the next non-inlined layer.
    SmallString<64> prefix;
    SmallVector<FlatSymbolRefAttr> newRef;

    // Side-effecting helper which returns false if the currently examined layer
    // is disabled, true otherwise.  If the current layer is being enabled, add
    // its name to the prefix.  If the layer is not specialized, we mangle the
    // name with the prefix which is then reset to the empty string, and copy it
    // into the new specialize layer reference.
    auto helper = [&](StringAttr ref) -> bool {
      auto specialization = getSpecialization(oldRoot, oldNestedRefs);

      // We are not specializing this layer. Mangle the name with the current
      // prefix.
      if (!specialization) {
        newRef.push_back(FlatSymbolRefAttr::get(
            StringAttr::get(ref.getContext(), prefix + ref.getValue())));
        prefix.clear();
        return true;
      }

      // We are enabling this layer, the next non-enabled layer should
      // include this layer's name as a prefix.
      if (*specialization == LayerSpecialization::Enable) {
        prefix.append(ref.getValue());
        prefix.append("_");
        return true;
      }

      // We are disabling this layer.
      return false;
    };

    if (!helper(oldRoot))
      return {};

    for (auto ref : layerRef.getNestedReferences()) {
      oldNestedRefs.push_back(ref);
      if (!helper(ref.getAttr()))
        return {};
    }

    if (newRef.empty())
      return {SymbolRefAttr()};

    // Root references need to be handled differently than nested references,
    // but since we don't know before hand which layer will form the new root
    // layer, we copy all layers to the same array, at the cost of unnecessarily
    // wrapping the new root reference into a FlatSymbolRefAttr and having to
    // unpack it again.
    auto newRoot = newRef.front().getAttr();
    return {SymbolRefAttr::get(newRoot, ArrayRef(newRef).drop_front())};
  }

  /// Specialize a RefType by specializing the layer color. If the RefType is
  /// colored with a disabled layer, this will return nullptr.
  RefType specializeRefType(RefType refType) {
    if (auto oldLayer = refType.getLayer()) {
      if (auto newLayer = specializeLayerRef(oldLayer))
        return RefType::get(refType.getType(), refType.getForceable(),
                            newLayer.getValue());
      return nullptr;
    }
    return refType;
  }

  Type specializeType(Type type) {
    if (auto refType = dyn_cast<RefType>(type))
      return specializeRefType(refType);
    return type;
  }

  /// Specialize a value by modifying its type.  Returns nullptr if this value
  /// should be disabled, and the original value otherwise.
  Value specializeValue(Value value) {
    if (auto newType = specializeType(value.getType())) {
      value.setType(newType);
      return value;
    }
    return nullptr;
  }

  /// Specialize a layerblock.  If the layerblock is disabled, it and all of its
  /// contents will be erased, and all removed inner symbols will be recorded
  /// so that we can later clean up hierarchical paths.  If the layer is
  /// enabled, then we will inline the contents of the layer to the same
  /// position and delete the layerblock.
  void specializeOp(LayerBlockOp layerBlock, InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    auto oldLayerRef = layerBlock.getLayerNameAttr();

    // Get the specialization for the current layerblock, not taking into
    // account if an outer layerblock was specialized.  We should not have
    // recursed inside of the disabled layerblock anyways, as it just gets
    // erased.
    auto specialization = getSpecialization(oldLayerRef);

    // We are not specializing this layerblock.
    if (!specialization) {
      //  We must update the name of this layerblock to reflect
      //  specializations of any outer layers.
      auto newLayerRef = specializeLayerRef(oldLayerRef).getValue();
      if (oldLayerRef != newLayerRef)
        layerBlock.setLayerNameAttr(newLayerRef);
      // Specialize inner operations, but keep them in their original
      // location.
      auto *block = layerBlock.getBody();
      auto bodyIP = InsertionPoint::atBlockEnd(block);
      specializeBlock(block, bodyIP, removedSyms);
      insertionPoint.moveOpBefore(layerBlock);
      return;
    }

    // We are enabling this layer, and all contents of this layer need to be
    // moved (inlined) to the insertion point.
    if (*specialization == LayerSpecialization::Enable) {
      // Move all contents to the insertion point and specialize them.
      specializeBlock(layerBlock.getBody(), insertionPoint, removedSyms);
      // Erase the now empty layerblock.
      layerBlock->erase();
      return;
    }

    // We are disabling this layerblock, so we can just erase the layerblock. We
    // need to record all the objects with symbols which are being deleted.
    auto moduleName = layerBlock->getParentOfType<FModuleOp>().getNameAttr();
    recordRemovedInnerSyms(removedSyms, moduleName, layerBlock.getBody());
    layerBlock->erase();
  }

  void specializeOp(WhenOp when, InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    // We need to specialize both arms of the when, but the inner ops should
    // be left where they are (and not inlined to the insertion point).
    auto *thenBlock = &when.getThenBlock();
    auto thenIP = InsertionPoint::atBlockEnd(thenBlock);
    specializeBlock(thenBlock, thenIP, removedSyms);
    if (when.hasElseRegion()) {
      auto *elseBlock = &when.getElseBlock();
      auto elseIP = InsertionPoint::atBlockEnd(elseBlock);
      specializeBlock(elseBlock, elseIP, removedSyms);
    }
    insertionPoint.moveOpBefore(when);
  }

  void specializeOp(MatchOp match, InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    for (size_t i = 0, e = match.getNumRegions(); i < e; ++i) {
      auto *caseBlock = &match.getRegion(i).front();
      auto caseIP = InsertionPoint::atBlockEnd(caseBlock);
      specializeBlock(caseBlock, caseIP, removedSyms);
    }
    insertionPoint.moveOpBefore(match);
  }

  void specializeOp(InstanceOp instance, InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    /// Update the types of any probe ports on the instance, and delete any
    /// probe port that is permanently disabled.
    llvm::BitVector disabledPorts(instance->getNumResults());
    for (auto result : instance->getResults())
      if (!specializeValue(result))
        disabledPorts.set(result.getResultNumber());
    if (disabledPorts.any()) {
      OpBuilder builder(instance);
      auto newInstance = instance.erasePorts(builder, disabledPorts);
      instance->erase();
      instance = newInstance;
    }

    // Specialize the required enable layers.  Due to the layer verifiers, there
    // should not be any disabled layer in this instance and this should be
    // infallible.
    auto newLayers = specializeEnableLayers(instance.getLayersAttr());
    instance.setLayersAttr(newLayers.getValue());

    insertionPoint.moveOpBefore(instance);
  }

  void specializeOp(InstanceChoiceOp instanceChoice,
                    InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    /// Update the types of any probe ports on the instanceChoice, and delete
    /// any probe port that is permanently disabled.
    llvm::BitVector disabledPorts(instanceChoice->getNumResults());
    for (auto result : instanceChoice->getResults())
      if (!specializeValue(result))
        disabledPorts.set(result.getResultNumber());
    if (disabledPorts.any()) {
      OpBuilder builder(instanceChoice);
      auto newInstanceChoice =
          instanceChoice.erasePorts(builder, disabledPorts);
      instanceChoice->erase();
      instanceChoice = newInstanceChoice;
    }

    // Specialize the required enable layers.  Due to the layer verifiers, there
    // should not be any disabled layer in this instanceChoice and this should
    // be infallible.
    auto newLayers = specializeEnableLayers(instanceChoice.getLayersAttr());
    instanceChoice.setLayersAttr(newLayers.getValue());

    insertionPoint.moveOpBefore(instanceChoice);
  }

  void specializeOp(WireOp wire, InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    if (specializeValue(wire.getResult())) {
      insertionPoint.moveOpBefore(wire);
    } else {
      if (auto innerSym = wire.getInnerSymAttr())
        recordRemovedInnerSym(removedSyms,
                              wire->getParentOfType<FModuleOp>().getNameAttr(),
                              innerSym);
      wire.erase();
    }
  }

  void specializeOp(RefDefineOp refDefine, InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    // If this is connected disabled probes, erase the refdefine op.
    if (auto layerRef = refDefine.getDest().getType().getLayer())
      if (!specializeLayerRef(layerRef)) {
        refDefine->erase();
        return;
      }
    insertionPoint.moveOpBefore(refDefine);
  }

  void specializeOp(RefSubOp refSub, InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    if (specializeValue(refSub->getResult(0)))
      insertionPoint.moveOpBefore(refSub);
    else

      refSub.erase();
  }

  void specializeOp(RefCastOp refCast, InsertionPoint &insertionPoint,
                    DenseSet<Attribute> &removedSyms) {
    if (specializeValue(refCast->getResult(0)))
      insertionPoint.moveOpBefore(refCast);
    else
      refCast.erase();
  }

  /// Specialize a block of operations, removing any probes which are
  /// disabled, and moving all operations to the insertion point.
  void specializeBlock(Block *block, InsertionPoint &insertionPoint,
                       DenseSet<Attribute> &removedSyms) {
    // Since this can erase operations that deal with disabled probes, we walk
    // the block in reverse to make sure that we erase uses before defs.
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
      TypeSwitch<Operation *>(&op)
          .Case<LayerBlockOp, WhenOp, MatchOp, InstanceOp, InstanceChoiceOp,
                WireOp, RefDefineOp, RefSubOp, RefCastOp>(
              [&](auto op) { specializeOp(op, insertionPoint, removedSyms); })
          .Default([&](Operation *op) {
            // By default all operations should be inlined from an enabled
            // layer.
            insertionPoint.moveOpBefore(op);
          });
    }
  }

  /// Specialize the list of known layers for an extmodule.
  ArrayAttr specializeKnownLayers(ArrayAttr layers) {
    SmallVector<Attribute> newLayers;
    for (auto layer : layers.getAsRange<SymbolRefAttr>()) {
      if (auto result = specializeLayerRef(layer))
        if (auto newLayer = result.getValue())
          newLayers.push_back(newLayer);
    }

    return ArrayAttr::get(context, newLayers);
  }

  /// Specialize the list of enabled layers for a module.  Return a disabled
  /// layer if one of the required layers has been disabled.
  Specialized<ArrayAttr> specializeEnableLayers(ArrayAttr layers) {
    SmallVector<Attribute> newLayers;
    for (auto layer : layers.getAsRange<SymbolRefAttr>()) {
      auto newLayer = specializeLayerRef(layer);
      if (!newLayer)
        return {};
      if (newLayer.getValue())
        newLayers.push_back(newLayer.getValue());
    }
    return ArrayAttr::get(context, newLayers);
  }

  void specializeModulePorts(FModuleLike moduleLike,
                             DenseSet<Attribute> &removedSyms) {
    auto oldTypeAttrs = moduleLike.getPortTypesAttr();

    // The list of new port types.
    SmallVector<Attribute> newTypeAttrs;
    newTypeAttrs.reserve(oldTypeAttrs.size());

    // This is the list of port indices which need to be removed because they
    // have been specialized away.
    llvm::BitVector disabledPorts(oldTypeAttrs.size());

    auto moduleName = moduleLike.getNameAttr();
    for (auto [index, typeAttr] :
         llvm::enumerate(oldTypeAttrs.getAsRange<TypeAttr>())) {
      // Specialize the type fo the port.
      if (auto type = specializeType(typeAttr.getValue())) {
        newTypeAttrs.push_back(TypeAttr::get(type));
      } else {
        // The port is being disabled, and should be removed.
        if (auto portSym = moduleLike.getPortSymbolAttr(index))
          recordRemovedInnerSym(removedSyms, moduleName, portSym);
        disabledPorts.set(index);
      }
    }

    // Erase the disabled ports.
    moduleLike.erasePorts(disabledPorts);

    // Update the rest of the port types.
    moduleLike.setPortTypesAttr(
        ArrayAttr::get(moduleLike.getContext(), newTypeAttrs));

    // We may also need to update the types on the block arguments.
    if (auto moduleOp = dyn_cast<FModuleOp>(moduleLike.getOperation()))
      for (auto [arg, typeAttr] :
           llvm::zip(moduleOp.getArguments(), newTypeAttrs))
        arg.setType(cast<TypeAttr>(typeAttr).getValue());
  }

  template <typename T>
  DenseSet<Attribute> specializeModuleLike(T op) {
    DenseSet<Attribute> removedSyms;

    // Specialize all operations in the body of the module. This must be done
    // before specializing the module ports so that we don't try to erase values
    // that still have uses.
    if constexpr (std::is_same_v<T, FModuleOp>) {
      auto *block = cast<FModuleOp>(op).getBodyBlock();
      auto bodyIP = InsertionPoint::atBlockEnd(block);
      specializeBlock(block, bodyIP, removedSyms);
    }

    // Specialize the ports of this module.
    specializeModulePorts(op, removedSyms);

    return removedSyms;
  }

  template <typename T>
  T specializeEnableLayers(T module, DenseSet<Attribute> &removedSyms) {
    // Update the required layers on the module.
    if (auto newLayers = specializeEnableLayers(module.getLayersAttr())) {
      module.setLayersAttr(newLayers.getValue());
      return module;
    }

    // If we disabled a layer which this module requires, we must delete the
    // whole module.
    auto moduleName = module.getNameAttr();
    removedSyms.insert(FlatSymbolRefAttr::get(moduleName));
    if constexpr (std::is_same_v<T, FModuleOp>)
      recordRemovedInnerSyms(removedSyms, moduleName,
                             cast<FModuleOp>(module).getBodyBlock());

    module->erase();
    return nullptr;
  }

  /// Specialize the known layers of an extmodule,.
  void specializeKnownLayers(FExtModuleOp module) {
    auto knownLayers = module.getKnownLayersAttr();
    module.setKnownLayersAttr(specializeKnownLayers(knownLayers));
  }

  /// Specialize a layer operation, by removing enabled layers and inlining
  /// their contents, deleting disabled layers and all nested layers, and
  /// mangling the names of any inlined layers.
  void specializeLayer(LayerOp layer) {
    StringAttr head = layer.getSymNameAttr();
    SmallVector<FlatSymbolRefAttr> nestedRefs;

    std::function<void(LayerOp, Block::iterator, const Twine &)> handleLayer =
        [&](LayerOp layer, Block::iterator insertionPoint,
            const Twine &prefix) {
          auto *block = &layer.getBody().getBlocks().front();
          auto specialization = getSpecialization(head, nestedRefs);

          // If we are not specializing the current layer, visit the inner
          // layers.
          if (!specialization) {
            // We only mangle the name and move the layer if the prefix is
            // non-empty, which indicates that we are enabling the parent
            // layer.
            if (!prefix.isTriviallyEmpty()) {
              layer.setSymNameAttr(
                  StringAttr::get(context, prefix + layer.getSymName()));
              auto *parentBlock = insertionPoint->getBlock();
              layer->moveBefore(parentBlock, insertionPoint);
            }
            for (auto nested :
                 llvm::make_early_inc_range(block->getOps<LayerOp>())) {
              nestedRefs.push_back(SymbolRefAttr::get(nested));
              handleLayer(nested, Block::iterator(nested), "");
              nestedRefs.pop_back();
            }
            return;
          }

          // We are enabling this layer.  We must inline inner layers, and
          // mangle their names.
          if (*specialization == LayerSpecialization::Enable) {
            for (auto nested :
                 llvm::make_early_inc_range(block->getOps<LayerOp>())) {
              nestedRefs.push_back(SymbolRefAttr::get(nested));
              handleLayer(nested, insertionPoint,
                          prefix + layer.getSymName() + "_");
              nestedRefs.pop_back();
            }
            // Erase the now empty layer.
            layer->erase();
            return;
          }

          // If we are disabling this layer, then we can just fully delete
          // it.
          layer->erase();
        };

    handleLayer(layer, Block::iterator(layer), "");
  }

  void operator()() {
    // Gather all operations we need to specialize, and all the ops that we
    // need to clean. We specialize layers and module's enable layers here
    // because that can delete the operations and must be done serially.
    SmallVector<Operation *> specialize;
    DenseSet<Attribute> removedSyms;
    for (auto &op : llvm::make_early_inc_range(*circuit.getBodyBlock())) {
      TypeSwitch<Operation *>(&op)
          .Case<FModuleOp>([&](FModuleOp module) {
            if (specializeEnableLayers(module, removedSyms))
              specialize.push_back(module);
          })
          .Case<FExtModuleOp>([&](FExtModuleOp module) {
            specializeKnownLayers(module);
            if (specializeEnableLayers(module, removedSyms))
              specialize.push_back(module);
          })
          .Case<LayerOp>([&](LayerOp layer) { specializeLayer(layer); });
    }

    // Function to merge two sets together.
    auto mergeSets = [](auto &&a, auto &&b) {
      a.insert(b.begin(), b.end());
      return std::forward<decltype(a)>(a);
    };

    // Specialize all modules in parallel. The result is a set of all inner
    // symbol references which are no longer valid due to disabling layers.
    removedSyms = transformReduce(
        context, specialize, removedSyms, mergeSets,
        [&](Operation *op) -> DenseSet<Attribute> {
          return TypeSwitch<Operation *, DenseSet<Attribute>>(op)
              .Case<FModuleOp, FExtModuleOp>(
                  [&](auto op) { return specializeModuleLike(op); });
        });

    // Remove all hierarchical path operations which reference deleted symbols,
    // and create a set of the removed paths operations.  We will have to remove
    // all annotations which use these paths.
    DenseSet<StringAttr> removedPaths;
    for (auto hierPath : llvm::make_early_inc_range(
             circuit.getBody().getOps<hw::HierPathOp>())) {
      auto namepath = hierPath.getNamepath().getValue();
      auto shouldDelete = [&](Attribute ref) {
        return removedSyms.contains(ref);
      };
      if (llvm::any_of(namepath.drop_back(), shouldDelete)) {
        removedPaths.insert(SymbolTable::getSymbolName(hierPath));
        hierPath->erase();
        continue;
      }
      // If we deleted the target of the hierpath, we don't need to add it to
      // the list of removedPaths, since no annotation will be left around to
      // reference this path.
      if (shouldDelete(namepath.back()))
        hierPath->erase();
    }

    // Walk all annotations in the circuit and remove the ones that have a
    // path which traverses or targets a removed instantiation.
    SmallVector<FModuleLike> clean;
    for (auto &op : *circuit.getBodyBlock())
      if (isa<FModuleOp, FExtModuleOp, FIntModuleOp, FMemModuleOp>(op))
        clean.push_back(cast<FModuleLike>(op));

    parallelForEach(context, clean, [&](FModuleLike module) {
      (AnnotationCleaner(removedPaths))(module);
    });
  }

  MLIRContext *context;
  CircuitOp circuit;
  const DenseMap<SymbolRefAttr, LayerSpecialization> &specializations;
  /// The default specialization mode to be applied when a layer has not been
  /// explicitly enabled or disabled.
  std::optional<LayerSpecialization> defaultSpecialization;
};

struct SpecializeLayersPass
    : public circt::firrtl::impl::SpecializeLayersBase<SpecializeLayersPass> {

  void runOnOperation() override {
    auto circuit = getOperation();
    SymbolTableCollection stc;

    // Set of layers to enable or disable.
    DenseMap<SymbolRefAttr, LayerSpecialization> specializations;

    // If we are not specialization any layers, we can return early.
    bool shouldSpecialize = false;

    // Record all the layers which are being enabled.
    if (auto enabledLayers = circuit.getEnableLayersAttr()) {
      shouldSpecialize = true;
      circuit.removeEnableLayersAttr();
      for (auto enabledLayer : enabledLayers.getAsRange<SymbolRefAttr>()) {
        // Verify that this is a real layer.
        if (!stc.lookupSymbolIn(circuit, enabledLayer)) {
          mlir::emitError(circuit.getLoc()) << "unknown layer " << enabledLayer;
          signalPassFailure();
          return;
        }
        specializations[enabledLayer] = LayerSpecialization::Enable;
      }
    }

    // Record all of the layers which are being disabled.
    if (auto disabledLayers = circuit.getDisableLayersAttr()) {
      shouldSpecialize = true;
      circuit.removeDisableLayersAttr();
      for (auto disabledLayer : disabledLayers.getAsRange<SymbolRefAttr>()) {
        // Verify that this is a real layer.
        if (!stc.lookupSymbolIn(circuit, disabledLayer)) {
          mlir::emitError(circuit.getLoc())
              << "unknown layer " << disabledLayer;
          signalPassFailure();
          return;
        }

        // Verify that we are not both enabling and disabling this layer.
        auto [it, inserted] = specializations.try_emplace(
            disabledLayer, LayerSpecialization::Disable);
        if (!inserted && it->getSecond() == LayerSpecialization::Enable) {
          mlir::emitError(circuit.getLoc())
              << "layer " << disabledLayer << " both enabled and disabled";
          signalPassFailure();
          return;
        }
      }
    }

    std::optional<LayerSpecialization> defaultSpecialization = std::nullopt;
    if (auto specialization = circuit.getDefaultLayerSpecialization()) {
      shouldSpecialize = true;
      defaultSpecialization = *specialization;
    }

    // If we did not transform the circuit, return early.
    // TODO: if both arrays are empty we could preserve specific analyses, but
    // not all analyses since we have modified the circuit op.
    if (!shouldSpecialize)
      return markAllAnalysesPreserved();

    // Run specialization on our circuit.
    SpecializeLayers(circuit, specializations, defaultSpecialization)();
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> firrtl::createSpecializeLayersPass() {
  return std::make_unique<SpecializeLayersPass>();
}
