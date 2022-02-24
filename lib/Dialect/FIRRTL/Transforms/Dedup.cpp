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
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SHA256.h"

using namespace circt;
using namespace firrtl;
using hw::InnerRefAttr;

/// This stores a mapping from a module name to every NLA that it particiapates
/// in.
using NLAMap = DenseMap<Attribute, std::vector<NonLocalAnchor>>;

NLAMap createNLAMap(CircuitOp circuit) {
  NLAMap nlaMap;
  for (auto nla : circuit.getBody()->getOps<NonLocalAnchor>()) {
    for (size_t i = 0, e = nla.namepath().size(); i != e; ++i)
      nlaMap[nla.modPart(i)].push_back(nla);
  }
  return nlaMap;
}

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

llvm::raw_ostream &printHex(llvm::raw_ostream &stream,
                            ArrayRef<uint8_t> bytes) {
  // Print the hash on a single line.
  return stream << format_bytes(bytes, llvm::None, 32) << "\n";
}

llvm::raw_ostream &printHash(llvm::raw_ostream &stream, llvm::SHA256 &data) {
  auto string = data.result();
  ArrayRef bytes(reinterpret_cast<const uint8_t *>(string.begin()),
                 string.size());
  return printHex(stream, bytes);
}

llvm::raw_ostream &printHash(llvm::raw_ostream &stream, std::string data) {
  ArrayRef bytes(reinterpret_cast<const uint8_t *>(data.c_str()),
                 data.length());
  return printHex(stream, bytes);
}

struct StructuralHasher {
  explicit StructuralHasher(MLIRContext *context) {
    nonessentialAttributes.insert(StringAttr::get(context, "annotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "name"));
    nonessentialAttributes.insert(StringAttr::get(context, "portAnnotations"));
    nonessentialAttributes.insert(StringAttr::get(context, "portNames"));
    nonessentialAttributes.insert(StringAttr::get(context, "portSyms"));
    nonessentialAttributes.insert(StringAttr::get(context, "portTypes"));
    nonessentialAttributes.insert(StringAttr::get(context, "sym_name"));
    nonessentialAttributes.insert(StringAttr::get(context, "inner_sym"));
  };

  std::string hash(FModuleLike module) {
    update(&(*module));
    auto hash = sha.final().str();
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

  void update(BundleType type) {
    update(type.getTypeID());
    for (auto &element : type.getElements()) {
      update(element.isFlip);
      update(element.type);
    }
  }

  void update(Type type) {
    if (auto bundle = type.dyn_cast<BundleType>())
      return update(bundle);
    update(type.getAsOpaquePointer());
  }

  void update(BlockArgument arg) {
    indexes[arg] = currentIndex++;
    update(arg.getArgNumber());
    update(arg.getType());
  }

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
      // Skip names and annotations.
      if (nonessentialAttributes.contains(name))
        continue;
      // Hash the interned pointer.
      update(name.getAsOpaquePointer());
      update(namedAttr.getValue().getAsOpaquePointer());
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

  // This is the actual running hash calculation. This is a stateful element
  // that should be reinitialized after each hash is produced.
  llvm::SHA256 sha;
};

//===----------------------------------------------------------------------===//
// Deduplication
//===----------------------------------------------------------------------===//

struct Deduper {

  using RenameMap = DenseMap<StringAttr, StringAttr>;

  Deduper(InstanceGraph &instanceGraph, SymbolTable &symbolTable,
          NLAMap &nlaMap, CircuitOp circuit)
      : context(circuit->getContext()), instanceGraph(instanceGraph),
        symbolTable(symbolTable), nlaMap(nlaMap), nlaBlock(circuit.getBody()),
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
      rewriteExtModuleNLAs(toModule.moduleNameAttr(),
                           fromModule.moduleNameAttr());

    // Replace all instance of one module with the other.  If the module returns
    // a different bundle type we have to fix up anything connecting to an
    // instance of it.
    fixupReferences(toModule, fromModule);
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

  /// This fixes up connects when the field names of a bundle type changes.  It
  /// finds all fields which were previously bulk connected and legalizes it
  /// into a connect for each field.
  void fixupConnect(ImplicitLocOpBuilder &builder, Value dst, Type dstType,
                    Value src, Type srcType) {
    // If its not a bundle type, the types are guaranteed to be unchanged.  If
    // it is a bundle type, we would rather bulk-connect the values instead of
    // decomposing the connect if the type is unchanged.
    if (dstType == dst.getType() && srcType == src.getType()) {
      builder.create<ConnectOp>(dst, src);
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
      fixupConnect(builder, dstField, dstBundle.getElementType(i), srcField,
                   srcBundle.getElementType(i));
    }
  }

  /// This fixes up a partial connect when the field names of a bundle type
  /// changes.  It finds all the fields which were previously connected and
  /// replaces them with new partial connects.
  using LazyValue = std::function<Value(ImplicitLocOpBuilder &)>;
  // NOLINTNEXTLINE(misc-no-recursion)
  void fixupPartialConnect(ImplicitLocOpBuilder &builder, LazyValue dst,
                           Type dstNewType, Type dstOldType, LazyValue src,
                           Type srcNewType, Type srcOldType) {
    // If the types didn't change, just emit a partial connect.
    if (dstOldType == dstNewType && srcOldType == srcNewType) {
      auto dstField = dst(builder);
      auto srcField = src(builder);
      builder.create<PartialConnectOp>(dstField, srcField);
      return;
    }
    // Check if they are bundle types.
    if (auto dstOldBundle = dstOldType.dyn_cast<BundleType>()) {
      auto dstNewBundle = dstNewType.cast<BundleType>();
      auto srcOldBundle = srcOldType.cast<BundleType>();
      auto srcNewBundle = srcNewType.cast<BundleType>();
      for (auto &pair : llvm::enumerate(dstOldBundle)) {
        // Find a matching field in the old type.
        auto dstField = pair.value();
        auto maybeIndex = srcOldBundle.getElementIndex(dstField.name);
        if (!maybeIndex)
          continue;
        auto dstIndex = pair.index();
        auto srcIndex = *maybeIndex;
        // Recurse on the matching field. The code is complicated because we are
        // trying to avoid creating subfield operations when no field ultimately
        // matches.
        Value dstValue;
        LazyValue dstLazy = [&](ImplicitLocOpBuilder &builder) -> Value {
          if (!dstValue)
            dstValue = builder.create<SubfieldOp>(dst(builder), dstIndex);
          return dstValue;
        };
        auto dstNewElement = dstNewBundle.getElementType(dstIndex);
        auto dstOldElement = dstOldBundle.getElementType(dstIndex);
        Value srcValue;
        LazyValue srcLazy = [&](ImplicitLocOpBuilder &builder) -> Value {
          if (!srcValue)
            srcValue = builder.create<SubfieldOp>(src(builder), srcIndex);
          return srcValue;
        };
        auto srcNewElement = srcNewBundle.getElementType(srcIndex);
        auto srcOldElement = srcOldBundle.getElementType(srcIndex);
        if (dstField.isFlip) {
          std::swap(srcLazy, dstLazy);
          std::swap(srcNewElement, dstNewElement);
          std::swap(srcOldElement, dstOldElement);
        }
        fixupPartialConnect(builder, dstLazy, dstNewElement, dstOldElement,
                            srcLazy, srcNewElement, srcOldElement);
      }
      return;
    }
    // If its not a bundle type, just replace it with a partial connect.
    builder.create<PartialConnectOp>(dst(builder), src(builder));
  }

  /// When we replace a bundle type with a similar bundle with different field
  /// names, we have to rewrite all the code to use the new field names. This
  /// mostly affects subfield result types and any bulk connects.
  // NOLINTNEXTLINE(misc-no-recursion)
  void fixupReferences(Value newValue, Type oldType) {
    SmallVector<std::pair<Value, Type>> workList;
    workList.emplace_back(newValue, oldType);
    while (!workList.empty()) {
      auto [newValue, oldType] = workList.pop_back_val();
      auto newType = newValue.getType();
      for (auto *op : llvm::make_early_inc_range(newValue.getUsers())) {
        // If the two types are identical, we don't need to do anything.
        if (oldType == newType)
          continue;
        if (auto subfield = dyn_cast<SubfieldOp>(op)) {
          // Rewrite a subfield op to return the correct type.
          auto index = subfield.fieldIndex();
          auto newResultType = newType.cast<BundleType>().getElementType(index);
          auto result = subfield.getResult();
          workList.emplace_back(result, result.getType());
          subfield.getResult().setType(newResultType);
          continue;
        }
        if (auto partial = dyn_cast<PartialConnectOp>(op)) {
          // Rewrite the partial connect to connect the same elements.
          auto dstOldType = partial.dest().getType();
          auto dstNewType = dstOldType;
          auto srcOldType = partial.src().getType();
          auto srcNewType = srcOldType;
          // Check which side of the partial connect was updated.
          if (newValue == partial.dest())
            dstOldType = oldType;
          else
            srcOldType = oldType;
          ImplicitLocOpBuilder builder(partial.getLoc(), partial);
          fixupPartialConnect(
              builder, [&](auto) { return partial.dest(); }, dstNewType,
              dstOldType, [&](auto) { return partial.src(); }, srcNewType,
              srcOldType);
          partial->erase();
          continue;
        }
        if (auto connect = dyn_cast<ConnectOp>(op)) {
          // Rewrite the connect to connect all values.
          auto dst = connect.dest();
          auto src = connect.src();
          ImplicitLocOpBuilder builder(connect.getLoc(), connect);
          // Check which side of the connect was updated.
          if (newValue == dst)
            fixupConnect(builder, dst, oldType, src, src.getType());
          else
            fixupConnect(builder, dst, dst.getType(), src, oldType);
          connect->erase();
        }
      }
    }
  }

  /// This is the root method to fixup module references when a module changes.
  /// It matches all the results of "to" module with the results of the "from"
  /// module.
  void fixupReferences(FModuleLike toModule, Operation *fromModule) {
    // Replace all instances of the other module.
    auto *fromNode = instanceGraph[fromModule];
    auto *toNode = instanceGraph[toModule];
    for (auto *oldInstRec : llvm::make_early_inc_range(fromNode->uses())) {
      auto oldInst = oldInstRec->getInstance();
      // Create an instance to replace the old module.
      auto newInst = OpBuilder(oldInst).create<InstanceOp>(
          oldInst.getLoc(), toModule, oldInst.nameAttr());
      newInst.annotationsAttr(oldInst.annotationsAttr());
      auto oldInnerSym = oldInst.inner_symAttr();
      if (oldInnerSym)
        newInst.inner_symAttr(oldInnerSym);

      // We have to replaceAll before fixing up references, we walk the new
      // usages when fixing up any references.
      oldInst.replaceAllUsesWith(newInst.getResults());
      oldInstRec->getParent()->addInstance(newInst, toNode);
      //  Update bulk connections and subfield operations.
      for (auto results :
           llvm::zip(newInst.getResults(), oldInst.getResultTypes()))
        fixupReferences(std::get<0>(results), std::get<1>(results));
      oldInstRec->erase();
      oldInst->erase();
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
      auto parent = cast<FModuleOp>(instanceRecord->getParent()->getModule());
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
      // Make sure we update the NLAMap if the toModule gets deduped later.
      nlaMap[toModuleName].push_back(nla);
      nlaMap[parent.getNameAttr()].push_back(nla);
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
          return record->getInstance().inner_symAttr() == targetInstanceName;
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

  /// This finds all NLAs which contain the "from" module, and renames any
  /// reference to the "to" module.
  void renameModuleInNLA(DenseMap<StringAttr, StringAttr> &renameMap,
                         StringAttr toName, StringAttr fromName,
                         NonLocalAnchor nla) {
    auto fromRef = FlatSymbolRefAttr::get(fromName);
    SmallVector<Attribute> namepath;
    for (auto element : nla.namepath()) {
      if (auto innerRef = element.dyn_cast<InnerRefAttr>()) {
        if (innerRef.getModule() == fromName) {
          auto to = renameMap[innerRef.getName()];
          assert(to && "should have been renamed");
          namepath.push_back(InnerRefAttr::get(toName, to));
        } else
          namepath.push_back(element);
      } else if (element == fromRef) {
        namepath.push_back(FlatSymbolRefAttr::get(toName));
      } else {
        namepath.push_back(element);
      }
    }
    nla.namepathAttr(ArrayAttr::get(context, namepath));
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
      llvm::erase_value(nlaMap[moduleName], nla);
      // Find the instance referenced by the NLA.
      auto *node = instanceGraph.lookup(moduleName);
      auto targetInstanceName = innerRef.getName();
      auto it = llvm::find_if(*node, [&](InstanceRecord *record) {
        return record->getInstance().inner_symAttr() == targetInstanceName;
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
    llvm::erase_value(nlaMap[nla.leafMod()], nla);
    targetMap.erase(nla.getNameAttr());
    nla->erase();
  }

  /// Process all NLAs referencing the "from" module to point to the "to"
  /// module. This is used after merging two modules together.
  void addAnnotationContext(RenameMap &renameMap, FModuleOp toModule,
                            FModuleOp fromModule) {
    auto toName = toModule.getNameAttr();
    auto fromName = fromModule.getNameAttr();
    // Create a copy of the current NLAs. We will be pushing and removing
    // NLAs from this op as we go.
    auto nlas = nlaMap[fromModule.getNameAttr()];
    for (auto nla : nlas) {
      // Change the NLA to target the toModule.
      if (toModule != fromModule)
        renameModuleInNLA(renameMap, toName, fromName, nla);
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
  void rewriteExtModuleNLAs(StringAttr toName, StringAttr fromName) {
    for (auto nla : nlaMap[fromName]) {
      SmallVector<Attribute> namepath;
      // External modules are guaranteed to be the final element of the NLA,
      // since they do not have any bodies.
      llvm::copy(nla.namepath().getValue().drop_back(),
                 std::back_inserter(namepath));
      namepath.push_back(FlatSymbolRefAttr::get(toName));
      nla.namepathAttr(ArrayAttr::get(context, namepath));
    }
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

  // This maps a module name to all NLAs it participates in. This is used to
  // fixup any NLAs when a module is deduped.
  NLAMap &nlaMap;

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
// DedupPass
//===----------------------------------------------------------------------===//

namespace {
class DedupPass : public DedupBase<DedupPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto circuit = getOperation();
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    SymbolTable symbolTable(circuit);
    NLAMap nlaMap = createNLAMap(circuit);
    Deduper deduper(instanceGraph, symbolTable, nlaMap, circuit);
    StructuralHasher hasher(&getContext());
    auto anythingChanged = false;

    // Modules annotated with this should not be considered for deduplication.
    auto noDedupClass =
        StringAttr::get(context, "firrtl.transforms.NoDedupAnnotation");

    // A map of all the module hashes that we have calculated so far.
    llvm::StringMap<Operation *> moduleHashes;

    // We must iterate the modules from the bottom up so that we can properly
    // deduplicate the modules. We have to store the visit order first so that
    // we can safely delete nodes as we go from the instance graph.
    for (auto *node : llvm::post_order(&instanceGraph)) {
      auto module = cast<FModuleLike>(node->getModule());
      // If the module is marked with NoDedup, just skip it.
      if (AnnotationSet(module).hasAnnotation(noDedupClass))
        continue;
      // Calculate the hash of the module.
      auto h = hasher.hash(module);
      // Check if there a module with the same hash.
      auto it = moduleHashes.find(h);
      if (it != moduleHashes.end()) {
        deduper.dedup(it->second, module);
        erasedModules++;
        anythingChanged = true;
        continue;
      }

      // Any module not deduplicated must be recorded.
      deduper.record(module);

      // TODO: if the module is marked must dedup, error here.

      // Record the module's hash if it has not been removed.
      moduleHashes[h] = module;
    }

    if (!anythingChanged)
      markAllAnalysesPreserved();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createDedupPass() {
  return std::make_unique<DedupPass>();
}
