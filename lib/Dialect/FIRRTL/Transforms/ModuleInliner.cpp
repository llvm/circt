//===- ModuleInliner.cpp - FIRRTL module inlining ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FIRRTL module instance inlining.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "firrtl-inliner"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using hw::InnerRefAttr;
using llvm::BitVector;

//===----------------------------------------------------------------------===//
// Module Inlining Support
//===----------------------------------------------------------------------===//

namespace {
/// A representation of an NLA that can be mutated.  This is intended to be used
/// in situations where you want to make a series of modifications to an NLA
/// while also being able to query information about it.  Finally, the NLA is
/// written back to the IR to replace the original NLA.
class MutableNLA {
  // Storage of the NLA this represents.
  NonLocalAnchor nla;

  // A namespace that can be used to generate new symbol names if needed.
  CircuitNamespace *circuitNamespace;

  /// A mapping of symbol to index in the NLA.
  DenseMap<Attribute, unsigned> symIdx;

  /// Records which elements of the path are inlined.
  BitVector inlinedSymbols;

  /// The point after which the NLA is flattened.  A value of "-1" indicates
  /// that this was never set.
  signed flattenPoint = -1;

  /// Indicates if the _original_ NLA is dead and should be deleted.  Updates
  /// may still need to be written if the newTops vector below is non-empty.
  bool dead = false;

  /// Stores new roots for the NLA.  If this is non-empty, then it indicates
  /// that the NLA should be copied and re-topped using the roots stored here.
  /// This is non-empty when athe NLA's root is inlined and the original NLA
  /// migrates to each instantiator of the original NLA.
  SmallVector<InnerRefAttr> newTops;

  /// Cache of roots that this module participates in.  This is only valid when
  /// newTops is non-empty.
  DenseSet<StringAttr> rootSet;

  /// Stores the size of the NLA path.
  unsigned int size;

  /// A mapping of module name to _new_ inner symbol name.  For convenience of
  /// how this pass works (operations are inlined *into* a new module), the key
  /// is the NEW module, after inlining/flattening as opposed to on the old
  /// module.
  DenseMap<Attribute, StringAttr> renames;

  /// Lookup a reference and apply any renames to it.  This requires both the
  /// module where the NEW reference lives (to lookup the rename) and the
  /// original ID of the reference (to fallback to if the reference was not
  /// renamed).
  StringAttr lookupRename(Attribute lastMod, unsigned idx = 0) {
    if (renames.count(lastMod))
      return renames[lastMod];
    return nla.refPart(idx);
  }

public:
  MutableNLA(NonLocalAnchor nla, CircuitNamespace *circuitNamespace)
      : nla(nla), circuitNamespace(circuitNamespace),
        inlinedSymbols(BitVector(nla.namepath().size(), true)),
        size(nla.namepath().size()) {
    for (size_t i = 0, e = size; i != e; ++i)
      symIdx.insert({nla.modPart(i), i});
  }

  /// This default, erroring constructor exists because the pass uses
  /// `DenseMap<Attribute, MutableNLA>`.  `DenseMap` requires a default
  /// constructor for the value type because its `[]` operator (which returns a
  /// reference) must default construct the value type for a non-existent key.
  /// This default constructor is never supposed to be used because the pass
  /// prepulates a `DenseMap<Attribute, MutbaleNLA>` before it runs and thereby
  /// guarantees that `[]` will always hit and never need to use the default
  /// constructor.
  MutableNLA() {
    llvm_unreachable(
        "the default constructor for MutableNLA should never be used");
  }

  /// Set the state of the mutable NLA to indicate that the _original_ NLA
  /// should be removed when updates are applied.
  void markDead() { dead = true; }

  /// Return the original NLA that this was pointing at.
  NonLocalAnchor getNLA() { return nla; }

  /// Writeback updates accumulated in this MutableNLA to the IR.  This method
  /// should only ever be called once and, if a writeback occurrs, the
  /// MutableNLA is NOT updated for further use.  Interacting with the
  /// MutableNLA in any way after calling this method may result in crashes.
  /// (This is done to save unnecessary state cleanup of a pass-private
  /// utility.)
  NonLocalAnchor applyUpdates() {
    // Delete an NLA which is either dead or has been made local.
    if (isLocal() || isDead()) {
      nla.erase();
      return nullptr;
    }

    // The NLA was never updated, just return the NLA and do not writeback
    // anything.
    if (inlinedSymbols.all() && newTops.empty() && flattenPoint == -1)
      return nla;

    // The NLA has updates.  Generate a new NLA with the same symbol and delete
    // the original NLA.
    OpBuilder b(nla);
    auto writeBack = [&](StringAttr root, StringAttr sym) -> NonLocalAnchor {
      SmallVector<Attribute> namepath;
      StringAttr lastMod;

      // Root of the namepatch.
      if (!inlinedSymbols.test(1))
        lastMod = root;
      else
        namepath.push_back(InnerRefAttr::get(root, lookupRename(root)));

      // Everything in the middle of the namepath (excluding the root and leaf).
      for (signed i = 1, e = inlinedSymbols.size() - 1; i != e; ++i) {
        if (i == flattenPoint) {
          lastMod = nla.modPart(i);
          break;
        }

        if (!inlinedSymbols.test(i + 1)) {
          if (!lastMod)
            lastMod = nla.modPart(i);
          continue;
        }

        // Update the inner symbol if it has been renamed.
        auto modPart = lastMod ? lastMod : nla.modPart(i);
        auto refPart = lookupRename(modPart, i);
        namepath.push_back(InnerRefAttr::get(modPart, refPart));
        lastMod = {};
      }

      // Leaf of the namepath.
      auto modPart = lastMod ? lastMod : nla.modPart(size - 1);
      auto refPart = lookupRename(modPart, size - 1);

      if (refPart)
        namepath.push_back(InnerRefAttr::get(modPart, refPart));
      else
        namepath.push_back(FlatSymbolRefAttr::get(modPart));

      return b.create<NonLocalAnchor>(b.getUnknownLoc(), sym,
                                      b.getArrayAttr(namepath));
    };

    NonLocalAnchor last;
    assert(!dead || !newTops.empty());
    if (!dead)
      last = writeBack(nla.root(), nla.getNameAttr());
    for (auto root : newTops)
      last = writeBack(root.getModule(), root.getName());

    nla.erase();
    return last;
  }

  void dump() {
    llvm::errs() << "  - orig:           " << nla << "\n"
                 << "    new:            " << *this << "\n"
                 << "    dead:           " << dead << "\n"
                 << "    isDead:         " << isDead() << "\n"
                 << "    isLocal:        " << isLocal() << "\n"
                 << "    inlinedSymbols: [";
    llvm::interleaveComma(inlinedSymbols.getData(), llvm::errs(), [](auto a) {
      llvm::errs() << llvm::formatv("{0:x-}", a);
    });
    llvm::errs() << "]\n"
                 << "    flattenPoint:   " << flattenPoint << "\n"
                 << "    renames:\n";
    for (auto rename : renames)
      llvm::errs() << "      - " << rename.first << " -> " << rename.second
                   << "\n";
  }

  /// Write the current state of this MutableNLA to a string using a format that
  /// looks like the NLA serialization.  This is intended to be used for
  /// debugging purposes.
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, MutableNLA &x) {
    auto writePathSegment = [&](StringAttr mod, StringAttr sym = {}) {
      if (sym)
        os << "#hw.innerNameRef<";
      os << "@" << mod.getValue();
      if (sym)
        os << "::@" << sym.getValue() << ">";
    };

    auto writeOne = [&](StringAttr root, StringAttr sym) {
      os << "firrtl.nla @" << sym.getValue() << " [";

      StringAttr lastMod;
      // Root of the namepath.
      if (!x.inlinedSymbols.test(1))
        lastMod = root;
      else
        writePathSegment(root, x.lookupRename(root));

      // Everything in the middle of the namepath (excluding the root and leaf).
      bool needsComma = false;
      for (signed i = 1, e = x.inlinedSymbols.size() - 1; i != e; ++i) {
        if (i == x.flattenPoint) {
          lastMod = x.nla.modPart(i);
          break;
        }

        if (!x.inlinedSymbols.test(i + 1)) {
          if (!lastMod)
            lastMod = x.nla.modPart(i);
          continue;
        }

        if (needsComma)
          os << ", ";
        auto modPart = lastMod ? lastMod : x.nla.modPart(i);
        auto refPart = x.nla.refPart(i);
        if (x.renames.count(modPart))
          refPart = x.renames[modPart];
        writePathSegment(modPart, refPart);
        needsComma = true;
        lastMod = {};
      }

      // Leaf of the namepath.
      os << ", ";
      auto modPart = lastMod ? lastMod : x.nla.modPart(x.size - 1);
      auto refPart = x.nla.refPart(x.size - 1);
      if (x.renames.count(modPart))
        refPart = x.renames[modPart];
      writePathSegment(modPart, refPart);
      os << "]";
    };

    SmallVector<InnerRefAttr> tops;
    if (!x.dead)
      tops.push_back(InnerRefAttr::get(x.nla.root(), x.nla.getNameAttr()));
    tops.append(x.newTops.begin(), x.newTops.end());

    bool multiary = !x.newTops.empty();
    if (multiary)
      os << "[";
    llvm::interleaveComma(tops, os, [&](InnerRefAttr a) {
      writeOne(a.getModule(), a.getName());
    });
    if (multiary)
      os << "]";

    return os;
  }

  /// Returns true if this NLA is dead.  There are several reasons why this
  /// could be dead:
  ///   1. This NLA has no uses and was not re-topped.
  ///   2. This NLA was flattened and its leaf reference is a Module.
  bool isDead() { return dead && newTops.empty(); }

  /// Returns true if this NLA is local.  For this to be local, every module
  /// after the root (up to the flatten point or the end) must be inlined.  The
  /// root is never truly inlined as inlined as inlining the root just sets a
  /// new root.
  bool isLocal() {
    unsigned end = flattenPoint > -1 ? flattenPoint + 1 : inlinedSymbols.size();
    return inlinedSymbols.find_first_in(1, end) == -1;
  }

  /// Return true if this NLA has a root that originates from a specific module.
  bool hasRoot(FModuleLike mod) {
    return (isDead() && nla.root() == mod.moduleNameAttr()) ||
           rootSet.contains(mod.moduleNameAttr());
  }

  /// Mark a module as inlined.  This will remove it from the NLA.
  void inlineModule(FModuleOp module) {
    auto sym = module.getNameAttr();
    assert(sym != nla.root() && "unable to inline the root module");
    assert(symIdx.count(sym) && "module is not in the symIdx map");
    auto idx = symIdx[sym];
    inlinedSymbols.reset(idx);
    // If we inlined the last module in the path and the NLA ended in a module,
    // then this NLA is dead.
    if (idx == size - 1 && nla.isModule())
      markDead();
  }

  /// Mark a module as flattened.  This has the effect of inlining all of its
  /// children.  Also mark the NLA was dead if the leaf reference of this NLA is
  /// a module.
  void flattenModule(FModuleOp module) {
    auto sym = module.getNameAttr();
    assert(symIdx.count(sym) && "module is not in the symIdx map");
    auto idx = symIdx[sym] - 1;
    flattenPoint = idx;
    // If the leaf reference is a module and we're flattening the NLA, then the
    // NLA must be dead.  Mark it as such.
    if (nla.namepath()[size - 1].isa<FlatSymbolRefAttr>())
      markDead();
  }

  StringAttr reTop(FModuleOp module) {
    StringAttr sym = nla.sym_nameAttr();
    if (!newTops.empty())
      sym = StringAttr::get(nla.getContext(),
                            circuitNamespace->newName(sym.getValue()));
    newTops.push_back(InnerRefAttr::get(module.getNameAttr(), sym));
    rootSet.insert(module.getNameAttr());
    symIdx.insert({module.getNameAttr(), 0});
    markDead();
    return sym;
  }

  ArrayRef<InnerRefAttr> getAdditionalSymbols() {
    return llvm::makeArrayRef(newTops);
  }

  void setInnerSym(Attribute module, StringAttr innerSym) {
    assert(symIdx.count(module) && "Mutalbe NLA did not contain symbol");
    renames.insert({module, innerSym});
  }
};
} // namespace

/// This function is used after inlining a module, to handle the conversion
/// between module ports and instance results. This maps each wire to the
/// result of the instance operation.  When future operations are cloned from
/// the current block, they will use the value of the wire instead of the
/// instance results.
static void mapResultsToWires(BlockAndValueMapping &mapper,
                              SmallVectorImpl<Value> &wires,
                              InstanceOp instance) {
  for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i) {
    auto result = instance.getResult(i);
    auto wire = wires[i];
    mapper.map(result, wire);
  }
}

/// Inlines, flattens, and removes dead modules in a circuit.
///
/// The inliner works in a top down fashion, starting from the top level module,
/// and inlines every possible instance. With this method of recursive top-down
/// inlining, each operation will be cloned directly to its final location.
///
/// The inliner uses a worklist to track which modules need to be processed.
/// When an instance op is not inlined, the referenced module is added to the
/// worklist. When the inliner is complete, it deletes every un-processed
/// module: either all instances of the module were inlined, or it was not
/// reachable from the top level module.
///
/// During the inlining process, every cloned operation with a name must be
/// prefixed with the instance's name. The top-down process means that we know
/// the entire desired prefix when we clone an operation, and can set the name
/// attribute once. This means that we will not create any intermediate name
/// attributes (which will be interned by the compiler), and helps keep down the
/// total memory usage.
namespace {
class Inliner {
public:
  /// Initialize the inliner to run on this circuit.
  Inliner(CircuitOp circuit);

  /// Run the inliner.
  void run();

private:
  /// Rename an operation and unique any symbols it has.
  void rename(StringRef prefix, Operation *op,
              ModuleNamespace &moduleNamespace);

  /// Clone and rename an operation.
  void cloneAndRename(StringRef prefix, OpBuilder &b,
                      BlockAndValueMapping &mapper, Operation &op,
                      const DenseMap<Attribute, Attribute> &symbolRenames,
                      const DenseSet<Attribute> &localSymbols,
                      ModuleNamespace &moduleNamespace);

  /// Rewrite the ports of a module as wires.  This is similar to
  /// cloneAndRename, but operating on ports.
  SmallVector<Value> mapPortsToWires(StringRef prefix, OpBuilder &b,
                                     BlockAndValueMapping &mapper,
                                     FModuleOp target,
                                     const DenseSet<Attribute> &localSymbols,
                                     ModuleNamespace &moduleNamespace);

  /// Returns true if the operation is annotated to be flattened.  This removes
  /// the flattened annotation (hence, this should only be called once on a
  /// module).
  bool shouldFlatten(Operation *op);

  /// Returns true if the operation is annotated to be inlined.
  bool shouldInline(Operation *op);

  /// Flattens a target module in to the insertion point of the builder,
  /// renaming all operations using the prefix.  This clones all operations from
  /// the target, and does not trigger inlining on the target itself.
  void flattenInto(StringRef prefix, OpBuilder &b, BlockAndValueMapping &mapper,
                   FModuleOp target, DenseSet<Attribute> localSymbols,
                   ModuleNamespace &moduleNamespace);

  /// Inlines a target module in to the location of the build, prefixing all
  /// operations with prefix.  This clones all operations from the target, and
  /// does not trigger inlining on the target itself.
  void inlineInto(StringRef prefix, OpBuilder &b, BlockAndValueMapping &mapper,
                  FModuleOp target,
                  DenseMap<Attribute, Attribute> &symbolRenames,
                  ModuleNamespace &moduleNamespace);

  /// Recursively flatten all instances in a module.
  void flattenInstances(FModuleOp module);

  /// Inline any instances in the module which were marked for inlining.
  void inlineInstances(FModuleOp module);

  CircuitOp circuit;
  MLIRContext *context;

  // A symbol table with references to each module in a circuit.
  SymbolTable symbolTable;

  /// The set of live modules.  Anything not recorded in this set will be
  /// removed by dead code elimination.
  DenseSet<Operation *> liveModules;

  /// Worklist of modules to process for inlining or flattening.
  SmallVector<FModuleOp, 16> worklist;

  /// A mapping of NLA symbol name to mutable NLA.
  DenseMap<Attribute, MutableNLA> nlaMap;

  /// A mapping of module names to NLA symbols that originate from that module.
  DenseMap<Attribute, SmallVector<Attribute>> rootMap;
};
} // namespace

/// If this operation or any child operation has a name, add the prefix to that
/// operation's name.  If the operation has any inner symbols, make sure that
/// these are unique in the namespace.
void Inliner::rename(StringRef prefix, Operation *op,
                     ModuleNamespace &moduleNamespace) {
  // Add a prefix to _anything_ that has a "name" attribute.
  if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
    op->setAttr("name", StringAttr::get(op->getContext(),
                                        (prefix + nameAttr.getValue())));

  // If the operation has an inner symbol, ensure that it is unique.  Record
  // renames for any NLAs that this participates in if the symbol was renamed.
  if (auto sym = op->getAttrOfType<StringAttr>("inner_sym")) {
    auto newSym = moduleNamespace.newName(sym.getValue());
    if (newSym != sym.getValue()) {
      auto newSymAttr = StringAttr::get(op->getContext(), newSym);
      op->setAttr("inner_sym", newSymAttr);
      for (Annotation anno : AnnotationSet(op)) {
        auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
        if (!sym)
          continue;
        nlaMap[sym.getAttr()].setInnerSym(
            moduleNamespace.module.moduleNameAttr(), newSymAttr);
      }
    }
  }

  // Recursively rename any child operations.
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &op : block)
        rename(prefix, &op, moduleNamespace);
}

/// This function is used before inlining a module, to handle the conversion
/// between module ports and instance results. For every port in the target
/// module, create a wire, and assign a mapping from each module port to the
/// wire. When the body of the module is cloned, the value of the wire will be
/// used instead of the module's ports.
SmallVector<Value>
Inliner::mapPortsToWires(StringRef prefix, OpBuilder &b,
                         BlockAndValueMapping &mapper, FModuleOp target,
                         const DenseSet<Attribute> &localSymbols,
                         ModuleNamespace &moduleNamespace) {
  SmallVector<Value> wires;
  auto portInfo = target.getPorts();
  for (unsigned i = 0, e = target.getNumPorts(); i < e; ++i) {
    auto arg = target.getArgument(i);
    // Get the type of the wire.
    auto type = arg.getType().cast<FIRRTLType>();

    // Compute a unique symbol if needed
    StringAttr newSym;
    StringAttr oldSym = portInfo[i].sym;
    if (!oldSym.getValue().empty())
      newSym = b.getStringAttr(moduleNamespace.newName(oldSym.getValue()));

    auto annotations = AnnotationSet::forPort(target, i);
    SmallVector<Annotation> newAnnotations;
    annotations.removeAnnotations([&](Annotation anno) {
      if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        // Update any NLAs with the new symbol name.
        if (oldSym != newSym)
          nlaMap[sym.getAttr()].setInnerSym(
              moduleNamespace.module.moduleNameAttr(), newSym);

        if (nlaMap[sym.getAttr()].isLocal() ||
            localSymbols.count(sym.getAttr())) {
          anno.removeMember("circt.nonlocal");
          newAnnotations.push_back(anno);
          return true;
        }
      }
      return false;
    });
    annotations.addAnnotations(newAnnotations);

    auto wire = b.create<WireOp>(target.getLoc(), type,
                                 (prefix + portInfo[i].getName()).str(),
                                 annotations.getArray(), newSym);
    wires.push_back(wire);
    mapper.map(arg, wire.getResult());
  }
  return wires;
}

/// Clone an operation, mapping used values and results with the mapper, and
/// apply the prefix to the name of the operation. This will clone to the
/// insert point of the builder.
void Inliner::cloneAndRename(
    StringRef prefix, OpBuilder &b, BlockAndValueMapping &mapper, Operation &op,
    const DenseMap<Attribute, Attribute> &symbolRenames,
    const DenseSet<Attribute> &localSymbols, ModuleNamespace &moduleNamespace) {
  // Strip any non-local annotations which are local.
  AnnotationSet annotations(&op);
  SmallVector<Annotation> newAnnotations;
  if (!annotations.empty()) {
    annotations.removeAnnotations([&](Annotation anno) {
      if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        // The NLA is local, rewrite it to be local.
        if (nlaMap[sym.getAttr()].isLocal() ||
            localSymbols.count(sym.getAttr())) {
          anno.removeMember("circt.nonlocal");
          newAnnotations.push_back(anno);
          return true;
        }
        // The NLA has been renamed.  This only matters for InstanceOps.
        if (!isa<InstanceOp>(op) || symbolRenames.empty())
          return false;
        NamedAttrList newAnnotation;
        if (auto newSym =
                symbolRenames.lookup(sym.getAttr()).cast<StringAttr>()) {
          anno.setMember("circt.nonlocal", FlatSymbolRefAttr::get(newSym));
          newAnnotations.push_back(anno);
          return true;
        }
      }
      return false;
    });
    if (!newAnnotations.empty())
      annotations.addAnnotations(newAnnotations);
  }

  // Clone and rename.
  auto *newOp = b.clone(op, mapper);
  rename(prefix, newOp, moduleNamespace);

  if (newAnnotations.empty())
    return;
  annotations.applyToOperation(newOp);
}

bool Inliner::shouldFlatten(Operation *op) {
  return AnnotationSet::removeAnnotations(op, [](Annotation a) {
    return a.isClass("firrtl.transforms.FlattenAnnotation");
  });
}

bool Inliner::shouldInline(Operation *op) {
  return AnnotationSet(op).hasAnnotation("firrtl.passes.InlineAnnotation");
}

void Inliner::flattenInto(StringRef prefix, OpBuilder &b,
                          BlockAndValueMapping &mapper, FModuleOp target,
                          DenseSet<Attribute> localSymbols,
                          ModuleNamespace &moduleNamespace) {
  DenseMap<Attribute, Attribute> symbolRenames;
  for (auto &op : *target.getBody()) {
    // If its not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      cloneAndRename(prefix, b, mapper, op, symbolRenames, localSymbols,
                     moduleNamespace);
      continue;
    }

    // If its not a regular module we can't inline it. Mark is as live.
    auto *module = symbolTable.lookup(instance.moduleName());
    auto target = dyn_cast<FModuleOp>(module);
    if (!target) {
      liveModules.insert(module);
      cloneAndRename(prefix, b, mapper, op, symbolRenames, localSymbols,
                     moduleNamespace);
      continue;
    }

    // Add any NLAs which start at this instance to the localSymbols set.
    // Anything in this set will be made local during the recursive flattenInto
    // walk.
    llvm::set_union(localSymbols, rootMap[target.getNameAttr()]);

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.name() + "_").str();
    auto wires = mapPortsToWires(nestedPrefix, b, mapper, target, localSymbols,
                                 moduleNamespace);
    mapResultsToWires(mapper, wires, instance);

    // Unconditionally flatten all instance operations.
    flattenInto(nestedPrefix, b, mapper, target, localSymbols, moduleNamespace);
  }
}

void Inliner::flattenInstances(FModuleOp module) {
  // Namespace used to generate new symbol names.
  ModuleNamespace moduleNamespace(module);

  for (auto &op : llvm::make_early_inc_range(*module.getBody())) {
    // If its not an instance op, skip it.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance)
      continue;

    // If its not a regular module we can't inline it. Mark is as live.
    auto *module = symbolTable.lookup(instance.moduleName());
    auto target = dyn_cast<FModuleOp>(module);
    if (!target) {
      liveModules.insert(module);
      continue;
    }

    // Preorder update of any non-local annotations this instance participates
    // in.  This needs to happen _before_ visiting modules so that internal
    // non-local annotations can be deleted if they are now local.
    AnnotationSet annotations(instance);
    for (auto anno : annotations) {
      if (anno.isClass("circt.nonlocal")) {
        auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
        nlaMap[sym.getAttr()].flattenModule(target);
      }
    }

    // Add any NLAs which start at this instance to the localSymbols set.
    // Anything in this set will be made local during the recursive flattenInto
    // walk.
    DenseSet<Attribute> localSymbols;
    llvm::set_union(localSymbols, rootMap[target.getNameAttr()]);

    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    BlockAndValueMapping mapper;
    OpBuilder b(instance);
    auto nestedPrefix = (instance.name() + "_").str();
    auto wires = mapPortsToWires(nestedPrefix, b, mapper, target, localSymbols,
                                 moduleNamespace);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(wires[i]);

    // Recursively flatten the target module.
    flattenInto(nestedPrefix, b, mapper, target, localSymbols, moduleNamespace);

    // Erase the replaced instance.
    instance.erase();
  }
}

void Inliner::inlineInto(StringRef prefix, OpBuilder &b,
                         BlockAndValueMapping &mapper, FModuleOp parent,
                         DenseMap<Attribute, Attribute> &symbolRenames,
                         ModuleNamespace &moduleNamespace) {
  // Inline everything in the module's body.
  for (auto &op : *parent.getBody()) {
    // If its not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      cloneAndRename(prefix, b, mapper, op, symbolRenames, {}, moduleNamespace);
      continue;
    }

    // If its not a regular module we can't inline it. Mark is as live.
    auto *module = symbolTable.lookup(instance.moduleName());
    auto target = dyn_cast<FModuleOp>(module);
    if (!target) {
      liveModules.insert(module);
      cloneAndRename(prefix, b, mapper, op, symbolRenames, {}, moduleNamespace);
      continue;
    }

    // If we aren't inlining the target, add it to the work list.
    if (!shouldInline(target)) {
      if (liveModules.insert(target).second) {
        worklist.push_back(target);
      }
      cloneAndRename(prefix, b, mapper, op, symbolRenames, {}, moduleNamespace);
      continue;
    }

    // Preorder update of any non-local annotations this instance participates
    // in.  This needs ot happen _before_ visiting modules so that internal
    // non-local annotations can be deleted if they are now local.
    auto toBeFlattened = shouldFlatten(target);
    AnnotationSet annotations(instance);
    for (auto anno : annotations) {
      if (anno.isClass("circt.nonlocal")) {
        auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
        if (toBeFlattened)
          nlaMap[sym.getAttr()].flattenModule(target);
        else
          nlaMap[sym.getAttr()].inlineModule(target);
      }
    }

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.name() + "_").str();
    auto wires =
        mapPortsToWires(nestedPrefix, b, mapper, target, {}, moduleNamespace);
    mapResultsToWires(mapper, wires, instance);

    // If we're about to inline a module that contains a non-local annotation
    // that starts at that module, then we need to both update the mutable NLA
    // to indicate that this has a new top and add an annotation on the instance
    // saying that this now participates in this new NLA.
    DenseMap<Attribute, Attribute> symbolRenames;
    if (!rootMap[target.getNameAttr()].empty()) {
      for (auto sym : rootMap[target.getNameAttr()]) {
        auto &mnla = nlaMap[sym];
        sym = mnla.reTop(parent);
        // TODO: Update any symbol renames which need to be used by the next
        // call of inlineInto.  This will then check each instance and rename
        // any symbols appropriately for that instance.
        symbolRenames.insert({mnla.getNLA().getNameAttr(), sym});
      }
    }

    // Inline the module, it can be marked as flatten and inline.
    if (toBeFlattened) {
      flattenInto(nestedPrefix, b, mapper, target, {}, moduleNamespace);
    } else {
      inlineInto(nestedPrefix, b, mapper, target, symbolRenames,
                 moduleNamespace);
    }
  }
}

void Inliner::inlineInstances(FModuleOp parent) {
  // Generate a namespace for this module so that we can safely inline symbols.
  ModuleNamespace moduleNamespace(parent);

  for (auto &op : llvm::make_early_inc_range(*parent.getBody())) {
    // If its not an instance op, skip it.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance)
      continue;

    // If its not a regular module we can't inline it. Mark is as live.
    auto *module = symbolTable.lookup(instance.moduleName());
    auto target = dyn_cast<FModuleOp>(module);
    if (!target) {
      liveModules.insert(module);
      continue;
    }

    // If we aren't inlining the target, add it to the work list.
    if (!shouldInline(target)) {
      if (liveModules.insert(target).second) {
        worklist.push_back(target);
      }
      continue;
    }

    // Preorder update of any non-local annotations this instance participates
    // in.  This needs ot happen _before_ visiting modules so that internal
    // non-local annotations can be deleted if they are now local.
    auto toBeFlattened = shouldFlatten(target);
    AnnotationSet annotations(instance);
    for (auto anno : annotations) {
      if (anno.isClass("circt.nonlocal")) {
        auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
        if (toBeFlattened)
          nlaMap[sym.getAttr()].flattenModule(target);
        else
          nlaMap[sym.getAttr()].inlineModule(target);
      }
    }

    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    BlockAndValueMapping mapper;
    OpBuilder b(instance);
    auto nestedPrefix = (instance.name() + "_").str();
    auto wires =
        mapPortsToWires(nestedPrefix, b, mapper, target, {}, moduleNamespace);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(wires[i]);

    DenseMap<Attribute, Attribute> symbolRenames;
    if (!rootMap[target.getNameAttr()].empty()) {
      for (auto sym : rootMap[target.getNameAttr()]) {
        auto &mnla = nlaMap[sym];
        sym = mnla.reTop(parent);
        // TODO: Update any symbol renames which need to be used by the next
        // call of inlineInto.  This will then check each instance and rename
        // any symbols appropriately for that instance.
        symbolRenames.insert({mnla.getNLA().getNameAttr(), sym});
      }
    }

    // Inline the module, it can be marked as flatten and inline.
    if (toBeFlattened) {
      flattenInto(nestedPrefix, b, mapper, target, {}, moduleNamespace);
    } else {
      inlineInto(nestedPrefix, b, mapper, target, symbolRenames,
                 moduleNamespace);
    }

    // Erase the replaced instance.
    instance.erase();
  }
}

Inliner::Inliner(CircuitOp circuit)
    : circuit(circuit), context(circuit.getContext()), symbolTable(circuit) {}

void Inliner::run() {
  auto *topModule = circuit.getMainModule();
  CircuitNamespace circuitNamespace(circuit);

  for (auto nla : circuit.getBody()->getOps<NonLocalAnchor>()) {
    auto mnla = MutableNLA(nla, &circuitNamespace);
    nlaMap.insert({nla.sym_nameAttr(), mnla});
    rootMap[mnla.getNLA().root()].push_back(nla.sym_nameAttr());
  }

  // Mark the top module as live, so it doesn't get deleted.
  liveModules.insert(topModule);

  // If the top module is not a regular module, there is nothing to do.
  if (auto fmodule = dyn_cast<FModuleOp>(topModule))
    worklist.push_back(fmodule);

  // If the module is marked for flattening, flatten it. Otherwise, inline
  // every instance marked to be inlined.
  while (!worklist.empty()) {
    auto module = worklist.pop_back_val();
    if (shouldFlatten(module)) {
      flattenInstances(module);
      continue;
    }
    inlineInstances(module);

    // Delete the flatten annotations. Any module with the inline annotation
    // will be deleted, as there won't be any remaining instances of it.
    AnnotationSet(module).removeAnnotationsWithClass(
        "firrtl.transforms.FlattenAnnotation");
  }

  // Delete all unreferenced modules.  Mark any NLAs that originate from dead
  // modules as also dead.
  for (auto mod :
       llvm::make_early_inc_range(circuit.getBody()->getOps<FModuleLike>())) {
    if (liveModules.count(mod))
      continue;
    for (auto nla : rootMap[mod.moduleNameAttr()])
      nlaMap[nla].markDead();
    mod.erase();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "NLA modifications:\n";
    for (auto nla : circuit.getBody()->getOps<NonLocalAnchor>()) {
      auto &mnla = nlaMap[nla.getNameAttr()];
      mnla.dump();
    }
  });

  // Writeback all NLAs to MLIR.
  for (auto &nla : nlaMap)
    nla.getSecond().applyUpdates();

  // Garbage collect any annotations which are now dead.  Duplicate annotations
  // which are now split.
  for (auto fmodule : circuit.getBody()->getOps<FModuleOp>()) {
    for (auto &op : *fmodule.getBody()) {
      AnnotationSet annotations(&op);
      // Early exit to avoid adding an empty annotations attribute to operations
      // which did not previously have annotations.
      if (annotations.empty())
        continue;

      SmallVector<Attribute> newAnnotations;
      auto processNLAs = [&](Annotation anno) -> bool {
        if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
          // If the symbol isn't in the NLA map, just skip it.  This avoids
          // problems where the nlaMap "[]" will try to construct a default
          // MutableNLA map (which it should never do).
          if (!nlaMap.count(sym.getAttr()))
            return false;

          auto mnla = nlaMap[sym.getAttr()];

          // Garbage collect dead NLA references.  This cleans up NLAs that go
          // through modules which we never visited.
          if (mnla.isDead())
            return true;

          // Do nothing if there are no additional NLAs to add or if we're
          // dealing with a root module.  Root modules have already been updated
          // earlier in the pass.  We only need to update NLA paths which are
          // not the root.
          auto newTops = mnla.getAdditionalSymbols();
          if (newTops.size() == 0 || mnla.hasRoot(fmodule))
            return false;

          // Add NLAs to the non-root portion of the NLA.  This only needs to
          // add symbols for NLAs which are after the first one.  We reused the
          // old symbol name for the first NLA.
          NamedAttrList newAnnotation;
          for (auto rootAndSym : newTops.drop_front()) {
            for (auto pair : anno.getDict()) {
              if (pair.getName().getValue() != "circt.nonlocal") {
                newAnnotation.push_back(pair);
                continue;
              }
              newAnnotation.push_back(
                  {pair.getName(),
                   FlatSymbolRefAttr::get(rootAndSym.getName())});
            }
            newAnnotations.push_back(
                DictionaryAttr::get(op.getContext(), newAnnotation));
          }
        }
        return false;
      };

      // Update annotations on the module.
      annotations.removeAnnotations(processNLAs);
      annotations.addAnnotations(newAnnotations);
      annotations.applyToOperation(&op);

      // Update annotations on the ports.
      SmallVector<Attribute> newPortAnnotations;
      for (auto port : fmodule.getPorts()) {
        newAnnotations.clear();
        port.annotations.removeAnnotations(processNLAs);
        port.annotations.addAnnotations(newAnnotations);
        newPortAnnotations.push_back(
            ArrayAttr::get(op.getContext(), port.annotations.getArray()));
      }
      fmodule->setAttr("portAnnotations",
                       ArrayAttr::get(op.getContext(), newPortAnnotations));
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InlinerPass : public InlinerBase<InlinerPass> {
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs()
               << "===- Running Module Inliner Pass "
                  "--------------------------------------------===\n");
    Inliner inliner(getOperation());
    inliner.run();
    LLVM_DEBUG(llvm::dbgs() << "===--------------------------------------------"
                               "------------------------------===\n");
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createInlinerPass() {
  return std::make_unique<InlinerPass>();
}
