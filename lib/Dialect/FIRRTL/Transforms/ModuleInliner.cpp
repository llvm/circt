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
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Support/Debug.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "firrtl-inliner"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using hw::InnerRefAttr;
using llvm::BitVector;

using InnerRefToNewNameMap = DenseMap<hw::InnerRefAttr, StringAttr>;

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
  hw::HierPathOp nla;

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

  /// Indicates if the NLA is only used to target a module
  /// (i.e., no ports or operations use this HierPathOp).
  /// This is needed to help determine when the HierPathOp is dead:
  /// if we inline/flatten a module, NLA's targeting (only) that module
  /// are now dead.
  bool moduleOnly = false;

  /// Stores new roots for the NLA.  If this is non-empty, then it indicates
  /// that the NLA should be copied and re-topped using the roots stored here.
  /// This is non-empty when the NLA's root is inlined and the original NLA
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
  MutableNLA(hw::HierPathOp nla, CircuitNamespace *circuitNamespace)
      : nla(nla), circuitNamespace(circuitNamespace),
        inlinedSymbols(BitVector(nla.getNamepath().size(), true)),
        size(nla.getNamepath().size()) {
    for (size_t i = 0, e = size; i != e; ++i)
      symIdx.insert({nla.modPart(i), i});
  }

  /// This default, erroring constructor exists because the pass uses
  /// `DenseMap<Attribute, MutableNLA>`.  `DenseMap` requires a default
  /// constructor for the value type because its `[]` operator (which returns a
  /// reference) must default construct the value type for a non-existent key.
  /// This default constructor is never supposed to be used because the pass
  /// prepopulates a `DenseMap<Attribute, MutableNLA>` before it runs and
  /// thereby guarantees that `[]` will always hit and never need to use the
  /// default constructor.
  MutableNLA() {
    llvm_unreachable(
        "the default constructor for MutableNLA should never be used");
  }

  /// Set the state of the mutable NLA to indicate that the _original_ NLA
  /// should be removed when updates are applied.
  void markDead() { dead = true; }

  /// Set the state of the mutable NLA to indicate the only target is a module.
  void markModuleOnly() { moduleOnly = true; }

  /// Return the original NLA that this was pointing at.
  hw::HierPathOp getNLA() { return nla; }

  /// Writeback updates accumulated in this MutableNLA to the IR.  This method
  /// should only ever be called once and, if a writeback occurrs, the
  /// MutableNLA is NOT updated for further use.  Interacting with the
  /// MutableNLA in any way after calling this method may result in crashes.
  /// (This is done to save unnecessary state cleanup of a pass-private
  /// utility.)
  hw::HierPathOp applyUpdates() {
    // Delete an NLA which is either dead or has been made local.
    if (isLocal() || isDead()) {
      nla.erase();
      return nullptr;
    }

    // The NLA was never updated, just return the NLA and do not writeback
    // anything.
    if (inlinedSymbols.all() && newTops.empty() && flattenPoint == -1 &&
        renames.empty())
      return nla;

    // The NLA has updates.  Generate a new NLA with the same symbol and delete
    // the original NLA.
    OpBuilder b(nla);
    auto writeBack = [&](StringAttr root, StringAttr sym) -> hw::HierPathOp {
      SmallVector<Attribute> namepath;
      StringAttr lastMod;

      // Root of the namepath.
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

      auto hp = b.create<hw::HierPathOp>(b.getUnknownLoc(), sym,
                                         b.getArrayAttr(namepath));
      hp.setVisibility(nla.getVisibility());
      return hp;
    };

    hw::HierPathOp last;
    assert(!dead || !newTops.empty());
    if (!dead)
      last = writeBack(nla.root(), nla.getNameAttr());
    for (auto root : newTops)
      last = writeBack(root.getRoot(), root.getTarget());

    nla.erase();
    return last;
  }

  void dump() {
    llvm::errs() << "  - orig:           " << nla << "\n"
                 << "    new:            " << *this << "\n"
                 << "    dead:           " << dead << "\n"
                 << "    isDead:         " << isDead() << "\n"
                 << "    isModuleOnly:   " << isModuleOnly() << "\n"
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
      writeOne(a.getRoot(), a.getTarget());
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

  /// Returns true if this NLA targets only a module.
  bool isModuleOnly() { return moduleOnly; }

  /// Returns true if this NLA is local.  For this to be local, every module
  /// after the root (up to the flatten point or the end) must be inlined.  The
  /// root is never truly inlined as inlining the root just sets a new root.
  bool isLocal() {
    unsigned end = flattenPoint > -1 ? flattenPoint + 1 : inlinedSymbols.size();
    return inlinedSymbols.find_first_in(1, end) == -1;
  }

  /// Return true if this NLA has a root that originates from a specific module.
  bool hasRoot(FModuleLike mod) {
    return (isDead() && nla.root() == mod.getModuleNameAttr()) ||
           rootSet.contains(mod.getModuleNameAttr());
  }

  /// Return true if either this NLA is rooted at modName, or is retoped to it.
  bool hasRoot(StringAttr modName) {
    return (nla.root() == modName) || rootSet.contains(modName);
  }

  /// Mark a module as inlined.  This will remove it from the NLA.
  void inlineModule(FModuleOp module) {
    auto sym = module.getNameAttr();
    assert(sym != nla.root() && "unable to inline the root module");
    assert(symIdx.count(sym) && "module is not in the symIdx map");
    auto idx = symIdx[sym];
    inlinedSymbols.reset(idx);
    // If we inlined the last module in the path and the NLA targets only that
    // module, then this NLA is dead.
    if (idx == size - 1 && moduleOnly)
      markDead();
  }

  /// Mark a module as flattened.  This has the effect of inlining all of its
  /// children.  Also mark the NLA as dead if the leaf reference of this NLA is
  /// a module and the only target is a module.
  void flattenModule(FModuleOp module) {
    auto sym = module.getNameAttr();
    assert(symIdx.count(sym) && "module is not in the symIdx map");
    auto idx = symIdx[sym] - 1;
    flattenPoint = idx;
    // If the NLA only targets a module and we're flattening the NLA,
    // then the NLA must be dead.  Mark it as such.
    if (moduleOnly)
      markDead();
  }

  StringAttr reTop(FModuleOp module) {
    StringAttr sym = nla.getSymNameAttr();
    if (!newTops.empty())
      sym = StringAttr::get(nla.getContext(),
                            circuitNamespace->newName(sym.getValue()));
    newTops.push_back(InnerRefAttr::get(module.getNameAttr(), sym));
    rootSet.insert(module.getNameAttr());
    symIdx.insert({module.getNameAttr(), 0});
    markDead();
    return sym;
  }

  ArrayRef<InnerRefAttr> getAdditionalSymbols() { return ArrayRef(newTops); }

  void setInnerSym(Attribute module, StringAttr innerSym) {
    assert(symIdx.count(module) && "Mutable NLA did not contain symbol");
    assert(!renames.count(module) && "Module already renamed");
    renames.insert({module, innerSym});
  }
};
} // namespace

/// This function is used after inlining a module, to handle the conversion
/// between module ports and instance results. This maps each wire to the
/// result of the instance operation.  When future operations are cloned from
/// the current block, they will use the value of the wire instead of the
/// instance results.
static void mapResultsToWires(IRMapping &mapper, SmallVectorImpl<Value> &wires,
                              InstanceOp instance) {
  for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i) {
    auto result = instance.getResult(i);
    auto wire = wires[i];
    mapper.map(result, wire);
  }
}

/// Process each operation, updating InnerRefAttr's using the specified map
/// and the given name as the containing IST of the mapped-to sym names.
static void replaceInnerRefUsers(ArrayRef<Operation *> newOps,
                                 const InnerRefToNewNameMap &map,
                                 StringAttr istName) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](hw::InnerRefAttr innerRef) {
    auto it = map.find(innerRef);
    // TODO: what to do with users that aren't local (or not mapped?).
    assert(it != map.end());

    return std::pair{hw::InnerRefAttr::get(istName, it->second),
                     WalkResult::skip()};
  });
  llvm::for_each(newOps,
                 [&](auto *op) { replacer.recursivelyReplaceElementsIn(op); });
}

/// Generate and creating map entries for new inner symbol based on old one
/// and an appropriate namespace for creating unique names for each.
static hw::InnerSymAttr uniqueInNamespace(hw::InnerSymAttr old,
                                          InnerRefToNewNameMap &map,
                                          hw::InnerSymbolNamespace &ns,
                                          StringAttr istName) {
  if (!old || old.empty())
    return old;

  bool anyChanged = false;

  SmallVector<hw::InnerSymPropertiesAttr> newProps;
  auto *context = old.getContext();
  for (auto &prop : old) {
    auto newSym = ns.newName(prop.getName().strref());
    if (newSym == prop.getName()) {
      newProps.push_back(prop);
      continue;
    }
    auto newSymStrAttr = StringAttr::get(context, newSym);
    auto newProp = hw::InnerSymPropertiesAttr::get(
        context, newSymStrAttr, prop.getFieldID(), prop.getSymVisibility());
    anyChanged = true;
    newProps.push_back(newProp);
  }

  auto newSymAttr = anyChanged ? hw::InnerSymAttr::get(context, newProps) : old;

  for (auto [oldProp, newProp] : llvm::zip(old, newSymAttr)) {
    assert(oldProp.getFieldID() == newProp.getFieldID());
    // Map InnerRef to this inner sym -> new inner sym.
    map[hw::InnerRefAttr::get(istName, oldProp.getName())] = newProp.getName();
  }

  return newSymAttr;
}

//===----------------------------------------------------------------------===//
// Inliner
//===----------------------------------------------------------------------===//

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
  Inliner(CircuitOp circuit, SymbolTable &symbolTable);

  /// Run the inliner.
  void run();

private:
  /// Inlining context, one per module being inlined into.
  /// Cleans up backedges on destruction.
  struct ModuleInliningContext {
    ModuleInliningContext(FModuleOp module)
        : module(module), modNamespace(module), b(module.getContext()) {}
    /// Top-level module for current inlining task.
    FModuleOp module;
    /// Namespace for generating new names in `module`.
    hw::InnerSymbolNamespace modNamespace;
    /// Builder, insertion point into module.
    OpBuilder b;
  };

  /// One inlining level, created for each instance inlined or flattened.
  /// All inner symbols renamed are recorded in relocatedInnerSyms,
  /// and new operations in newOps.  On destruction newOps are fixed up.
  struct InliningLevel {
    InliningLevel(ModuleInliningContext &mic, FModuleOp childModule)
        : mic(mic), childModule(childModule) {}

    /// Top-level inlining context.
    ModuleInliningContext &mic;
    /// Map of inner-refs to the new inner sym.
    InnerRefToNewNameMap relocatedInnerSyms;
    /// All operations cloned are tracked here.
    SmallVector<Operation *> newOps;
    /// Wires and other values introduced for ports.
    SmallVector<Value> wires;
    /// The module being inlined (this "level").
    FModuleOp childModule;
    /// The explicit debug scope of the inlined instance.
    Value debugScope;

    ~InliningLevel() {
      replaceInnerRefUsers(newOps, relocatedInnerSyms,
                           mic.module.getNameAttr());
    }
  };

  /// Returns true if the NLA matches the current path.  This will only return
  /// false if there is a mismatch indicating that the NLA definitely is
  /// referring to some other path.
  bool doesNLAMatchCurrentPath(hw::HierPathOp nla);

  /// Rename an operation and unique any symbols it has.
  /// Returns true iff symbol was changed.
  bool rename(StringRef prefix, Operation *op, InliningLevel &il);

  /// Rename an InstanceOp and unique any symbols it has.
  /// Requires old and new operations to appropriately update the `HierPathOp`'s
  /// that it participates in.
  bool renameInstance(StringRef prefix, InliningLevel &il, InstanceOp oldInst,
                      InstanceOp newInst,
                      const DenseMap<Attribute, Attribute> &symbolRenames);

  /// Clone and rename an operation.  Insert the operation into the inlining
  /// level.
  void cloneAndRename(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                      Operation &op,
                      const DenseMap<Attribute, Attribute> &symbolRenames,
                      const DenseSet<Attribute> &localSymbols);

  /// Rewrite the ports of a module as wires.  This is similar to
  /// cloneAndRename, but operating on ports.
  /// Wires are added to il.wires.
  void mapPortsToWires(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                       const DenseSet<Attribute> &localSymbols);

  /// Returns true if the operation is annotated to be flattened.
  bool shouldFlatten(Operation *op);

  /// Returns true if the operation is annotated to be inlined.
  bool shouldInline(Operation *op);

  /// Flattens a target module into the insertion point of the builder,
  /// renaming all operations using the prefix.  This clones all operations from
  /// the target, and does not trigger inlining on the target itself.
  void flattenInto(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                   DenseSet<Attribute> localSymbols);

  /// Inlines a target module into the insertion point of the builder,
  /// prefixing all operations with prefix.  This clones all operations from
  /// the target, and does not trigger inlining on the target itself.
  void inlineInto(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                  DenseMap<Attribute, Attribute> &symbolRenames);

  /// Recursively flatten all instances in a module.
  void flattenInstances(FModuleOp module);

  /// Inline any instances in the module which were marked for inlining.
  void inlineInstances(FModuleOp module);

  /// Create a debug scope for an inlined instance at the current insertion
  /// point of the `il.mic` builder.
  void createDebugScope(InliningLevel &il, InstanceOp instance,
                        Value parentScope = {});

  /// Identify all module-only NLA's, marking their MutableNLA's accordingly.
  void identifyNLAsTargetingOnlyModules();

  /// Populate the activeHierpaths with the HierPaths that are active given the
  /// current hierarchy. This is the set of HierPaths that were active in the
  /// parent, and on the current instance. Also HierPaths that are rooted at
  /// this module are also added to the active set.
  void setActiveHierPaths(StringAttr moduleName, StringAttr instInnerSym) {
    llvm::SmallVector<StringAttr> path{moduleName};
    if (instInnerSym)
      path.push_back(instInnerSym);
    auto &instPaths = instOpHierPaths[InnerRefAttr::get(path)];
    if (currentPath.empty()) {
      activeHierpaths.insert(instPaths.begin(), instPaths.end());
      return;
    }
    DenseSet<StringAttr> hPaths(instPaths.begin(), instPaths.end());
    // Only the hierPaths that this instance participates in, and is active in
    // the current path must be kept active for the child modules.
    llvm::set_intersect(activeHierpaths, hPaths);
    // Also, the nlas, that have current instance as the top must be added to
    // the active set.
    for (auto hPath : instPaths)
      if (nlaMap[hPath].hasRoot(moduleName))
        activeHierpaths.insert(hPath);
  }

  CircuitOp circuit;
  MLIRContext *context;

  // A symbol table with references to each module in a circuit.
  SymbolTable &symbolTable;

  /// The set of live modules.  Anything not recorded in this set will be
  /// removed by dead code elimination.
  DenseSet<Operation *> liveModules;

  /// Worklist of modules to process for inlining or flattening.
  SmallVector<FModuleOp, 16> worklist;

  /// A mapping of NLA symbol name to mutable NLA.
  DenseMap<Attribute, MutableNLA> nlaMap;

  /// A mapping of module names to NLA symbols that originate from that module.
  DenseMap<Attribute, SmallVector<Attribute>> rootMap;

  /// The current instance path.  This is a pair<ModuleName, InstanceName>.
  /// This is used to distinguish if a non-local annotation applies to the
  /// current instance or not.
  SmallVector<std::pair<Attribute, Attribute>> currentPath;

  DenseSet<StringAttr> activeHierpaths;

  /// Record the HierPathOps that each InstanceOp participates in. This is a map
  /// from the InnerRefAttr to the list of HierPathOp names. The InnerRefAttr
  /// corresponds to the InstanceOp.
  DenseMap<InnerRefAttr, SmallVector<StringAttr>> instOpHierPaths;

  /// The debug scopes created for inlined instances. Scopes that are unused
  /// after inlining will be deleted again.
  SmallVector<debug::ScopeOp> debugScopes;
};
} // namespace

/// Check if the NLA applies to our instance path. This works by verifying the
/// instance paths backwards starting from the current module. We drop the back
/// element from the NLA because it obviously matches the current operation.
bool Inliner::doesNLAMatchCurrentPath(hw::HierPathOp nla) {
  return (activeHierpaths.find(nla.getSymNameAttr()) != activeHierpaths.end());
}

/// If this operation or any child operation has a name, add the prefix to that
/// operation's name.  If the operation has any inner symbols, make sure that
/// these are unique in the namespace.  Record renamed inner symbols
/// in relocatedInnerSyms map for renaming local users.
bool Inliner::rename(StringRef prefix, Operation *op, InliningLevel &il) {
  // Debug operations with implicit module scope now need an explicit scope,
  // since inlining has destroyed the module whose scope they implicitly used.
  auto updateDebugScope = [&](auto op) {
    if (!op.getScope())
      op.getScopeMutable().assign(il.debugScope);
  };
  if (auto varOp = dyn_cast<debug::VariableOp>(op))
    return updateDebugScope(varOp), false;
  if (auto scopeOp = dyn_cast<debug::ScopeOp>(op))
    return updateDebugScope(scopeOp), false;

  // Add a prefix to things that has a "name" attribute.  We don't prefix
  // memories since it will affect the name of the generated module.
  // TODO: We should find a way to prefix the instance of a memory module.
  if (!isa<MemOp, SeqMemOp, CombMemOp, MemoryPortOp>(op)) {
    if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
      op->setAttr("name", StringAttr::get(op->getContext(),
                                          (prefix + nameAttr.getValue())));
  }

  // If the operation has an inner symbol, ensure that it is unique.  Record
  // renames for any NLAs that this participates in if the symbol was renamed.
  auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(op);
  if (!symOp)
    return false;
  auto oldSymAttr = symOp.getInnerSymAttr();
  auto newSymAttr =
      uniqueInNamespace(oldSymAttr, il.relocatedInnerSyms, il.mic.modNamespace,
                        il.childModule.getNameAttr());

  if (!newSymAttr)
    return false;

  // If there's a symbol on the root and it changed, do NLA work.
  if (auto newSymStrAttr = newSymAttr.getSymName();
      newSymStrAttr && newSymStrAttr != oldSymAttr.getSymName()) {
    for (Annotation anno : AnnotationSet(op)) {
      auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
      if (!sym)
        continue;
      // If this is a breadcrumb, we update the annotation path
      // unconditionally. If this is the leaf of the NLA, we need to make
      // sure we only update the annotation if the current path matches the
      // NLA. This matters when the same module is inlined twice and the NLA
      // only applies to one of them.
      auto &mnla = nlaMap[sym.getAttr()];
      if (!doesNLAMatchCurrentPath(mnla.getNLA()))
        continue;
      mnla.setInnerSym(il.mic.module.getModuleNameAttr(), newSymStrAttr);
    }
  }

  symOp.setInnerSymbolAttr(newSymAttr);

  return newSymAttr != oldSymAttr;
}

bool Inliner::renameInstance(
    StringRef prefix, InliningLevel &il, InstanceOp oldInst, InstanceOp newInst,
    const DenseMap<Attribute, Attribute> &symbolRenames) {
  // TODO: There is currently no good way to annotate an explicit parent scope
  // on instances. Just emit a note in debug runs until this is resolved.
  LLVM_DEBUG({
    if (il.debugScope)
      llvm::dbgs() << "Discarding parent debug scope for " << oldInst << "\n";
  });

  // Add this instance to the activeHierpaths. This ensures that NLAs that this
  // instance participates in will be updated correctly.
  auto parentActivePaths = activeHierpaths;
  assert(oldInst->getParentOfType<FModuleOp>() == il.childModule);
  if (auto instSym = getInnerSymName(oldInst))
    setActiveHierPaths(oldInst->getParentOfType<FModuleOp>().getNameAttr(),
                       instSym);
  // List of HierPathOps that are valid based on the InstanceOp being inlined
  // and the InstanceOp which is being replaced after inlining. That is the set
  // of HierPathOps that is common between these two.
  SmallVector<StringAttr> validHierPaths;
  auto oldParent = oldInst->getParentOfType<FModuleOp>().getNameAttr();
  auto oldInstSym = getInnerSymName(oldInst);

  if (oldInstSym) {
    // Get the innerRef to the original InstanceOp that is being inlined here.
    // For all the HierPathOps that the instance being inlined participates
    // in.
    auto oldInnerRef = InnerRefAttr::get(oldParent, oldInstSym);
    for (auto old : instOpHierPaths[oldInnerRef]) {
      // If this HierPathOp is valid at the inlining context, where the
      // instance is being inlined at. That is, if it exists in the
      // activeHierpaths.
      if (activeHierpaths.find(old) != activeHierpaths.end())
        validHierPaths.push_back(old);
      else
        // The HierPathOp could have been renamed, check for the other retoped
        // names, if they are active at the inlining context.
        for (auto additionalSym : nlaMap[old].getAdditionalSymbols())
          if (activeHierpaths.find(additionalSym.getTarget()) !=
              activeHierpaths.end()) {
            validHierPaths.push_back(old);
            break;
          }
    }
  }

  assert(getInnerSymName(newInst) == oldInstSym);

  // Do the renaming, creating new symbol as needed.
  auto symbolChanged = rename(prefix, newInst, il);

  // If the symbol changed, update instOpHierPaths accordingly.
  auto newSymAttr = getInnerSymName(newInst);
  if (symbolChanged) {
    assert(newSymAttr);
    // The InstanceOp is renamed, so move the HierPathOps to the new
    // InnerRefAttr.
    auto newInnerRef = InnerRefAttr::get(
        newInst->getParentOfType<FModuleOp>().getNameAttr(), newSymAttr);
    instOpHierPaths[newInnerRef] = validHierPaths;
    // Update the innerSym for all the affected HierPathOps.
    for (auto nla : instOpHierPaths[newInnerRef]) {
      if (!nlaMap.count(nla))
        continue;
      auto &mnla = nlaMap[nla];
      mnla.setInnerSym(newInnerRef.getRoot(), newSymAttr);
    }
  }

  if (newSymAttr) {
    auto innerRef = InnerRefAttr::get(
        newInst->getParentOfType<FModuleOp>().getNameAttr(), newSymAttr);
    SmallVector<StringAttr> &nlaList = instOpHierPaths[innerRef];
    // Now rename the Updated HierPathOps that this InstanceOp participates in.
    for (const auto &en : llvm::enumerate(nlaList)) {
      auto oldNLA = en.value();
      if (auto newSym = symbolRenames.lookup(oldNLA))
        nlaList[en.index()] = cast<StringAttr>(newSym);
    }
  }
  activeHierpaths = std::move(parentActivePaths);
  return symbolChanged;
}

/// This function is used before inlining a module, to handle the conversion
/// between module ports and instance results. For every port in the target
/// module, create a wire, and assign a mapping from each module port to the
/// wire. When the body of the module is cloned, the value of the wire will be
/// used instead of the module's ports.
void Inliner::mapPortsToWires(StringRef prefix, InliningLevel &il,
                              IRMapping &mapper,
                              const DenseSet<Attribute> &localSymbols) {
  auto target = il.childModule;
  auto portInfo = target.getPorts();
  for (unsigned i = 0, e = target.getNumPorts(); i < e; ++i) {
    auto arg = target.getArgument(i);
    // Get the type of the wire.
    auto type = type_cast<FIRRTLType>(arg.getType());

    // Compute new symbols if needed.
    auto oldSymAttr = portInfo[i].sym;
    auto newSymAttr =
        uniqueInNamespace(oldSymAttr, il.relocatedInnerSyms,
                          il.mic.modNamespace, target.getNameAttr());

    StringAttr newRootSymName, oldRootSymName;
    if (oldSymAttr)
      oldRootSymName = oldSymAttr.getSymName();
    if (newSymAttr)
      newRootSymName = newSymAttr.getSymName();

    SmallVector<Attribute> newAnnotations;
    for (auto anno : AnnotationSet::forPort(target, i)) {
      // If the annotation is not non-local, copy it to the clone.
      if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        auto &mnla = nlaMap[sym.getAttr()];
        // If the NLA does not match the path, we don't want to copy it over.
        if (!doesNLAMatchCurrentPath(mnla.getNLA()))
          continue;
        // Update any NLAs with the new symbol name.
        // This does not handle per-field symbols used in NLA's.
        if (oldRootSymName != newRootSymName)
          mnla.setInnerSym(il.mic.module.getModuleNameAttr(), newRootSymName);
        // If all paths of the NLA have been inlined, make it local.
        if (mnla.isLocal() || localSymbols.count(sym.getAttr()))
          anno.removeMember("circt.nonlocal");
      }
      newAnnotations.push_back(anno.getAttr());
    }

    Value wire =
        il.mic.b
            .create<WireOp>(
                target.getLoc(), type,
                StringAttr::get(context, (prefix + portInfo[i].getName())),
                NameKindEnumAttr::get(context, NameKindEnum::DroppableName),
                ArrayAttr::get(context, newAnnotations), newSymAttr,
                /*forceable=*/UnitAttr{})
            .getResult();
    il.wires.push_back(wire);
    mapper.map(arg, wire);
  }
}

/// Clone an operation, mapping used values and results with the mapper, and
/// apply the prefix to the name of the operation. This will clone to the
/// insert point of the builder.  Insert the operation into the level.
void Inliner::cloneAndRename(
    StringRef prefix, InliningLevel &il, IRMapping &mapper, Operation &op,
    const DenseMap<Attribute, Attribute> &symbolRenames,
    const DenseSet<Attribute> &localSymbols) {
  // Strip any non-local annotations which are local.
  AnnotationSet oldAnnotations(&op);
  SmallVector<Annotation> newAnnotations;
  for (auto anno : oldAnnotations) {
    // If the annotation is not non-local, it will apply to all inlined
    // instances of this op. Add it to the cloned op.
    if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
      // Retrieve the corresponding NLA.
      auto &mnla = nlaMap[sym.getAttr()];
      // If the NLA does not match the path we don't want to copy it over.
      if (!doesNLAMatchCurrentPath(mnla.getNLA()))
        continue;
      // The NLA has become local, rewrite the annotation to be local.
      if (mnla.isLocal() || localSymbols.count(sym.getAttr()))
        anno.removeMember("circt.nonlocal");
    }
    // Attach this annotation to the cloned operation.
    newAnnotations.push_back(anno);
  }

  // Clone and rename.
  auto *newOp = il.mic.b.clone(op, mapper);

  // Rename the new operation and any contained operations.
  // (add prefix to it, if named, and unique-ify symbol, updating NLA's).
  op.walk<mlir::WalkOrder::PreOrder>([&](Operation *origOp) {
    auto *newOpToRename = mapper.lookup(origOp);
    assert(newOpToRename);
    // TODO: If want to work before ExpandWhen's, more work needed!
    // Handle what we can for now.
    assert((origOp == &op || !isa<InstanceOp>(origOp)) &&
           "Cannot handle instances not at top-level");

    // Instances require extra handling to update HierPathOp's if their symbols
    // change.
    if (auto oldInst = dyn_cast<InstanceOp>(origOp))
      renameInstance(prefix, il, oldInst, cast<InstanceOp>(newOpToRename),
                     symbolRenames);
    else
      rename(prefix, newOpToRename, il);
  });

  // We want to avoid attaching an empty annotation array on to an op that
  // never had an annotation array in the first place.
  if (!newAnnotations.empty() || !oldAnnotations.empty())
    AnnotationSet(newAnnotations, context).applyToOperation(newOp);

  il.newOps.push_back(newOp);
}

bool Inliner::shouldFlatten(Operation *op) {
  return AnnotationSet(op).hasAnnotation(flattenAnnoClass);
}

bool Inliner::shouldInline(Operation *op) {
  return AnnotationSet(op).hasAnnotation(inlineAnnoClass);
}

// NOLINTNEXTLINE(misc-no-recursion)
void Inliner::flattenInto(StringRef prefix, InliningLevel &il,
                          IRMapping &mapper, DenseSet<Attribute> localSymbols) {
  auto target = il.childModule;
  auto moduleName = target.getNameAttr();
  DenseMap<Attribute, Attribute> symbolRenames;
  for (auto &op : *target.getBodyBlock()) {
    // If it's not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      cloneAndRename(prefix, il, mapper, op, symbolRenames, localSymbols);
      continue;
    }

    // If it's not a regular module we can't inline it. Mark it as live.
    auto *module = symbolTable.lookup(instance.getModuleName());
    auto childModule = dyn_cast<FModuleOp>(module);
    if (!childModule) {
      liveModules.insert(module);

      cloneAndRename(prefix, il, mapper, op, symbolRenames, localSymbols);
      continue;
    }

    // Add any NLAs which start at this instance to the localSymbols set.
    // Anything in this set will be made local during the recursive flattenInto
    // walk.
    llvm::set_union(localSymbols, rootMap[childModule.getNameAttr()]);
    auto instInnerSym = getInnerSymName(instance);
    auto parentActivePaths = activeHierpaths;
    setActiveHierPaths(moduleName, instInnerSym);
    currentPath.emplace_back(moduleName, instInnerSym);

    InliningLevel childIL(il.mic, childModule);
    createDebugScope(childIL, instance, il.debugScope);

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, childIL, mapper, localSymbols);
    mapResultsToWires(mapper, childIL.wires, instance);

    // Unconditionally flatten all instance operations.
    flattenInto(nestedPrefix, childIL, mapper, localSymbols);
    currentPath.pop_back();
    activeHierpaths = parentActivePaths;
  }
}

void Inliner::flattenInstances(FModuleOp module) {
  auto moduleName = module.getNameAttr();
  ModuleInliningContext mic(module);

  for (auto &op : llvm::make_early_inc_range(*module.getBodyBlock())) {
    // If it's not an instance op, skip it.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance)
      continue;

    // If it's not a regular module we can't inline it. Mark it as live.
    auto *targetModule = symbolTable.lookup(instance.getModuleName());
    auto target = dyn_cast<FModuleOp>(targetModule);
    if (!target) {
      liveModules.insert(targetModule);
      continue;
    }
    if (auto instSym = getInnerSymName(instance)) {
      auto innerRef = InnerRefAttr::get(moduleName, instSym);
      // Preorder update of any non-local annotations this instance participates
      // in.  This needs to happen _before_ visiting modules so that internal
      // non-local annotations can be deleted if they are now local.
      for (auto targetNLA : instOpHierPaths[innerRef]) {
        nlaMap[targetNLA].flattenModule(target);
      }
    }

    // Add any NLAs which start at this instance to the localSymbols set.
    // Anything in this set will be made local during the recursive flattenInto
    // walk.
    DenseSet<Attribute> localSymbols;
    llvm::set_union(localSymbols, rootMap[target.getNameAttr()]);
    auto instInnerSym = getInnerSymName(instance);
    auto parentActivePaths = activeHierpaths;
    setActiveHierPaths(moduleName, instInnerSym);
    currentPath.emplace_back(moduleName, instInnerSym);

    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    IRMapping mapper;
    mic.b.setInsertionPoint(instance);

    InliningLevel il(mic, target);
    createDebugScope(il, instance);

    auto nestedPrefix = (instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, il, mapper, localSymbols);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(il.wires[i]);

    // Recursively flatten the target module.
    flattenInto(nestedPrefix, il, mapper, localSymbols);
    currentPath.pop_back();
    activeHierpaths = parentActivePaths;

    // Erase the replaced instance.
    instance.erase();
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
void Inliner::inlineInto(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                         DenseMap<Attribute, Attribute> &symbolRenames) {
  auto target = il.childModule;
  auto inlineToParent = il.mic.module;
  auto moduleName = target.getNameAttr();
  // Inline everything in the module's body.
  for (auto &op : *target.getBodyBlock()) {
    // If it's not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      cloneAndRename(prefix, il, mapper, op, symbolRenames, {});
      continue;
    }

    // If it's not a regular module we can't inline it. Mark it as live.
    auto *module = symbolTable.lookup(instance.getModuleName());
    auto childModule = dyn_cast<FModuleOp>(module);
    if (!childModule) {
      liveModules.insert(module);
      cloneAndRename(prefix, il, mapper, op, symbolRenames, {});
      continue;
    }

    // If we aren't inlining the target, add it to the work list.
    if (!shouldInline(childModule)) {
      if (liveModules.insert(childModule).second) {
        worklist.push_back(childModule);
      }
      cloneAndRename(prefix, il, mapper, op, symbolRenames, {});
      continue;
    }

    auto toBeFlattened = shouldFlatten(childModule);
    if (auto instSym = getInnerSymName(instance)) {
      auto innerRef = InnerRefAttr::get(moduleName, instSym);
      // Preorder update of any non-local annotations this instance participates
      // in.  This needs to happen _before_ visiting modules so that internal
      // non-local annotations can be deleted if they are now local.
      for (auto sym : instOpHierPaths[innerRef]) {
        if (toBeFlattened)
          nlaMap[sym].flattenModule(childModule);
        else
          nlaMap[sym].inlineModule(childModule);
      }
    }

    // The InstanceOp `instance` might not have a symbol, if it does not
    // participate in any HierPathOp. But the reTop might add a symbol to it, if
    // a HierPathOp is added to this Op. If we're about to inline a module that
    // contains a non-local annotation that starts at that module, then we need
    // to both update the mutable NLA to indicate that this has a new top and
    // add an annotation on the instance saying that this now participates in
    // this new NLA.
    DenseMap<Attribute, Attribute> symbolRenames;
    if (!rootMap[childModule.getNameAttr()].empty()) {
      for (auto sym : rootMap[childModule.getNameAttr()]) {
        auto &mnla = nlaMap[sym];
        // Retop to the new parent, which is the topmost module (and not
        // immediate parent) in case of recursive inlining.
        sym = mnla.reTop(inlineToParent);
        StringAttr instSym = getInnerSymName(instance);
        if (!instSym) {
          instSym = StringAttr::get(
              context, il.mic.modNamespace.newName(instance.getName()));
          instance.setInnerSymAttr(hw::InnerSymAttr::get(instSym));
        }
        instOpHierPaths[InnerRefAttr::get(moduleName, instSym)].push_back(
            cast<StringAttr>(sym));
        // TODO: Update any symbol renames which need to be used by the next
        // call of inlineInto.  This will then check each instance and rename
        // any symbols appropriately for that instance.
        symbolRenames.insert({mnla.getNLA().getNameAttr(), sym});
      }
    }
    auto instInnerSym = getInnerSymName(instance);
    auto parentActivePaths = activeHierpaths;
    setActiveHierPaths(moduleName, instInnerSym);
    // This must be done after the reTop, since it might introduce an innerSym.
    currentPath.emplace_back(moduleName, instInnerSym);

    InliningLevel childIL(il.mic, childModule);
    createDebugScope(childIL, instance, il.debugScope);

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, childIL, mapper, {});
    mapResultsToWires(mapper, childIL.wires, instance);

    // Inline the module, it can be marked as flatten and inline.
    if (toBeFlattened) {
      flattenInto(nestedPrefix, childIL, mapper, {});
    } else {
      inlineInto(nestedPrefix, childIL, mapper, symbolRenames);
    }
    currentPath.pop_back();
    activeHierpaths = parentActivePaths;
  }
}

void Inliner::inlineInstances(FModuleOp module) {
  // Generate a namespace for this module so that we can safely inline symbols.
  auto moduleName = module.getNameAttr();

  SmallVector<Value> wires;
  ModuleInliningContext mic(module);

  for (auto &op : llvm::make_early_inc_range(*module.getBodyBlock())) {
    // If it's not an instance op, skip it.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance)
      continue;

    // If it's not a regular module we can't inline it. Mark it as live.
    auto *childModule = symbolTable.lookup(instance.getModuleName());
    auto target = dyn_cast<FModuleOp>(childModule);
    if (!target) {
      liveModules.insert(childModule);
      continue;
    }

    // If we aren't inlining the target, add it to the work list.
    if (!shouldInline(target)) {
      if (liveModules.insert(target).second) {
        worklist.push_back(target);
      }
      continue;
    }

    auto toBeFlattened = shouldFlatten(target);
    if (auto instSym = getInnerSymName(instance)) {
      auto innerRef = InnerRefAttr::get(moduleName, instSym);
      // Preorder update of any non-local annotations this instance participates
      // in.  This needs to happen _before_ visiting modules so that internal
      // non-local annotations can be deleted if they are now local.
      for (auto sym : instOpHierPaths[innerRef]) {
        if (toBeFlattened)
          nlaMap[sym].flattenModule(target);
        else
          nlaMap[sym].inlineModule(target);
      }
    }

    // The InstanceOp `instance` might not have a symbol, if it does not
    // participate in any HierPathOp. But the reTop might add a symbol to it, if
    // a HierPathOp is added to this Op.
    DenseMap<Attribute, Attribute> symbolRenames;
    if (!rootMap[target.getNameAttr()].empty() && !toBeFlattened) {
      for (auto sym : rootMap[target.getNameAttr()]) {
        auto &mnla = nlaMap[sym];
        sym = mnla.reTop(module);
        StringAttr instSym = getOrAddInnerSym(
            instance, [&](FModuleLike mod) -> hw::InnerSymbolNamespace & {
              return mic.modNamespace;
            });
        instOpHierPaths[InnerRefAttr::get(moduleName, instSym)].push_back(
            cast<StringAttr>(sym));
        // TODO: Update any symbol renames which need to be used by the next
        // call of inlineInto.  This will then check each instance and rename
        // any symbols appropriately for that instance.
        symbolRenames.insert({mnla.getNLA().getNameAttr(), sym});
      }
    }
    auto instInnerSym = getInnerSymName(instance);
    auto parentActivePaths = activeHierpaths;
    setActiveHierPaths(moduleName, instInnerSym);
    // This must be done after the reTop, since it might introduce an innerSym.
    currentPath.emplace_back(moduleName, instInnerSym);
    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    IRMapping mapper;
    mic.b.setInsertionPoint(instance);
    auto nestedPrefix = (instance.getName() + "_").str();

    InliningLevel childIL(mic, target);
    createDebugScope(childIL, instance);

    mapPortsToWires(nestedPrefix, childIL, mapper, {});
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(childIL.wires[i]);

    // Inline the module, it can be marked as flatten and inline.
    if (toBeFlattened) {
      flattenInto(nestedPrefix, childIL, mapper, {});
    } else {
      // Recursively inline all the child modules under `parent`, that are
      // marked to be inlined.
      inlineInto(nestedPrefix, childIL, mapper, symbolRenames);
    }
    currentPath.pop_back();
    activeHierpaths = parentActivePaths;

    // Erase the replaced instance.
    instance.erase();
  }
}

void Inliner::createDebugScope(InliningLevel &il, InstanceOp instance,
                               Value parentScope) {
  auto op = il.mic.b.create<debug::ScopeOp>(
      instance.getLoc(), instance.getInstanceNameAttr(),
      instance.getModuleNameAttr().getAttr(), parentScope);
  debugScopes.push_back(op);
  il.debugScope = op;
}

void Inliner::identifyNLAsTargetingOnlyModules() {
  DenseSet<Operation *> nlaTargetedModules;

  // Identify candidate NLA's: those that end in a module
  for (auto &[sym, mnla] : nlaMap) {
    auto nla = mnla.getNLA();
    if (nla.isModule()) {
      auto mod = symbolTable.lookup<FModuleLike>(nla.leafMod());
      assert(mod &&
             "NLA ends in module reference but does not target FModuleLike?");
      nlaTargetedModules.insert(mod);
    }
  }

  // Helper to scan leaf modules for users of NLAs, gathering by symbol names
  auto scanForNLARefs = [&](FModuleLike mod) {
    DenseSet<StringAttr> referencedNLASyms;
    auto scanAnnos = [&](const AnnotationSet &annos) {
      for (auto anno : annos)
        if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
          referencedNLASyms.insert(sym.getAttr());
    };
    // Scan ports
    for (unsigned i = 0, e = mod.getNumPorts(); i != e; ++i)
      scanAnnos(AnnotationSet::forPort(mod, i));

    // Scan operations (and not the module itself):
    // (Walk includes module for lack of simple/generic way to walk body only)
    mod.walk([&](Operation *op) {
      if (op == mod.getOperation())
        return;
      scanAnnos(AnnotationSet(op));

      // Check MemOp and InstanceOp port annotations, special case
      TypeSwitch<Operation *>(op).Case<MemOp, InstanceOp>([&](auto op) {
        for (auto portAnnoAttr : op.getPortAnnotations())
          scanAnnos(AnnotationSet(cast<ArrayAttr>(portAnnoAttr)));
      });
    });

    return referencedNLASyms;
  };

  // Reduction operator
  auto mergeSets = [](auto &&a, auto &&b) {
    a.insert(b.begin(), b.end());
    return std::move(a);
  };

  // Walk modules in parallel, scanning for references to NLA's
  // Gather set of NLA's referenced by each module's ports/operations.
  SmallVector<FModuleLike, 0> mods(nlaTargetedModules.begin(),
                                   nlaTargetedModules.end());
  auto nonModOnlyNLAs =
      transformReduce(circuit->getContext(), mods, DenseSet<StringAttr>{},
                      mergeSets, scanForNLARefs);

  // Mark NLA's that were not referenced as module-only
  for (auto &[_, mnla] : nlaMap) {
    auto nla = mnla.getNLA();
    if (nla.isModule() && !nonModOnlyNLAs.count(nla.getSymNameAttr()))
      mnla.markModuleOnly();
  }
}

Inliner::Inliner(CircuitOp circuit, SymbolTable &symbolTable)
    : circuit(circuit), context(circuit.getContext()),
      symbolTable(symbolTable) {}

void Inliner::run() {
  CircuitNamespace circuitNamespace(circuit);

  // Gather all NLA's, build information about the instance ops used:
  for (auto nla : circuit.getBodyBlock()->getOps<hw::HierPathOp>()) {
    auto mnla = MutableNLA(nla, &circuitNamespace);
    nlaMap.insert({nla.getSymNameAttr(), mnla});
    rootMap[mnla.getNLA().root()].push_back(nla.getSymNameAttr());
    for (auto p : nla.getNamepath())
      if (auto ref = dyn_cast<InnerRefAttr>(p))
        instOpHierPaths[ref].push_back(nla.getSymNameAttr());
  }
  // Mark 'module-only' the NLA's that only target modules.
  // These may be deleted when their module is inlined/flattened.
  identifyNLAsTargetingOnlyModules();

  // Mark the top module as live, so it doesn't get deleted.
  for (auto module : circuit.getOps<FModuleLike>()) {
    if (module.canDiscardOnUseEmpty())
      continue;
    liveModules.insert(module);
    if (isa<FModuleOp>(module))
      worklist.push_back(cast<FModuleOp>(module));
  }

  // If the module is marked for flattening, flatten it. Otherwise, inline
  // every instance marked to be inlined.
  while (!worklist.empty()) {
    auto module = worklist.pop_back_val();
    if (shouldFlatten(module)) {
      flattenInstances(module);
      // Delete the flatten annotation, the transform was performed.
      // Even if visited again in our walk (for inlining),
      // we've just flattened it and so the annotation is no longer needed.
      AnnotationSet::removeAnnotations(module, flattenAnnoClass);
    } else {
      inlineInstances(module);
    }
  }

  // Delete debug scopes that ended up being unused. Erase them in reverse order
  // since scopes at the back may have uses on scopes at the front.
  for (auto scopeOp : llvm::reverse(debugScopes))
    if (scopeOp.use_empty())
      scopeOp.erase();
  debugScopes.clear();

  // Delete all unreferenced modules.  Mark any NLAs that originate from dead
  // modules as also dead.
  for (auto mod : llvm::make_early_inc_range(
           circuit.getBodyBlock()->getOps<FModuleLike>())) {
    if (liveModules.count(mod))
      continue;
    for (auto nla : rootMap[mod.getModuleNameAttr()])
      nlaMap[nla].markDead();
    mod.erase();
  }

  // Remove leftover inline annotations, and check no flatten annotations
  // remain as they should have been processed and removed.
  for (auto mod : circuit.getBodyBlock()->getOps<FModuleLike>()) {
    if (shouldInline(mod)) {
      assert(mod.isPublic() &&
             "non-public module with inline annotation still present");
      AnnotationSet::removeAnnotations(mod, inlineAnnoClass);
    }
    assert(!shouldFlatten(mod) && "flatten annotation found on live module");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "NLA modifications:\n";
    for (auto nla : circuit.getBodyBlock()->getOps<hw::HierPathOp>()) {
      auto &mnla = nlaMap[nla.getNameAttr()];
      mnla.dump();
    }
  });

  // Writeback all NLAs to MLIR.
  for (auto &nla : nlaMap)
    nla.getSecond().applyUpdates();

  // Garbage collect any annotations which are now dead.  Duplicate annotations
  // which are now split.
  for (auto fmodule : circuit.getBodyBlock()->getOps<FModuleOp>()) {
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
        if (newTops.empty() || mnla.hasRoot(fmodule))
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
                 FlatSymbolRefAttr::get(rootAndSym.getTarget())});
          }
          newAnnotations.push_back(DictionaryAttr::get(context, newAnnotation));
        }
      }
      return false;
    };
    fmodule.walk([&](Operation *op) {
      AnnotationSet annotations(op);
      // Early exit to avoid adding an empty annotations attribute to operations
      // which did not previously have annotations.
      if (annotations.empty())
        return;

      // Update annotations on the op.
      newAnnotations.clear();
      annotations.removeAnnotations(processNLAs);
      annotations.addAnnotations(newAnnotations);
      annotations.applyToOperation(op);
    });

    // Update annotations on the ports.
    SmallVector<Attribute> newPortAnnotations;
    for (auto port : fmodule.getPorts()) {
      newAnnotations.clear();
      port.annotations.removeAnnotations(processNLAs);
      port.annotations.addAnnotations(newAnnotations);
      newPortAnnotations.push_back(
          ArrayAttr::get(context, port.annotations.getArray()));
    }
    fmodule->setAttr("portAnnotations",
                     ArrayAttr::get(context, newPortAnnotations));
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InlinerPass : public InlinerBase<InlinerPass> {
  void runOnOperation() override {
    LLVM_DEBUG(debugPassHeader(this) << "\n");
    Inliner inliner(getOperation(), getAnalysis<SymbolTable>());
    inliner.run();
    LLVM_DEBUG(debugFooter() << "\n");
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createInlinerPass() {
  return std::make_unique<InlinerPass>();
}
