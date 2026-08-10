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
// The challenge here is the robust and efficient computation of the final paths
// along all inlining contexts, and ensuring the actual inlining can be done
// directly and reliably.
//
// The actual inlining is done top-down, recursing from each module to clone
// directly everything we've been directed to clone.  This transformation can
// have a number of effects on the annotations on cloned modules and the paths
// used to encode their behavior.
//
// The glossary, phase architecture, and invariants almost exclusively focus
// around ensuring all the possible outcomes are known and ready before the
// inlining actually begins, greatly simplifying its task.
//
// Read the invariants alongside the phases they correspond to, top to bottom.
//
// Glossary:
//
//   VNLA            VirtualNLA.  Defined below.
//   context (VNLA)  one (source hierpath x surviving copy); the planning unit
//   - origSym       the source hw.hierpath's symbol
//   - realizedSym   the hierpath symbol a context is emitted under (P4):
//                   the primary keeps origSym, forks mint fresh names with
//                   canonicalization to reuse equal hierpaths effectively.
//   - wasUsed       an annotation referenced this context, used to gate
//                   fork-context emission.
//
//   Context kinds:
//   - fork          a non-primary context; emitted under a fresh symbol
//                   or folded into a path-equal canonical context
//   - convergent    a fork folded onto a canonical from a different origSym
//   - canonical     the emitted representative of path-equal contexts
//   - duplicate     a context folded onto its canonical context
//   - local         a context whose path collapsed to its terminal alone
//   - ghost         a context P2 minted whose copy P3 never realizes
//
//   Other:
//   - activeNLAs    the active set of contexts at a given inlining level:
//                   narrowed by the recursive walk's path down to the leaf
//   - dangling      a reference or handle kept past the point its target
//                   is erased or renamed
//   - dead-rooted   a hierpath left with no surviving context at all
//   - foreign       an inner-ref in a cloned body naming any module other
//                   than the child being inlined; no update is defined
//
// The pass runs as four phases.  Outputs freeze before the next phase reads
// them; analysis only flows forward.
//
//  * P1  Classify (InliningFacts):
//          Compute per-module classifications (inline/flatten, liveness, ...),
//          and the parent-first order used for the computation.
//  * P2  Plan (NLAPlanner):
//          A `VirtualNLA` per surviving hierpath context, and the routing table
//          of contexts through each instance.  Enumerates contexts
//          approximately with a bottom-up walk, and then top-down precisely
//          refines to the actual root.  This and InliningFacts (P1) leave the
//          IR untouched in all cases.
//  * P3  Clone (Inliner):
//          Top-down walk inlining marked bodies into parents: patching
//          inner-symbol users, filling hops' final symbols, recording
//          nonlocal-annotation ownership.  Writes no annotation.
//  * P4  Write back:
//          Serially canonicalize contexts (minting symbols), rewrite all
//          annotations in parallel (the single writer), then materialize/erase
//          hierpaths serially.
//
// Prerequisites:
//  * Inline/flatten markers are annotations on regular modules.
//    LowerAnnotations attaches them there; hand-written IR is checked in P1.
//  * Hierpath roots must resolve to modules (diagnosed in P2).
//  * The instance graph is required (assumed) to be acyclic.  In assert builds
//    this is tracked and diagnosed.  This is a compiler-wide contract and not
//    worth production cycles (same reason we don't do it in our verifiers).
//
// Diagnosed and rejected:
//  * Inlining an instance sitting under anything but a module or layer block.
//  * Inlining a body with an inner reference to another module's body.
//
// Both fire during the clone walk (P3), folding into walks we already perform.
// The pass fails, but the IR may be left partially inlined.
// Only a P1/P2 rejection guarantees the input is untouched.
//
// The invariants this rests on, referenced by number at their use sites.
// [asserted]/[diagnosed] marks the ones that break loudly.  The rest are
// structural (upheld by construction, no runtime check).
//
// Frozen state and identity (P1/P2):
//  * I1  (frozen-facts)
//        The `ModuleClassification` is frozen after P1.
//  * I2  (pointer-identity)
//        The `VirtualNLA` pool is frozen after P2.
//        P3/P4 identify a context by its raw pointer.
//  * I3  (field-writers)
//        P3 mutates only a hop's `finalSym`.
//        P4 alone writes `realizedSym`, `wasUsed`, and the canonical tables.
//
// Planning tables (P2):
//  * I4  (ordered-ids)
//        Ids are creation-ordered, contiguous per source symbol and per root.
//  * I5  (routing-sorted)
//        Routing-table entries are born id-sorted and duplicate-free.
//        [asserted: NLAPlanner::run]
//  * I6  (active-sorted)
//        Every level's `activeNLAs` is id-sorted.
//  * I7  (stable-keys)
//        Routing keys (original instance ops) outlive every query.
//        Cloned ops are never lookup keys, so no stale-address aliasing.
//
// Path semantics:
//  * I8  (terminal-survives)
//        Terminal hops never evaporate; only instance hops do.
//        [asserted: VirtualNLA::create]
//        A context keeping an instance hop is non-local; one bottomed out at
//        its terminal alone (incl. any one-element source path) is local.
//        There, the annotation localizes onto the op and the path is dropped.
//  * I9  (ghost-contexts)
//        `underFlatten` ORs over parents (any, not all):
//        P2 may mint contexts P3 never realizes.
//        `wasUsed` gates fork emission, not the primary.
//        Retention (I15) prevents a ghost renaming a survivor's symbol.
//
// Clone walk (P3):
//  * I10 (mint-once)
//        Only P3 mints inner symbol names; one inner-sym namespace per module.
//  * I11 (relocation-total)
//        `relocatedInnerSyms` is total over the clones' inner-refs.
//        A miss is a foreign reference.  [diagnosed]
//  * I12 (parents-first)
//        P3 processes parents strictly before children:
//        bodies clone pristine,
//        and each context's leaf op is cloned exactly once.
//  * I13 (activation-vs-ownership)
//        Activation is route-based and can exceed ownership.
//        Every mutation is also gated on the current inlining destination
//        matching the precomputed final module (`finalMod`).
//        (I14, single-writer; filed under write back).
//
// Write back (P4):
//  * I14 (single-writer)
//        P4's parallel rewrite has a single writer per context:
//        the module holding its last hop's `finalMod`.
//  * I15 (retention)
//        Every origSym with a surviving context has exactly one primary
//        claimant (`realizedSym == origSym`),
//        emitted unconditionally.  [asserted: writebackHierPaths]
//        Only a context-less (dead-rooted) origSym is ever erased.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Support/Debug.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TrailingObjects.h"

#define DEBUG_TYPE "firrtl-inliner"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INLINER
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

using hw::InnerRefAttr;
using InnerRefToNewNameMap = DenseMap<hw::InnerRefAttr, StringAttr>;

//===----------------------------------------------------------------------===//
// Module classification (P1)
//===----------------------------------------------------------------------===//

namespace {

/// Return the instance if the pass can splice a body through this operation.
/// Today that is exactly a plain InstanceOp.
///
/// Non-null answers the operation kind and nothing more.
/// Whether the instance is inlined also depends on its target being a regular
/// firrtl.module, and on the inline/flatten marks.
/// Null is definitive: the pass never splices through anything else.
///
/// The operation kind is a conservative stand-in for the precise question,
/// the meet over the operation's possible targets in the instance graph.
/// Under that framing an instance_choice naming one module throughout would be
/// inlinable, and a new instantiation kind slots in by answering that meet.
static InstanceOp getInlinableInstance(Operation *op) {
  return dyn_cast_or_null<InstanceOp>(op);
}

/// Module facts regarding inlining.
struct ModuleInfo {
  /// The module carries an inline annotation.
  bool hasInline : 1;

  /// The module carries a flatten annotation.
  bool hasFlatten : 1;

  /// Does /any/ instantiation path flatten this module?
  bool underFlatten : 1;

  /// Does any instantiation path /not/ flatten this module?
  /// Note: zero-length path counts as well (e.g., public).
  ///
  /// Note: Only used during computation, consumers want isLive.
  bool hasUnflattenedPath : 1;

  /// Will this module exist in the final result?
  bool isLive : 1;

  /// Derived named predicates over the facts above, used in the walk.

  /// This module's regular children remain instantiated: some path keeps it
  /// instantiated and it is not flattening them away.
  bool keepsChildrenInstantiated() const {
    return hasUnflattenedPath && !hasFlatten;
  }

  /// Whether this module's body may be flattened.
  bool mayBeFlattened() const { return underFlatten || hasFlatten; }

  // (No default member initialization of bitfield members until C++20)
  ModuleInfo()
      : hasInline(false), hasFlatten(false), underFlatten(false),
        hasUnflattenedPath(false), isLive(false) {}
};

class InliningFacts {
public:
  using ModuleClassification = DenseMap<Operation *, ModuleInfo>;

private:
  ModuleClassification classification;
  SmallVector<FModuleOp, 0> schedule;

  InliningFacts(ModuleClassification &&classification,
                SmallVector<FModuleOp, 0> &&schedule)
      : classification(std::move(classification)),
        schedule(std::move(schedule)) {}

public:
  /// Analyze circuit and compute per-module inlining facts.
  ///
  /// Returns facts on success, on failure emits diagnostics.
  static FailureOr<InliningFacts> compute(CircuitOp circuit,
                                          InstanceGraph &instanceGraph,
                                          const mlir::SymbolTable &symbolTable);

  /// Get per-module classification information.
  const ModuleInfo &getModuleInfo(FModuleLike fmod) const {
    assert(fmod && "queried with null fmodulelike");
    auto it = classification.find(fmod);
    assert(it != classification.end() && "module not found");
    return it->second;
  }

  /// Get classification for the specified operation, if known.
  std::optional<ModuleInfo> getModuleInfoIfPresent(Operation *op) const {
    auto it = classification.find(op);
    if (it == classification.end())
      return std::nullopt;
    return it->second;
  }

  /// Convenience helpers, projecting out common fields.

  bool isLive(FModuleLike mod) const { return getModuleInfo(mod).isLive; }
  bool hasInline(FModuleLike mod) const { return getModuleInfo(mod).hasInline; }
  bool hasFlatten(FModuleLike mod) const {
    return getModuleInfo(mod).hasFlatten;
  }

  // Convenience helper to project out `isLive` if operation is known, or false.
  bool isKnownLive(Operation *op) const {
    auto ret = getModuleInfoIfPresent(op);
    return ret && ret->isLive;
  }

  /// Regular modules in inverse post-order: parents strictly before children.
  /// Frozen with the facts, it is the order this analysis was computed over.
  /// Entries are op handles: they dangle once modules are erased.
  ArrayRef<FModuleOp> getSchedule() const { return schedule; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const {
    auto &os = llvm::dbgs();
    auto printOne = [&os](StringRef name, ModuleInfo info) {
      os << "@" << name << ": inline=" << info.hasInline
         << " flatten=" << info.hasFlatten
         << " underFlatten=" << info.underFlatten
         << " unflattenedPath=" << info.hasUnflattenedPath
         << " live=" << info.isLive << "\n";
    };
    // Schedule order for regular modules, then the residue sorted by name.
    for (auto module : schedule)
      printOne(module.getModuleName(), classification.lookup(module));
    SmallVector<std::pair<StringRef, ModuleInfo>> rest;
    for (auto &[op, info] : classification)
      if (!isa<FModuleOp>(op))
        rest.emplace_back(cast<FModuleLike>(op).getModuleName(), info);
    llvm::sort(rest, llvm::less_first());
    for (auto &[name, info] : rest)
      printOne(name, info);
  }
#endif
};

} // namespace

FailureOr<InliningFacts>
InliningFacts::compute(CircuitOp circuit, InstanceGraph &instanceGraph,
                       const mlir::SymbolTable &symbolTable) {
  InliningFacts::ModuleClassification classification;
  SmallVector<FModuleOp, 0> schedule;

  // Cache these attributes.
  auto *ctx = circuit.getContext();
  auto inlineAnnoClassAttr = StringAttr::get(ctx, inlineAnnoClass);
  auto flattenAnnoClassAttr = StringAttr::get(ctx, flattenAnnoClass);

  // Find roots of symbol uses within `op` and mark them live.  Root liveness is
  // all that's promised here (and all likely/able to encounter in practice).
  // The symbol portion of inner refs are kept alive through this walk too.
  auto markSymbolUses = [&](Operation &op) -> LogicalResult {
    // If we can't compute uses, bail.
    auto symbolUses = SymbolTable::getSymbolUses(&op);
    if (!symbolUses)
      return op.emitError("cannot analyze symbol uses of this operation");
    for (const auto &use : *symbolUses) {
      auto root = use.getSymbolRef().getRootReference();
      if (auto moduleLike = symbolTable.lookup<FModuleLike>(root)) {
        auto &info = classification[moduleLike];
        info.isLive = true;
        info.hasUnflattenedPath = true;
      }
    }
    return success();
  };

  // The circuit's attributes contain symbol references that are expected to
  // resolve within the circuit, walk those now.  Since the circuit is itself a
  // symbol table this will only walk the attributes.  FYI: These attributes
  // technically are interpreted by MLIR to resolve in the circuit parent's
  // table, but we use them to name operations within the circuit.  None of
  // these are known to point to modules today, check anyway.
  if (failed(markSymbolUses(*circuit.getOperation())))
    return failure();

  for (auto &op : circuit.getOps()) {
    // Initialize module information.  Not order-dependent.
    if (auto module = dyn_cast<FModuleLike>(op)) {
      auto &info = classification[module];
      AnnotationSet anno(module);
      info.hasInline = anno.hasAnnotation(inlineAnnoClassAttr);
      info.hasFlatten = anno.hasAnnotation(flattenAnnoClassAttr);

      // Reject inline/flatten on anything but a regular module (FModuleOp).
      // LowerAnnotations restricts these to FModuleOp; this catches raw IR.
      if (!isa<FModuleOp>(module) && (info.hasInline || info.hasFlatten))
        return emitError(module.getLoc()) << "inline/flatten annotations are "
                                             "only valid on a 'firrtl.module'";

      // Does anything opaque (see getInlinableInstance) instantiate this?
      auto instantiators = instanceGraph.lookup(module)->uses();
      auto opaqueRecIt = llvm::find_if(instantiators, [](InstanceRecord *rec) {
        return !getInlinableInstance(rec->getInstance());
      });
      bool hasOpaqueUse = opaqueRecIt != instantiators.end();

      if (!module.canDiscardOnUseEmpty() || hasOpaqueUse) {
        info.isLive = true;
        info.hasUnflattenedPath = true;
      }
      if (info.hasInline && hasOpaqueUse) {
        auto diag = mlir::emitWarning(module.getLoc())
                    << "module marked inline is also instantiated by an "
                       "operation that cannot be inlined; it is inlined only "
                       "into its 'firrtl.instance' parents and retained";
        diag.attachNote((*opaqueRecIt)->getInstance()->getLoc())
            << "instantiated here";
      }
      continue;
    }

    // Symbol use analysis:

    // Ignore symbol uses in NLAs.
    if (isa<hw::HierPathOp>(op))
      continue;

    if (failed(markSymbolUses(op)))
      return failure();
  }

  // Calculate inlining info top-down.
  instanceGraph.walkInversePostOrder([&](igraph::InstanceGraphNode &node) {
    auto *mod = node.getModule().getOperation();
    assert(isa<FModuleLike>(mod) && "instance graph contains non-fmodulelike");

    // Save IPO over FModuleOp's for later.
    if (auto fmod = dyn_cast<FModuleOp>(mod))
      schedule.push_back(fmod);
    auto &modInfo = classification[mod];

    // Skip if no non-inlined path to this module.
    if (!modInfo.hasUnflattenedPath && !modInfo.underFlatten)
      return;

    for (auto *edge : node) {
      auto *childMod = edge->getTarget()->getModule().getOperation();
      auto &childInfo = classification[childMod];
      bool isRegularModule = isa<FModuleOp>(childMod);
      assert(
          (isRegularModule || !(childInfo.hasInline || childInfo.hasFlatten)) &&
          "non-fmoduleop with inline/flatten annotation");
      if (isRegularModule && modInfo.mayBeFlattened())
        childInfo.underFlatten = true;

      // A non-regular child is never inlined and so it is always instantiated.
      // A regular child does iff this module does and isn't flattening it.
      if (modInfo.keepsChildrenInstantiated() || !isRegularModule)
        childInfo.hasUnflattenedPath = true;

      // Set liveness.
      if (childInfo.hasUnflattenedPath &&
          (!isRegularModule || !childInfo.hasInline))
        childInfo.isLive = true;
    }
  });

  return InliningFacts(std::move(classification), std::move(schedule));
}

//===----------------------------------------------------------------------===//
// NLA Planning (P2)
//===----------------------------------------------------------------------===//

/// Compute the final shape of all NLAs and their routing through the design.
/// All NLAs in the design are converted into one or more VNLA (VirtualNLA),
/// which contain surviving portions of the original NLA as well as new context
/// needed to handle situations where an NLA root is inlined and "forks".

namespace {

/// One hop of a VirtualNLA's surviving path.
///
/// Tracks the original and final module/symbol locations for a hierpath hop.
struct SurvivingHop {
  /// The original module containing this hop.
  StringAttr origMod;
  /// Inner symbol within origMod, when the hop has one.
  ///
  /// A terminal hop's sym may name a non-instance, a module-only terminal hop
  /// has none.
  StringAttr origSym;
  /// The module this hop lands in after inlining.
  StringAttr finalMod;
  /// I3: the only field P3 mutates, set when the walk realizes the hop.
  ///
  /// Everything else on a hop and its VirtualNLA is read-only after P2.
  StringAttr finalSym;
};

/// A virtual NLA representing one surviving hierpath context after inlining.
/// Frozen after P2 (I2); only `finalSym` is mutated by P3 (I3).
class VirtualNLA final : llvm::TrailingObjects<VirtualNLA, SurvivingHop> {
  friend TrailingObjects;

  unsigned numHops;

  VirtualNLA(unsigned id, StringAttr origSym, ArrayRef<SurvivingHop> path)
      : numHops(path.size()), id(id), origSym(origSym) {
    llvm::uninitialized_copy(path, getTrailingObjects());
  }

public:
  /// Unique creation-ordered identifier for this context (I4).
  unsigned id;
  /// The original hierpath symbol this context was derived from.
  StringAttr origSym;
  /// The symbol that this context's hierpath is emitted under, minted by P4.
  /// The first canonical claimant of an origSym keeps it.  Later ones mint
  /// fresh names; duplicates and locals get none.
  StringAttr realizedSym;
  /// Whether this context was referenced by any annotation (I9).  Duplicates
  /// use canonical context symbols.  New hierpaths are emitted iff set.
  bool wasUsed = false;

  static VirtualNLA *create(llvm::BumpPtrAllocator &alloc, unsigned id,
                            StringAttr origSym, ArrayRef<SurvivingHop> path) {
    // Every context keeps at least its terminal hop (I8), so the path is never
    // empty; `back()`/`isLocal()` and the writeback rely on this.  Enforce it
    // once, at the single construction site, not at each use.
    assert(!path.empty() && "a VNLA always keeps its terminal hop (I8)");
    size_t size = totalSizeToAlloc<SurvivingHop>(path.size());
    auto *mem = alloc.Allocate(size, alignof(VirtualNLA));
    return new (mem) VirtualNLA(id, origSym, path);
  }

  bool isLocal() const { return numHops <= 1; }

  ArrayRef<SurvivingHop> getPath() const {
    return {getTrailingObjects(), numHops};
  }

  MutableArrayRef<SurvivingHop> getPathMutable() {
    return {getTrailingObjects(), numHops};
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const {
    llvm::dbgs() << llvm::formatv("    VirtualNLA {0}: origSym @{1}", id,
                                  origSym);
    if (isLocal()) {
      llvm::dbgs() << " -> local\n";
    } else {
      llvm::dbgs() << llvm::formatv(", hops: {0}\n", numHops);
      for (const auto &hop : getPath()) {
        llvm::dbgs() << llvm::formatv(
            "      - {0}::{1} -> {2}::{3}\n", hop.origMod,
            (hop.origSym ? hop.origSym.str() : "*"), hop.finalMod,
            (hop.finalSym ? hop.finalSym.str() : "(TBD)"));
      }
    }
  }
#endif
};

static_assert(std::is_trivially_destructible_v<VirtualNLA>,
              "VirtualNLA is arena-allocated; destructors never run");

} // namespace

/// Context collections are ordered by creation id throughout (I4/I5/I6).
static bool vnlaIdLess(const VirtualNLA *a, const VirtualNLA *b) {
  return a->id < b->id;
}

namespace {

/// One hop of an absolute NLA path.
///
/// Hops that are instances have their operation stored in `inst`.  If the hop
/// has an inner symbol, it is stored in `sym` (else null).
///
/// Non-terminal hops must be instances, while terminal hops can be module or
/// innerrefs.  A module-only terminal has neither `inst` nor `sym`.
struct PathHop {
  StringAttr mod;  ///< Containing module.
  Operation *inst; ///< The instance op, if this hop is an instance.
  StringAttr sym;  ///< Inner symbol, if known.
  bool operator==(const PathHop &o) const {
    return mod == o.mod && inst == o.inst && sym == o.sym;
  }
};

/// A hop-path view with precomputed hash: the trimmed-upper-path dedup key.
/// The view must reference storage that outlives the set (`upperPaths`).
struct TrimmedPathRef {
  ArrayRef<PathHop> path;
  llvm::hash_code hash;

  static TrimmedPathRef get(ArrayRef<PathHop> path) {
    llvm::hash_code h = llvm::hash_value(path.size());
    for (const PathHop &hop : path)
      h = llvm::hash_combine(h, hop.mod.getAsOpaquePointer(), hop.inst,
                             hop.sym.getAsOpaquePointer());
    return {path, h};
  }
};

/// P2: Plans VirtualNLA contexts for each hierpath that survives inlining.
/// Traces up from roots, trims paths, deduplicates, and builds routing tables.
/// Read-only after run(); a rejected plan leaves IR untouched.
class NLAPlanner {
public:
  NLAPlanner(CircuitOp circuit, SymbolTable &symbolTable,
             InstanceGraph &instanceGraph, const InliningFacts &facts)
      : circuit(circuit), symbolTable(symbolTable),
        instanceGraph(instanceGraph), facts(facts) {}

  LogicalResult run();
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump();
#endif

  /// Source-path terminal-shape counters, copied into the pass statistics.
  ///
  /// The leaf-rename machinery (`updateVirtualNLALeafSymbols` and the scan in
  /// `rename`) exists to serve paths ending at an inner symbol.
  struct Statistics {
    size_t endInnerSym = 0;
    size_t endModule = 0;
  } stats;

private:
  /// Create a VNLA using our allocator; its `id` is its index in `allVNLAs`.
  VirtualNLA *createVNLA(StringAttr origSym, ArrayRef<SurvivingHop> path);

  /// Trace upward from a root module until reaching surviving modules,
  /// discovering all contexts where the root appears after inlining.
  /// Depth-first over the instance graph's uses, with an explicit stack.
  LogicalResult
  traceUpUntilSurviving(StringAttr rootModName, hw::HierPathOp diagAnchor,
                        SmallVectorImpl<SmallVector<PathHop>> &discoveredPaths);

  /// Create a VirtualNLA for one concrete path context.
  /// The new context lands in `allVNLAs` and in `pathRoutingTable`.
  /// The result carries only whether planning this context succeeded.
  LogicalResult
  processSinglePathContext(StringAttr origSym,
                           const SmallVectorImpl<PathHop> &absPath,
                           hw::HierPathOp diagAnchor);

  /// Return the index of the minimal root within `upperPath`: how many leading
  /// hops to drop so the context roots at the deepest module still surviving on
  /// this concrete path.
  ///
  /// `rootMod` is the (original) NLA root that the last upper hop climbs into.
  /// Its own fate decides: real climb above (keep) or spurious (trim).
  size_t minimalRootIndex(ArrayRef<PathHop> upperPath, StringAttr rootMod);

  /// Resolve an instance named by (`module`, `innerSym`) to its op.
  ///
  /// Every non-terminal namepath hop is an instance by definition, and the
  /// instance graph already holds those ops (no IR walk / IST needed).
  ///
  /// Each module's instances are indexed once, on its first hop.  Total cost is
  /// one graph-node sweep per distinct hop module, not per hop.
  ///
  /// Returns null when the sym names a non-instance (a terminal wire/port hop)
  /// or the module is unknown; such hops are not routed through.
  Operation *resolveInstanceHop(StringAttr module, StringAttr innerSym);

  CircuitOp circuit;
  SymbolTable &symbolTable;
  InstanceGraph &instanceGraph;
  const InliningFacts &facts;

  /// Per-module instance index for `resolveInstanceHop`.  Only modules
  /// hierpaths hop through are indexed.
  DenseMap<StringAttr, DenseMap<StringAttr, Operation *>> instanceHopIndex;

  /// Stable pool allocation for virtual NLA structures.
  llvm::BumpPtrAllocator alloc;

  using VirtualNLAHandles = SmallVector<VirtualNLA *>;

public:
  /// Instance op -> the VNLAs routing through that instance (safe per I7).
  /// Keyed on the op: one lookup per descent, no (module, sym) probing.
  DenseMap<Operation *, VirtualNLAHandles> pathRoutingTable;
  /// NLA sym -> set of its VNLA's: contiguous slices of `allVNLAs` (I4).
  /// Materialized once the pool freezes; appending after would dangle them.
  DenseMap<StringAttr, ArrayRef<VirtualNLA *>> origToVNLAs;
  /// All VNLA's in creation order, contiguous per origSym (I4).
  SmallVector<VirtualNLA *> allVNLAs;

  /// Source symbol -> its HierPathOp, recorded while bucketing during run().
  /// Valid pass-wide: hierpath ops are untouched until the final writeback.
  DenseMap<StringAttr, hw::HierPathOp> hierPathOps;
};

} // namespace

/// DenseMapInfo specialization for TrimmedPathRef to enable deduplication.
template <>
struct llvm::DenseMapInfo<TrimmedPathRef> {
  static unsigned getHashValue(const TrimmedPathRef &key) {
    return static_cast<unsigned>(key.hash);
  }
  static bool isEqual(const TrimmedPathRef &a, const TrimmedPathRef &b) {
    return a.hash == b.hash && a.path == b.path;
  }
};

LogicalResult NLAPlanner::run() {
  // Bucket HierPathOps by their root, preserving their encounter order.
  //
  // Allows us to reuse `traceUpUntilSurviving` and the trimming/dedup work
  // across all NLA's sharing a root (bucket).
  //
  // VNLA creation is contiguous per origSym (I4), grouped by root.
  llvm::MapVector<StringAttr, SmallVector<hw::HierPathOp>> byRoot;
  for (auto nla : circuit.getOps<hw::HierPathOp>()) {
    byRoot[nla.root()].push_back(nla);
    hierPathOps[nla.getSymNameAttr()] = nla;
  }

  for (auto &[origRoot, nlas] : byRoot) {
    // 1. Perform the work shared by all NLA's in this bucket:

    // 1.1 Calculate all the upperPaths by tracing upwards as needed.
    SmallVector<SmallVector<PathHop>> upperPaths;
    if (failed(traceUpUntilSurviving(origRoot, nlas.front(), upperPaths)))
      return failure();

    // 1.2 Trim these back down and dedup, being careful to preserve their
    // order.  Hash equivalence (collision) does not mean contexts are lost.
    SmallVector<SmallVector<PathHop>> trimmedUppers;
    llvm::SmallDenseSet<TrimmedPathRef, 8> seenTrimmed;
    for (auto &upperPath : upperPaths) {
      size_t root = minimalRootIndex(upperPath, origRoot);
      ArrayRef<PathHop> trimmed(upperPath.begin() + root, upperPath.end());
      if (seenTrimmed.insert(TrimmedPathRef::get(trimmed)).second)
        trimmedUppers.emplace_back(trimmed.begin(), trimmed.end());
    }

    // 2. Process each NLA in the bucket.
    for (auto nla : nlas) {
      auto origSym = nla.getSymNameAttr();

      // 2.1 Extract the "hop"s from the NLA.
      SmallVector<PathHop> nlaHops;
      for (auto element : nla.getNamepath()) {
        if (auto ref = dyn_cast<InnerRefAttr>(element)) {
          nlaHops.push_back({ref.getModule(),
                             resolveInstanceHop(ref.getModule(), ref.getName()),
                             ref.getName()});
        } else if (auto flat = dyn_cast<FlatSymbolRefAttr>(element))
          nlaHops.push_back({flat.getAttr(), nullptr, StringAttr()});
        else
          llvm_unreachable("NLA element must be innerref or flat symbol");
      }

      // Track statistics about terminal shape.
      ++(nlaHops.back().sym ? stats.endInnerSym : stats.endModule);

      // 2.2 Construct each (bucket-common) prefix + NLA path, and hand to
      // helper to walk the full path and produce final VNLA for each.
      for (auto &trimmedUpper : trimmedUppers) {
        SmallVector<PathHop> absolutePath;
        llvm::append_range(absolutePath, trimmedUpper);
        llvm::append_range(absolutePath, nlaHops);

        if (failed(processSinglePathContext(origSym, absolutePath, nla)))
          return failure();
      }
    }
  }

  // I5: routing entries are born id-sorted and duplicate-free.
  assert(llvm::all_of(pathRoutingTable,
                      [](const auto &entry) {
                        return llvm::is_sorted(entry.second, vnlaIdLess);
                      }) &&
         "routing entries must be born id-sorted (I5)");

  // 3. Materialize the per-symbol group views that point to VNLA ranges:
  // contexts are contiguous per origSym in creation order (I4).
  //
  // The pool is frozen from here on (I2).
  for (size_t i = 0, e = allVNLAs.size(); i < e;) {
    StringAttr origSym = allVNLAs[i]->origSym;
    size_t groupStart = i;
    while (i < e && allVNLAs[i]->origSym == origSym)
      ++i;
    origToVNLAs[origSym] =
        ArrayRef<VirtualNLA *>(&allVNLAs[groupStart], i - groupStart);
  }

  return success();
}

VirtualNLA *NLAPlanner::createVNLA(StringAttr origSym,
                                   ArrayRef<SurvivingHop> path) {
  // I4: VNLA id is monotonic and P2 processes one hierpath at a time,
  // so ids are creation-ordered and contiguous per source symbol.
  auto id = allVNLAs.size();
  allVNLAs.push_back(VirtualNLA::create(alloc, id, origSym, path));
  return allVNLAs.back();
}

LogicalResult NLAPlanner::traceUpUntilSurviving(
    StringAttr rootModName, hw::HierPathOp diagAnchor,
    SmallVectorImpl<SmallVector<PathHop>> &discoveredPaths) {
  using UseIterator =
      decltype(std::declval<igraph::InstanceGraphNode>().uses().begin());

  /// Stack frame for iterative upward trace through the instance graph.
  struct Frame {
    StringAttr modName;
    UseIterator currentEdge;
    UseIterator endEdge;
    bool isFirstVisit;
  };

  SmallVector<Frame, 16> stack;
  SmallVector<PathHop, 8> currentPath;

#ifndef NDEBUG
  DenseMap<StringAttr, bool> visited;
#endif

  // Edge-derived names come from the instance graph itself: always resolve.
  // Only the root, straight from the namepath, can name a missing module.
  auto pushState = [&](StringAttr name) -> LogicalResult {
    auto *node = instanceGraph.lookupOrNull(name);
    if (!node)
      return diagAnchor.emitOpError()
             << "names non-existent root module @" << name;
    auto uses = node->uses();
    stack.push_back({name, uses.begin(), uses.end(), /*isFirstVisit=*/true});

#ifndef NDEBUG
    if (visited[name])
      return mlir::emitError(node->getModule().getLoc(),
                             "instance graph contains cycle");
    visited[name] = true;
#endif

    return success();
  };
  auto popState = [&]() {
#ifndef NDEBUG
    auto name = stack.back().modName;
    auto it = visited.find(name);
    assert(it != visited.end() && "visited map missing module");
    assert(it->second && "visited not set for module");
    it->second = false;
#endif
    stack.pop_back();
  };

  if (failed(pushState(rootModName)))
    return failure();

  while (!stack.empty()) {
    auto &frame = stack.back();
    if (frame.isFirstVisit) {
      frame.isFirstVisit = false;

      auto *currentModNode = instanceGraph.lookup(frame.modName);
      auto *currentModOp = currentModNode->getModule().getOperation();
      auto infoIfValid = facts.getModuleInfoIfPresent(currentModOp);
      // This is expected unreachable, but diagnose for safety.
      if (!infoIfValid)
        return mlir::emitError(
                   currentModOp->getLoc(),
                   "hierarchical path traced up through unknown operation")
                   .attachNote(diagAnchor.getLoc())
               << "encountered tracing up from root of this hierarchical path";
      auto info = *infoIfValid;

      // If this module is live in the output in any context, emit VNLA for it.
      // A live root's own context is discovered first (frame pushed first).
      // Primary selection rests on that ordering (I4).
      if (info.isLive)
        discoveredPaths.push_back(llvm::to_vector(llvm::reverse(currentPath)));

      // If this module is unconditionally live, we're done tracing upwards.
      if (!info.hasInline && !info.underFlatten) {
        popState();
        if (!stack.empty())
          currentPath.pop_back();
        continue;
      }
    }
    // If we've exhausted all edges, we're done with this frame.
    if (frame.currentEdge == frame.endEdge) {
      popState();
      if (!stack.empty())
        currentPath.pop_back();
      continue;
    }

    // Trace up the current edge and advance it on the frame.
    auto *edge = *frame.currentEdge;
    ++frame.currentEdge;

    auto *instOp = edge->getInstance().getOperation();
    // Only climb through inlinable instances (see getInlinableInstance).
    // Opaque instantiations keep their targets alive and are handled above.
    //
    // There is no copy in this parent to enumerate -> don't climb!
    if (!getInlinableInstance(instOp))
      continue;
    auto parentName = edge->getParent()->getModule().getModuleNameAttr();
    currentPath.push_back({parentName, instOp, getInnerSymName(instOp)});
    if (failed(pushState(parentName)))
      return failure();
  }

  return success();
}

size_t NLAPlanner::minimalRootIndex(ArrayRef<PathHop> upperPath,
                                    StringAttr rootMod) {
  // The bottom-up trace over-approximates to ensure coverage.
  // Here we walk top-down to find the minimal root point precisely.
  //
  // Returns its index in `upperPath`; `upperPath.size()` roots at `rootMod`.
  // The namepath itself is the user's spec and is never trimmed.
  //
  // This rooting is coupled to inline/flatten knowledge.
  // A new way for a module to be inlined away or relocated must re-derive it,
  // or a surviving root is misidentified (misrouted annotation; I9 churn).
  //
  // Inline and flatten act differently, deciding how deep to root:
  //  - Inline: never survives as a root but children do.  Keep looking.
  //  - Flatten: subtree doesn't survive, this is as deep as we can root.
  //
  // The dropped prefix has no flatten, so the kept suffix evaluates the same.
  bool isTransitiveFlatten = false;
  // `root` never stays unset: we traced upwards until this was certain.
  size_t root = 0;
  for (size_t i = 0, e = upperPath.size(); i <= e; ++i) {
    // Flatten means we're done searching, use the deepest we've found.
    if (isTransitiveFlatten)
      break;
    StringAttr mod = i < e ? upperPath[i].mod : rootMod;
    const auto &info =
        facts.getModuleInfo(symbolTable.lookup<FModuleLike>(mod));
    // Inline modules don't survive as a root; can still root below them.
    if (!info.hasInline)
      root = i;
    isTransitiveFlatten |= info.hasFlatten;
  }
  return root;
}

Operation *NLAPlanner::resolveInstanceHop(StringAttr module,
                                          StringAttr innerSym) {
  auto [entry, inserted] = instanceHopIndex.try_emplace(module);
  if (inserted) {
    auto *node = instanceGraph.lookupOrNull(module);
    if (!node)
      return nullptr;
    for (auto *record : *node) {
      auto *inst = record->getInstance().getOperation();
      if (auto sym = getInnerSymName(inst))
        entry->second.try_emplace(sym, inst);
    }
  }
  return entry->second.lookup(innerSym);
}

LogicalResult
NLAPlanner::processSinglePathContext(StringAttr origSym,
                                     const SmallVectorImpl<PathHop> &absPath,
                                     hw::HierPathOp diagAnchor) {
  SmallVector<SurvivingHop> survivingHops;
  assert(!absPath.empty() && "empty absolute path -- empty namepath?");

  StringAttr currentDest = absPath.front().mod;
  auto destMod = symbolTable.lookup<FModuleLike>(currentDest);
  const auto &destInfo = facts.getModuleInfo(destMod);

  bool isTransitiveFlatten = destInfo.hasFlatten;
  // The module whose flatten most recently set `isTransitiveFlatten`;
  // non-null exactly when the flag is set.  Names the culprit in diagnostics.
  StringAttr flattenCause = isTransitiveFlatten ? currentDest : StringAttr{};
  for (auto it = absPath.begin(), end = absPath.end(); it != end; ++it) {
    const auto &hop = *it;
    bool isTerminal = std::next(it) == end;

    // A hop's operation is null, inlinable, or opaque
    // (see getInlinableInstance).
    auto hopInst = getInlinableInstance(hop.inst);
    bool isOpaqueInstanceHop = hop.inst && !hopInst;

    // Interior hops read the recorded next module; a terminal names one only
    // through a plain instance.
    StringAttr nextModName;
    if (!isTerminal)
      nextModName = std::next(it)->mod;
    else if (hopInst)
      nextModName = hopInst.getReferencedModuleNameAttr();

    // The next module's facts, defaulted for the end of the path.
    bool nextHasInline = false;
    bool nextHasFlatten = false;
    bool nextIsRegular = false;
    if (nextModName) {
      assert((isTerminal || !hopInst ||
              std::next(it)->mod == hopInst.getReferencedModuleNameAttr()) &&
             "recorded next module disagrees with the instance");
      auto modOp = symbolTable.lookup<FModuleLike>(nextModName);
      assert(modOp && "interior namepath module missing -- ran unverified?");
      const auto &info = facts.getModuleInfo(modOp);
      nextHasInline = info.hasInline;
      nextHasFlatten = info.hasFlatten;
      nextIsRegular = isa<FModuleOp>(modOp);
    }
    // - A hop 'evaporates' only when an instance is inlined.
    // - Terminal hops never evaporate (I8).
    // - Non-regular modules are never inlined; their instances relocate.
    // - Neither are opaque instantiations; they relocate with their operation.
    bool isEvaporating = nextModName && nextIsRegular && !isOpaqueInstanceHop &&
                         (isTransitiveFlatten || nextHasInline);

    // Diagnose evaporated terminal instance hops, keep I8 accurate and avoid
    // bugs.  Only old-style annotations have this shape, which is recoverable
    // if that is the only user.  Other users (e.g., XMR or verbatim) must be
    // detected and rejected.  This will be handled in follow-on.  Expected to
    // be unreachable from current frontend + pipeline.
    // https://github.com/llvm/circt/issues/10908
    if (isEvaporating && isTerminal) {
      assert(hop.inst && "expected instance operation");
      auto diag = diagAnchor.emitError(
          "hierpath points to inlined instance, cannot proceed");
      diag.attachNote(hop.inst->getLoc())
          << "hierpath targets this inlined instance";
      // Name the absorption cause to make the fix actionable.
      if (nextHasInline) {
        diag.attachNote(symbolTable.lookup(nextModName)->getLoc())
            << "target module is marked inline";
      } else {
        // The walk's own scope state names the culprit: the most recent
        // flatten still active here.  A flatten above a choice hop is not a
        // cause; the choice began a fresh scope.
        assert(flattenCause && "flatten-caused absorption without a cause");
        diag.attachNote(symbolTable.lookup(flattenCause)->getLoc())
            << "flattening this module inlines the instance";
      }
      return failure();
    }

    // An opaque instance's target begins a fresh flatten scope: flatten does
    // not reach through it (as with an extmodule).  Terminal state is never
    // carried forward (the loop ends); don't let it redefine the locals.
    if (!isTerminal) {
      if (isOpaqueInstanceHop) {
        isTransitiveFlatten = nextHasFlatten;
        flattenCause = nextHasFlatten ? nextModName : StringAttr{};
      } else {
        isTransitiveFlatten |= nextHasFlatten;
        if (nextHasFlatten)
          flattenCause = nextModName;
      }
    }
    if (!isEvaporating) {
      // `sym` is null only for a non-instance hop (terminal or module-only).
      // Namepath InnerRefs always name a sym.
      // A climbed symless instance evaporates before reaching here
      // (it fronts a regular inline/underFlatten module).
      StringAttr sym = hop.sym;
      assert((sym || isTerminal || !hop.inst) &&
             "surviving instance hop without an inner symbol");
      StringAttr finalSym;
      if (currentDest == hop.mod || isTerminal /* terminals keep their sym */)
        finalSym = sym;
      survivingHops.push_back({/*origMod=*/hop.mod, /*origSym=*/sym,
                               /*finalMod=*/currentDest,
                               /*finalSym=*/finalSym});
      if (!isTerminal && nextModName)
        currentDest = nextModName;
    }
  }

  // Create the VNLA using the next id.
  auto *vnla = createVNLA(origSym, survivingHops);

  // Register this context under each instance it descends through.
  // Non-instance hops (module-only, terminal wire/port) are not routed.
  // Creation-order appends give I5's sorting; the DAG gives its dup-freedom.
  for (const auto &hop : absPath)
    if (hop.inst)
      pathRoutingTable[hop.inst].push_back(vnla);

  return success();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void NLAPlanner::dump() {
  llvm::dbgs() << "\nVirtualNLAs (creation order):\n";
  for (auto *vnla : allVNLAs)
    vnla->dump();

  llvm::dbgs() << "\nPath Routing Table (Instance -> Routed VirtualNLAs):\n";

  // Keys are instance ops; sort by (containing module, inner sym) for stable
  // output across runs.
  auto modOf = [](Operation *op) -> StringAttr {
    auto mod = op->getParentOfType<FModuleLike>();
    return mod ? mod.getModuleNameAttr() : StringAttr();
  };
  SmallVector<Operation *> insts;
  for (const auto &[inst, _] : pathRoutingTable)
    insts.push_back(inst);
  llvm::sort(insts, [&](Operation *a, Operation *b) {
    auto am = modOf(a), bm = modOf(b);
    if (am != bm)
      return (am ? am.getValue() : "") < (bm ? bm.getValue() : "");
    auto as = getInnerSymName(a), bs = getInnerSymName(b);
    return (as ? as.getValue() : "") < (bs ? bs.getValue() : "");
  });

  for (auto *inst : insts) {
    const auto &vnlas = pathRoutingTable.lookup(inst);
    llvm::dbgs() << "  @" << modOf(inst);
    if (auto instSym = getInnerSymName(inst))
      llvm::dbgs() << "::" << instSym;
    else
      llvm::dbgs() << "::<op@" << inst << ">";

    llvm::dbgs() << " -> [";
    llvm::interleaveComma(vnlas, llvm::dbgs(), [&](VirtualNLA *vnla) {
      llvm::dbgs() << "#" << vnla->id;
    });
    llvm::dbgs() << "]\n";
  }

  llvm::dbgs() << "\n";
}
#endif

//===----------------------------------------------------------------------===//
// Module Inlining Support
//===----------------------------------------------------------------------===//

/// Map each of the instance's results to its corresponding replacement wire.
/// Later clones from the parent block then read the wires.
static void mapResultsToWires(IRMapping &mapper, SmallVectorImpl<Value> &wires,
                              InstanceOp instance) {
  for (auto [result, wire] : llvm::zip_equal(instance.getResults(), wires))
    mapper.map(result, wire);
}

/// Process each operation, updating InnerRefAttr's using the specified map,
/// with the given name as the containing IST of the mapped-to sym names.
///
/// Every inner-ref in the cloned ops names the child being inlined, so `map`
/// covers them all (I11).
///
/// A miss is a reference to another module, in an unknown capacity (!).
/// There is no correct update, so it is diagnosed.
static LogicalResult replaceInnerRefUsers(ArrayRef<Operation *> newOps,
                                          const InnerRefToNewNameMap &map,
                                          StringAttr istName) {
  hw::InnerRefAttr foreign;
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](hw::InnerRefAttr innerRef) {
    auto it = map.find(innerRef);
    if (it == map.end()) {
      if (!foreign)
        foreign = innerRef;
      return std::pair{innerRef, WalkResult::skip()};
    }
    return std::pair{hw::InnerRefAttr::get(istName, it->second),
                     WalkResult::skip()};
  });
  for (auto *op : newOps) {
    replacer.recursivelyReplaceElementsIn(op);
    if (foreign)
      return op->emitError("unsupported inner reference ")
             << foreign << " found while inlining";
  }
  return success();
}

/// Unique each of `old`'s symbols in `ns`; record old-ref -> new-name entries
/// in `map` under `istName`.
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
    assert(oldProp.getFieldID() == newProp.getFieldID() &&
           "uniquing must preserve fieldIDs");
    // Record every prop, changed or not: the map must be total (I11).
    map[hw::InnerRefAttr::get(istName, oldProp.getName())] = newProp.getName();
  }

  return newSymAttr;
}

//===----------------------------------------------------------------------===//
// Inliner
//===----------------------------------------------------------------------===//

/// Inlines, flattens, and removes dead modules in a circuit.
///
/// The inliner works top-down, in parents-before-children order.
/// Only live modules (I1) are visited; every marked instance is inlined.
/// Each operation clones directly to its final location.
/// Dead modules are erased at the end.
///
/// Every cloned operation with a name gets the instance-name prefix.
/// Top-down, the entire prefix is known at clone time, so the name attribute is
/// set exactly once (no interned intermediates).
namespace {
class Inliner {
public:
  /// Initialize the inliner to run on this circuit.
  Inliner(CircuitOp circuit, SymbolTable &symbolTable,
          CircuitNamespace &circuitNamespace,
          const InliningFacts &inliningFacts, NLAPlanner &nlaPlanner);

  /// Run the inliner.
  LogicalResult run();

  /// Work counters, copied into the pass statistics after run().
  struct Statistics {
    size_t instancesInlined = 0;   ///< Inlined via an inline mark.
    size_t instancesFlattened = 0; ///< Inlined via flatten.
    size_t deadModules = 0;        ///< Modules erased after inlining.
    size_t hierPathsUpdated = 0;   ///< HierPaths retargeted in place.
    size_t hierPathsForked = 0;    ///< Fork contexts emitted as new hierpaths.
    size_t hierPathsMerged = 0;    ///< Contexts folded onto a canonical.
    size_t hierPathsErased = 0;    ///< Erased: no surviving target
                                   ///< (dead-rooted).
  } stats;

private:
  //===- Inlining contexts ------------------------------------------------===//

  /// Inlining context, one per module being inlined into.
  struct ModuleInliningContext {
    ModuleInliningContext(FModuleOp module)
        : module(module), modNamespace(module), b(module.getContext()) {}
    FModuleOp module; ///< Top-level module for current inlining task.
    /// Inner-symbol namespace for minted names (I10).
    /// Every inlining level below this module shares it.
    /// Built from the pristine body: I12 defers mutation to this one visit.
    hw::InnerSymbolNamespace modNamespace;
    OpBuilder b; ///< Builder, insertion point into module.
  };

  /// One inlining level, created for each instance inlined or flattened.
  /// Renamed inner symbols land in relocatedInnerSyms; clones in newOps.
  /// `finalize()` fixes the clones up once the level is complete.
  struct InliningLevel {
    InliningLevel(ModuleInliningContext &mic, FModuleOp childModule)
        : mic(mic), childModule(childModule) {}

    ModuleInliningContext &mic;              ///< Top-level inlining context.
    InnerRefToNewNameMap relocatedInnerSyms; ///< Inner-ref rename map.
    SmallVector<Operation *> newOps;         ///< All cloned operations.
    SmallVector<Value> wires;                ///< Wires created for ports.
    FModuleOp childModule;                   ///< The module being inlined.
    Value debugScope;                        ///< Debug scope of the instance.
    /// VNLAs active at this level, id-sorted (I6; see setActiveNLAsForChild).
    SmallVector<VirtualNLA *> activeNLAs;

    /// Set the active contexts for this inlining level.
    void setActivePaths(ArrayRef<VirtualNLA *> nlas) {
      activeNLAs.assign(nlas);
    }

    /// Retarget the inner references of this level's clones once complete.
    /// Called on the creator's success path.
    /// A level abandoned to a pass failure skips it.
    LogicalResult finalize() {
      return replaceInnerRefUsers(newOps, relocatedInnerSyms,
                                  mic.module.getNameAttr());
    }
  };

  //===- P3: clone and rename ---------------------------------------------===//

  /// Rename an operation and unique any symbols it has.
  /// Returns true iff symbol was changed.
  bool rename(StringRef prefix, Operation *op, InliningLevel &il);

  /// Rename an instance-like op, uniquing any symbols it has.
  /// Requires old and new operations, to update the hierpath hops involved.
  bool renameInstance(StringRef prefix, InliningLevel &il, Operation *oldInst,
                      Operation *newInst);

  /// Clone and rename an operation.
  /// Insert the operation into the inlining level.
  void cloneAndRename(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                      Operation &op);

  /// Record, for a freshly cloned op, which contexts own its nonlocal
  /// annotations (`circt.nonlocal` in the annotation payload).
  ///
  /// For each hierpath named by the clone's annotations, record the owning
  /// contexts: those whose route (surviving instance path) passes through this
  /// level, gated on ownership (I13).  Only the walk knows a clone's position
  /// on any route (afterwards clones are indistinguishable), hence the record.
  /// P3 records, P4 writes.
  ///
  /// Every clone with nonlocal annotations still gets an entry, possibly empty,
  /// so the writeback never applies the original-op rule to it.
  void recordContexts(Operation *newOp, const InliningLevel &il);

  /// Record a renamed leaf inner symbol on the active contexts, or the
  /// materialized path dangles.
  ///
  /// Matching is per-field by (origMod, origSym), destination-gated (I12/I13).
  /// Needed only to preserve NLA leaf symbols.
  void updateVirtualNLALeafSymbols(Inliner::InliningLevel &il,
                                   hw::InnerSymAttr oldSymAttr,
                                   hw::InnerSymAttr newSymAttr);

  /// Compute the contexts active inside a child inlining level: the
  /// intersection (I5/I6) of the parent's active set with the contexts routed
  /// through `instance`.
  ///
  /// `std::nullopt` = top-level entry, no parent filter.
  /// Inputs are id-sorted; the result stays id-sorted.
  void setActiveNLAsForChild(std::optional<ArrayRef<VirtualNLA *>> activeNLAs,
                             InliningLevel &childIL, Operation *instance);

  /// Rewrite the ports of a module as wires.
  /// This is similar to cloneAndRename, but operating on ports.
  /// Wires are added to il.wires.
  void mapPortsToWires(StringRef prefix, InliningLevel &il, IRMapping &mapper);

  //===- P3: the walk -----------------------------------------------------===//

  /// Returns true if the operation is annotated to be flattened.
  bool shouldFlatten(FModuleLike mod);

  /// Returns true if the operation is annotated to be inlined.
  bool shouldInline(FModuleLike mod);

  /// Check we're not inlining into anything other than layerblock or module.
  /// In the future, could check this per-inlined-operation.
  LogicalResult checkInstanceParents(InstanceOp instance);

  /// Walk the specified block, invoking `process` forward, pre-order.
  ///
  /// Handles cloning supported operations with regions, so that `process` is
  /// only invoked on regionless operations.
  LogicalResult
  inliningWalk(OpBuilder &builder, Block *block, IRMapping &mapper,
               llvm::function_ref<LogicalResult(Operation *op)> process);

  /// Clone a target module's body into the insertion point of the builder,
  /// renaming all operations using the prefix, and recurse into the instances
  /// the pass inlines.
  ///
  /// Under `flatten` every regular-module child is inlined,
  /// otherwise only the children marked for it.
  /// (A flatten-marked child switches its subtree into flatten mode.)
  ///
  /// Does not trigger inlining on the target itself.
  LogicalResult processInto(StringRef prefix, InliningLevel &il,
                            IRMapping &mapper, bool flatten);

  /// Replace with its body every instance in `module` the pass inlines:
  /// every regular-module instance when `flatten` is set, otherwise the ones
  /// marked for inlining.
  LogicalResult processInstances(FModuleOp module, bool flatten);

  /// Create a debug scope for an inlined instance at the current insertion
  /// point of the `il.mic` builder.
  void createDebugScope(InliningLevel &il, InstanceOp instance,
                        Value parentScope = {});

  /// P3: inline/flatten the live modules in parents-before-children order
  /// (I12), then drop debug scopes that ended up unused.
  LogicalResult inlineModules();

  /// Erase the modules the analysis marked dead.
  void eraseDeadModules();

  //===- P4: write back ---------------------------------------------------===//

  /// Append `anno`, rewritten for one context (`matched`) of its source NLA:
  ///   context went local -> drop the `circt.nonlocal` member
  ///   context kept       -> `circt.nonlocal` = canonical owner's realizedSym
  ///
  /// `origSym` is the annotation's current nonlocal symbol.
  /// Retargeting also flags the context used.
  /// P4-only; the context's single writer makes `wasUsed` race-free (I14).
  void appendContextAnno(Annotation anno, StringAttr origSym,
                         VirtualNLA *matched, SmallVectorImpl<Attribute> &out);

  /// P4: (serially) canonicalize every non-local context; see `canonicalize`.
  /// Also mints each emitted context's `realizedSym`.
  void canonicalizeContexts();

  /// P4: rewrite all annotations, in parallel across regular modules.
  /// The single annotation writer.
  void rewriteAnnotations();

  /// P4: materialize the surviving hierpaths and erase the rest.
  void writebackHierPaths();

  /// Build the resolved namepath (an ArrayAttr of inner-refs / flat symbols)
  /// from a VNLA's surviving hops.
  ///
  /// Callable once a VNLA's path is final.
  ArrayAttr materializeNamepath(VirtualNLA *vnla);

  /// Late-convergence canonicalization:
  /// Contexts from different NLAs can realize identical namepaths.
  /// A hierpath is defined solely by its namepath, so they share one op.
  /// The first to canonicalize (deterministic sweep order) owns it.
  ///
  /// The mapping lands in `canonicalOf`.
  ///
  /// Also mints `realizedSym` for canonicals: the primary already claimed
  /// origSym at selection, so a canonical fork always mints; duplicates borrow.
  ///
  /// Serial-sweep only, once every path is final.
  void canonicalize(VirtualNLA *vnla);

  /// Read-only canonical lookup for the parallel annotation rewrite:
  /// every non-local VNLA has been through `canonicalize` by then, so this only
  /// reads `canonicalOf`.
  ///
  /// Returns `vnla` itself when it is canonical or excluded.
  VirtualNLA *canonicalOrSelf(VirtualNLA *vnla) const {
    return canonicalOf.lookup_or(vnla, vnla);
  }

  //===- State ------------------------------------------------------------===//

  CircuitOp circuit;
  MLIRContext *context;

  /// A symbol table with references to each module in a circuit.
  SymbolTable &symbolTable;

  /// Namespace for generating unique circuit-level names.
  /// Module-level namespaces live on the MICs (I10).
  CircuitNamespace &circuitNamespace;

  /// Analysis / planner results (P1 and P2).
  const InliningFacts &inliningFacts;
  NLAPlanner &nlaPlanner;

  /// Late-convergence canonicalization side tables.
  /// Inliner-owned so VirtualNLA stays frozen after the prepass (I2).
  ///
  /// `canonicalByPath` interns each distinct resolved namepath to the VNLA that
  /// owns its hierpath; `canonicalOf` maps every non-local VNLA to that owner
  /// (itself when canonical).
  ///
  /// A duplicate is skipped at emission, borrows the owner's realizedSym, and
  /// propagates usedness onto it.
  DenseMap<ArrayAttr, VirtualNLA *> canonicalByPath;
  DenseMap<VirtualNLA *, VirtualNLA *> canonicalOf;

  /// Assert-only bookkeeping: origSyms claimed by their group's primary, so
  /// `canonicalize` can check the claim order (I15).
  struct ClaimedSyms {
#ifndef NDEBUG
    DenseSet<StringAttr> syms;
    void claim(StringAttr sym) { syms.insert(sym); }
    bool has(StringAttr sym) const { return syms.contains(sym); }
#else
    void claim(StringAttr) {}
    bool has(StringAttr) const { return true; }
#endif
  } claimed;

  /// For every op the walk cloned that carries `circt.nonlocal` annotations,
  /// the contexts that own them: active on the clone's descent (route) and
  /// owned by its destination module (I13).
  ///
  /// Written by P3, read-only in P4.
  ///
  /// Keys are cloned ops and stay valid through the P4 reads:
  /// the pass erases only originals (consumed instances, dead-module bodies)
  /// and unused debug scopes, never a clone, so no key is freed and no stale
  /// entry can alias a live query.
  ///
  /// The other side of I7: routing keys are originals that outlive the walk.
  /// These keys are clones that do.
  DenseMap<Operation *, SmallVector<VirtualNLA *, 2>> clonedAnnoContexts;

  /// The debug scopes created for inlined instances.
  /// Scopes that are unused after inlining will be deleted again.
  SmallVector<debug::ScopeOp> debugScopes;
};
} // namespace

//===- Driver -------------------------------------------------------------===//

Inliner::Inliner(CircuitOp circuit, SymbolTable &symbolTable,
                 CircuitNamespace &circuitNamespace,
                 const InliningFacts &inliningFacts, NLAPlanner &nlaPlanner)
    : circuit(circuit), context(circuit.getContext()), symbolTable(symbolTable),
      circuitNamespace(circuitNamespace), inliningFacts(inliningFacts),
      nlaPlanner(nlaPlanner) {}

LogicalResult Inliner::run() {
  if (failed(inlineModules()))
    return failure();
  eraseDeadModules();

  canonicalizeContexts();
  rewriteAnnotations();
  writebackHierPaths();

  return success();
}

//===- P3: clone and rename -----------------------------------------------===//

/// Prefix the op's name and unique its inner symbols in the module namespace.
/// Renames land in `relocatedInnerSyms` for the level's inner-ref fixup.
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

  // Prefix the "name" attribute, when present.
  if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
    op->setAttr("name", StringAttr::get(op->getContext(),
                                        (prefix + nameAttr.getValue())));

  // Unique any inner symbols; reflect renames on the active contexts' leaves.
  auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(op);
  if (!symOp)
    return false;
  auto oldSymAttr = symOp.getInnerSymAttr();
  auto newSymAttr =
      uniqueInNamespace(oldSymAttr, il.relocatedInnerSyms, il.mic.modNamespace,
                        il.childModule.getNameAttr());

  if (!newSymAttr)
    return false;

  // TODO: Gate this on whether this participates in NLAs to avoid
  // unnecessary scanning.
  updateVirtualNLALeafSymbols(il, oldSymAttr, newSymAttr);
  symOp.setInnerSymbolAttr(newSymAttr);

  return newSymAttr != oldSymAttr;
}

bool Inliner::renameInstance(StringRef prefix, InliningLevel &il,
                             Operation *oldInst, Operation *newInst) {
  // TODO: No way yet to annotate an explicit parent scope on instances.
  // Just emit a note in debug runs until this is resolved.
  LLVM_DEBUG({
    if (il.debugScope)
      llvm::dbgs() << "Discarding parent debug scope for " << *oldInst << "\n";
  });

  auto oldInstSym = getInnerSymName(oldInst);
  auto symbolChanged = rename(prefix, newInst, il);
  auto newSymAttr = getInnerSymName(newInst);

  // Record the hop even when the symbol is unchanged: a relocated hop's
  // `finalSym` starts null and is only filled here, at its clone.
  if (oldInstSym) {
    assert(newSymAttr && "uniquing dropped an instance sym?");
    StringAttr origMod = il.childModule.getModuleNameAttr();
    StringAttr destMod = il.mic.module.getModuleNameAttr();
    for (auto *nla : il.activeNLAs) {
      for (auto &hop : nla->getPathMutable()) {
        // The `finalMod` test is the I13 ownership gate: an active context
        // belonging to a different copy of this instance must not be updated.
        if (hop.origMod == origMod && hop.origSym == oldInstSym &&
            hop.finalMod == destMod) {
          hop.finalSym = newSymAttr;
        }
      }
    }
  }
  return symbolChanged;
}

void Inliner::recordContexts(Operation *newOp, const InliningLevel &il) {
  StringAttr destMod = il.mic.module.getModuleNameAttr();

  // Intersect a hierpath symbol's context group with the id-sorted active set,
  // keeping only those this destination module owns.

  // A sym's group spans a gap-free id interval (I4; ids globally unique).
  //
  // Its intersection with the id-sorted set (I6) forms one contiguous slice:
  // two searches at the bounds, no per-member probing, id-ordered result.
  auto matchContexts = [&](FlatSymbolRefAttr sym, ArrayRef<VirtualNLA *> active,
                           SmallVectorImpl<VirtualNLA *> &out) {
    auto it = nlaPlanner.origToVNLAs.find(sym.getAttr());
    if (it == nlaPlanner.origToVNLAs.end())
      return;
    ArrayRef<VirtualNLA *> group = it->second;
    const auto *lo = llvm::lower_bound(active, group.front(), vnlaIdLess);
    const auto *hi =
        std::upper_bound(lo, active.end(), group.back(), vnlaIdLess);
    for (; lo != hi; ++lo) {
      // I13: ownership.
      //
      // An annotation is written by the module holding the context's leaf: the
      // annotated op lives there, and it clones into the destination.
      //
      // Activation can be broader than ownership: parent-copy contexts route
      // through the same original instance ops but belong to the parent's
      // clone, not this one.
      //
      // Nonempty per I8: the terminal hop always survives.
      auto path = (*lo)->getPath();
      assert(!path.empty() && "terminal hop is expected to always survive");
      if (path.back().finalMod != destMod)
        continue;
      // One op may name a hierpath more than once; record each owned context
      // once (the writeback re-associates by origSym).
      if (!llvm::is_contained(out, *lo))
        out.push_back(*lo);
    }
  };

  // Annotations: record the owning contexts for the writeback.
  //
  // A clone with nonlocal annotations always gets an entry, possibly empty,
  // so the writeback never applies the original-op rule to it.
  bool hasNonlocal = false;
  SmallVector<VirtualNLA *, 2> annoContexts;
  auto visitAnno = [&](Annotation anno) {
    auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
    if (!sym)
      return;
    hasNonlocal = true;
    matchContexts(sym, il.activeNLAs, annoContexts);
  };
  if (auto annos = newOp->getAttrOfType<ArrayAttr>("annotations"))
    for (Attribute attr : annos)
      visitAnno(Annotation(attr));
  if (auto portAnnos = newOp->getAttrOfType<ArrayAttr>("portAnnotations"))
    for (auto portArray : portAnnos.getAsRange<ArrayAttr>())
      for (Attribute attr : portArray)
        visitAnno(Annotation(attr));
  if (hasNonlocal)
    clonedAnnoContexts[newOp] = std::move(annoContexts);
}

void Inliner::updateVirtualNLALeafSymbols(Inliner::InliningLevel &il,
                                          hw::InnerSymAttr oldSymAttr,
                                          hw::InnerSymAttr newSymAttr) {
  // TODO: Record per-target leaf NLAs in recordContexts, then jump to them.
  // For now, scan the level's active set.
  if (!oldSymAttr || oldSymAttr == newSymAttr)
    return;
  assert(newSymAttr && "renamed to a null sym?");
  StringAttr origMod = il.childModule.getModuleNameAttr();
  StringAttr destMod = il.mic.module.getModuleNameAttr();
  for (auto *nla : il.activeNLAs) {
    // A local context is tracked too: retention can pin it as a one-hop
    // primary, whose leaf symbol must then reflect this rename.
    //
    // `back()` is safe local or not: a VNLA always has its terminal hop (I8,
    // asserted at construction).
    auto &last = nla->getPathMutable().back();
    // `finalMod` is the I13 ownership gate; `origMod` matching is total because
    // the leaf is cloned from its original def exactly once (I12).
    if (last.origMod == origMod && last.finalMod == destMod) {
      for (auto prop : oldSymAttr.getProps()) {
        if (last.origSym == prop.getName()) {
          last.finalSym = newSymAttr.getSymIfExists(prop.getFieldID());
          break;
        }
      }
    }
  }
}

void Inliner::setActiveNLAsForChild(
    std::optional<ArrayRef<VirtualNLA *>> activeNLAs, InliningLevel &childIL,
    Operation *instance) {
  // One lookup by the instance op; the routing entry is born id-sorted and
  // duplicate-free (I5, verified once at the end of planning).
  ArrayRef<VirtualNLA *> instNLAs;
  if (auto it = nlaPlanner.pathRoutingTable.find(instance);
      it != nlaPlanner.pathRoutingTable.end())
    instNLAs = it->second;

  // An empty parent set stays empty; the child default is empty, so leave it.
  if (!activeNLAs) {
    childIL.setActivePaths(instNLAs);
  } else if (!activeNLAs->empty() && !instNLAs.empty()) {
    // Both ranges are id-sorted (I5/I6).
    //
    // A shared instance's routing entry can be fork-count-sized while the
    // active set has narrowed, or the reverse.
    //
    // So: walk the smaller range, binary-search the larger range.
    //
    // The smaller is walked in id order, so the result stays sorted (I6).
    ArrayRef<VirtualNLA *> probe = instNLAs, in = *activeNLAs;
    if (probe.size() > in.size())
      std::swap(probe, in);
    SmallVector<VirtualNLA *> childActiveNLAs;
    for (auto *vnla : probe)
      if (llvm::binary_search(in, vnla, vnlaIdLess))
        childActiveNLAs.push_back(vnla);
    childIL.setActivePaths(childActiveNLAs);
  }
}

/// Create a wire per target-module port at the insertion point, mapping each
/// port to its wire; the cloned body then reads the wires.
void Inliner::mapPortsToWires(StringRef prefix, InliningLevel &il,
                              IRMapping &mapper) {
  auto target = il.childModule;
  auto portInfo = target.getPorts();
  for (unsigned i = 0, e = target.getNumPorts(); i < e; ++i) {
    auto arg = target.getArgument(i);
    auto type = type_cast<FIRRTLType>(arg.getType());

    auto oldSymAttr = portInfo[i].sym;
    auto newSymAttr =
        uniqueInNamespace(oldSymAttr, il.relocatedInnerSyms,
                          il.mic.modNamespace, target.getNameAttr());

    // Record the renamed port symbol on the active contexts' leaf hops: a
    // renamed port that is an NLA leaf must be reflected even when the port
    // itself carries no annotations.
    //
    // The path may be kept alive by an annotation elsewhere.
    updateVirtualNLALeafSymbols(il, oldSymAttr, newSymAttr);

    // The wire keeps the port's annotations verbatim; as with cloneAndRename,
    // the walk only records which contexts own them and P4 rewrites them.
    auto wireOp = WireOp::create(
        il.mic.b, target.getLoc(), type,
        StringAttr::get(context, (prefix + portInfo[i].getName())),
        NameKindEnumAttr::get(context, NameKindEnum::DroppableName),
        AnnotationSet::forPort(target, i).getArrayAttr(), newSymAttr,
        /*forceable=*/UnitAttr{});
    recordContexts(wireOp, il);
    Value wire = wireOp.getResult();
    il.wires.push_back(wire);
    mapper.map(arg, wire);
  }
}

/// Clone `op` at the mic builder's insertion point, rename it, record its
/// annotation contexts, and add it to the level.
void Inliner::cloneAndRename(StringRef prefix, InliningLevel &il,
                             IRMapping &mapper, Operation &op) {
  // Clone and rename.
  //
  // Annotations are copied verbatim: the walk only records which contexts own
  // them; the final writeback is the single annotation writer (P4), running
  // when every context's path is final.
  assert(op.getNumRegions() == 0 &&
         "operation with regions should not reach cloneAndRename");
  auto *newOp = il.mic.b.cloneWithoutRegions(op, mapper);

  // Instance renames must also land on the hierpath hops involved.
  if (isa<FInstanceLike>(&op))
    renameInstance(prefix, il, &op, newOp);
  else
    rename(prefix, newOp, il);

  recordContexts(newOp, il);

  il.newOps.push_back(newOp);
}

//===- P3: the walk -------------------------------------------------------===//

bool Inliner::shouldFlatten(FModuleLike mod) {
  return inliningFacts.hasFlatten(mod);
}

bool Inliner::shouldInline(FModuleLike mod) {
  return inliningFacts.hasInline(mod);
}

LogicalResult Inliner::inliningWalk(
    OpBuilder &builder, Block *block, IRMapping &mapper,
    llvm::function_ref<LogicalResult(Operation *op)> process) {
  /// Insertion points: target in the destination, source in the original.
  struct IPs {
    OpBuilder::InsertPoint target;
    Block::iterator source;
  };
  // Invariant: no Block::iterator == end(), can't getBlock().
  SmallVector<IPs> inliningStack;
  if (block->empty())
    return success();

  inliningStack.push_back(IPs{builder.saveInsertionPoint(), block->begin()});
  OpBuilder::InsertionGuard guard(builder);

  while (!inliningStack.empty()) {
    auto target = inliningStack.back().target;
    builder.restoreInsertionPoint(target);
    Operation *source;
    // Take the frame's next op; pop the frame once its block is exhausted.
    {
      auto &ips = inliningStack.back();
      source = &*ips.source;
      auto end = source->getBlock()->end();
      if (++ips.source == end)
        inliningStack.pop_back();
    }

    if (source->getNumRegions() == 0) {
      // `process` must leave the insertion point where it found it.
      assert(builder.saveInsertionPoint().getPoint() == target.getPoint());
      if (failed(process(source)))
        return failure();
      assert(builder.saveInsertionPoint().getPoint() == target.getPoint());

      continue;
    }

    // Limited support for region-containing operations.
    if (!isa<LayerBlockOp, WhenOp, MatchOp>(source))
      return source->emitError("unsupported operation '")
             << source->getName() << "' cannot be inlined";

    // Not cloneAndRename: nothing to prefix, no annotations, no symbols --
    // hence also absent from `newOps` and the level's inner-ref fixup.
    auto *newOp = builder.cloneWithoutRegions(*source, mapper);
    for (auto [newRegion, oldRegion] : llvm::reverse(
             llvm::zip_equal(newOp->getRegions(), source->getRegions()))) {
      if (oldRegion.empty()) {
        assert(newRegion.empty());
        continue;
      }
      // Single-block regions only, presently.
      assert(oldRegion.hasOneBlock());

      auto &oldBlock = oldRegion.getBlocks().front();
      auto &newBlock = newRegion.emplaceBlock();
      mapper.map(&oldBlock, &newBlock);

      for (auto arg : oldBlock.getArguments())
        mapper.map(arg, newBlock.addArgument(arg.getType(), arg.getLoc()));

      if (oldBlock.empty())
        continue;

      inliningStack.push_back(
          IPs{OpBuilder::InsertPoint(&newBlock, newBlock.begin()),
              oldBlock.begin()});
    }
  }
  return success();
}

LogicalResult Inliner::checkInstanceParents(InstanceOp instance) {
  auto *parent = instance->getParentOp();
  while (!isa<FModuleLike>(parent)) {
    if (!isa<LayerBlockOp>(parent))
      return instance->emitError("cannot inline instance")
                 .attachNote(parent->getLoc())
             << "containing operation '" << parent->getName()
             << "' not safe to inline into";
    parent = parent->getParentOp();
  }
  return success();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult Inliner::processInto(StringRef prefix, InliningLevel &il,
                                   IRMapping &mapper, bool flatten) {
  auto target = il.childModule;

  LLVM_DEBUG(llvm::dbgs() << (flatten ? "flattening " : "inlining ")
                          << target.getModuleName() << " into "
                          << il.mic.module.getModuleName() << "\n");

  auto visit = [&](Operation *op) {
    // If the pass can't inline through it, clone it and continue.
    auto instance = getInlinableInstance(op);
    if (!instance) {
      cloneAndRename(prefix, il, mapper, *op);
      return success();
    }

    // Not a regular module: uninlinable; the analysis marked it live.
    auto *moduleOp = symbolTable.lookup(instance.getModuleName());
    auto childModule = dyn_cast<FModuleOp>(moduleOp);
    if (!childModule) {
      assert(inliningFacts.isKnownLive(moduleOp) &&
             "a kept non-module instance must target a live module");
      cloneAndRename(prefix, il, mapper, *op);
      return success();
    }

    // Flatten inlines every child; otherwise only those marked for it.
    // A child the pass keeps is cloned as a live instance.
    if (!flatten && !shouldInline(childModule)) {
      assert(inliningFacts.isLive(childModule) &&
             "a kept child module must be live");
      cloneAndRename(prefix, il, mapper, *op);
      return success();
    }

    if (failed(checkInstanceParents(instance)))
      return failure();

    ++(flatten ? stats.instancesFlattened : stats.instancesInlined);

    InliningLevel childIL(il.mic, childModule);
    setActiveNLAsForChild(il.activeNLAs, childIL, instance);
    createDebugScope(childIL, instance, il.debugScope);

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, childIL, mapper);
    mapResultsToWires(mapper, childIL.wires, instance);

    // A flatten-marked child switches its whole subtree into flatten mode.
    if (failed(processInto(nestedPrefix, childIL, mapper,
                           flatten || shouldFlatten(childModule))))
      return failure();
    return childIL.finalize();
  };

  return inliningWalk(il.mic.b, target.getBodyBlock(), mapper, visit);
}

LogicalResult Inliner::processInstances(FModuleOp module, bool flatten) {
  auto moduleName = module.getNameAttr();
  ModuleInliningContext mic(module);

  LLVM_DEBUG(llvm::dbgs() << "inlining instances within " << moduleName
                          << "...\n");
  auto visit = [&](FInstanceLike instanceLike) {
    auto instance = getInlinableInstance(instanceLike.getOperation());
    if (!instance)
      return WalkResult::advance();
    // Not a regular module: uninlinable; the analysis marked it live.
    auto moduleOp = symbolTable.lookup<FModuleLike>(instance.getModuleName());
    assert(moduleOp && "instance target missing -- ran unverified?");
    auto target = dyn_cast<FModuleOp>(*moduleOp);
    if (!target) {
      assert(inliningFacts.isLive(moduleOp) &&
             "a kept non-module instance must target a live module");
      return WalkResult::advance();
    }

    // Flatten inlines every child; otherwise only those marked for it.
    if (!flatten && !shouldInline(target))
      return WalkResult::advance();

    if (failed(checkInstanceParents(instance)))
      return WalkResult::interrupt();

    ++(flatten ? stats.instancesFlattened : stats.instancesInlined);

    // Create the wire mapping for results + ports.
    // We RAUW the results instead of mapping them.
    IRMapping mapper;
    mic.b.setInsertionPoint(instance);

    InliningLevel childIL(mic, target);
    setActiveNLAsForChild(/* Activate all through this instance */ std::nullopt,
                          childIL, instance);
    createDebugScope(childIL, instance);

    auto nestedPrefix = (instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, childIL, mapper);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(childIL.wires[i]);

    // A flatten-marked child switches its whole subtree into flatten mode.
    if (failed(processInto(nestedPrefix, childIL, mapper,
                           flatten || shouldFlatten(target))) ||
        failed(childIL.finalize()))
      return WalkResult::interrupt();

    instance.erase();
    return WalkResult::skip();
  };

  return failure(module.getBodyBlock()
                     ->walk<mlir::WalkOrder::PreOrder>(visit)
                     .wasInterrupted());
}

void Inliner::createDebugScope(InliningLevel &il, InstanceOp instance,
                               Value parentScope) {
  auto op = debug::ScopeOp::create(
      il.mic.b, instance.getLoc(), instance.getInstanceNameAttr(),
      instance.getModuleNameAttr().getAttr(), parentScope);
  debugScopes.push_back(op);
  il.debugScope = op;
}

LogicalResult Inliner::inlineModules() {
  // Process live modules in the analysis's parents-before-children order
  // (I12): a parent always clones a child's pristine definition body, since
  // a retained child's own body is only mutated by its later self-visit.
  //
  // Dead modules are skipped here and erased after.
  for (auto moduleOp : inliningFacts.getSchedule()) {
    const auto &info = inliningFacts.getModuleInfo(moduleOp);
    if (!info.isLive)
      continue;
    // Consume the inline/flatten annotations: InliningFacts is their only
    // reader (everything else consults the frozen ModuleClassification, I1).
    //
    // Every P1/P2 diagnosis has already run, so a run that fails before this
    // loop leaves the input untouched; the walk's own diagnoses (instance
    // parents, foreign inner refs) fire mid-clone and do not.
    if (info.hasFlatten || info.hasInline)
      AnnotationSet::removeAnnotations(moduleOp, [](Annotation anno) {
        return anno.isClass(flattenAnnoClass, inlineAnnoClass);
      });
    if (failed(processInstances(moduleOp, info.hasFlatten)))
      return failure();
  }

  // Delete debug scopes that ended up unused.
  // Erase in reverse: back scopes may have uses on front scopes.
  for (auto scopeOp : llvm::reverse(debugScopes))
    if (scopeOp.use_empty())
      scopeOp.erase();
  debugScopes.clear();

  return success();
}

void Inliner::eraseDeadModules() {
  for (auto mod : llvm::make_early_inc_range(circuit.getOps<FModuleLike>())) {
    if (inliningFacts.isKnownLive(mod))
      continue;
    mod.erase();
    ++stats.deadModules;
  }
}

//===- P4: write back -----------------------------------------------------===//

ArrayAttr Inliner::materializeNamepath(VirtualNLA *vnla) {
  SmallVector<Attribute> pathAttrs;
  for (auto &hop : vnla->getPath()) {
    // Hops that originally had inner symbols must have their final symbols
    // assigned by now.  These were set during plan (in-place/terminal) or
    // filled in during cloning in P3.
    //
    // This going wrong will corrupt the deduplication mechanisms,
    // so be sure to check it.
    assert((hop.finalSym || !hop.origSym) &&
           "materializing a hop whose final symbol was never filled");
    if (hop.finalSym)
      pathAttrs.push_back(InnerRefAttr::get(hop.finalMod, hop.finalSym));
    else
      pathAttrs.push_back(FlatSymbolRefAttr::get(hop.finalMod));
  }
  return ArrayAttr::get(context, pathAttrs);
}

void Inliner::canonicalize(VirtualNLA *vnla) {
  // A local context never canonicalizes.
  //
  // The annotation localizes onto the op and the path is dropped.
  assert(!vnla->isLocal() && "local VNLAs have no hierpath to canonicalize");
  assert(!vnla->realizedSym && "context canonicalized twice");
  VirtualNLA *canon =
      canonicalByPath.try_emplace(materializeNamepath(vnla), vnla)
          .first->second;
  canonicalOf[vnla] = canon;
  if (canon == vnla) {
    // Only forks reach here; the primary claimed origSym before any fork
    // canonicalizes, so a canonical fork always mints (I15).
    assert(claimed.has(vnla->origSym) &&
           "primary claims origSym before any fork canonicalizes (I15)");
    vnla->realizedSym = StringAttr::get(
        context, circuitNamespace.newName(vnla->origSym.getValue()));
  }
}

void Inliner::appendContextAnno(Annotation anno, StringAttr origSym,
                                VirtualNLA *matched,
                                SmallVectorImpl<Attribute> &out) {
  if (matched->isLocal()) {
    anno.removeMember("circt.nonlocal");
    out.push_back(anno.getAttr());
    return;
  }
  matched->wasUsed = true;
  StringAttr canonSym = canonicalOrSelf(matched)->realizedSym;
  // Keep the annotation as-is if it already names that symbol.
  if (canonSym == origSym) {
    out.push_back(anno.getAttr());
    return;
  }
  anno.setMember("circt.nonlocal", FlatSymbolRefAttr::get(canonSym));
  out.push_back(anno.getAttr());
}

void Inliner::canonicalizeContexts() {
  // Serially, now that all paths are final (the walk is done).
  // Leaves `canonicalOf` complete and read-only for the parallel rewrite.
  //
  // Retention: an original hierpath symbol can be named by users the inliner
  // does not rewrite (an sv.xmr.ref target, a circuit-level annotation).
  //
  // Those are invisible here, so every origSym with a surviving context is kept
  // 1:1 rather than dropped when no annotation happens to name it: pinned to a
  // `primary` context and emitted unconditionally by writeback.
  //
  // Only fork symbols stay usedness-gated, since we created them.
  // GC of genuinely dead paths is not the job of this pass.
  //
  // Process one origSym group at a time (contiguous per I4).
  for (size_t i = 0, e = nlaPlanner.allVNLAs.size(); i < e;) {
    StringAttr origSym = nlaPlanner.allVNLAs[i]->origSym;
    size_t groupStart = i;
    while (i < e && nlaPlanner.allVNLAs[i]->origSym == origSym)
      ++i;
    ArrayRef<VirtualNLA *> group(&nlaPlanner.allVNLAs[groupStart],
                                 i - groupStart);

    // Pick the primary (I15): the first non-local context, else the front.
    // A non-local namepath best matches the source hierpath.
    // An all-local group still pins its symbol through its one-hop path.
    VirtualNLA *primary = nullptr;
    for (auto *v : group)
      if (!v->isLocal()) {
        primary = v;
        break;
      }
    if (!primary)
      primary = group.front();

    // The primary keeps origSym unconditionally and is its own canonical,
    // even when another origSym's primary already claimed this exact path.

    // Each original may have its own external user:
    // both must survive, so primaries never merge; only forks do.
    primary->realizedSym = origSym;
    claimed.claim(origSym);
    canonicalOf[primary] = primary;
    // Seed `canonicalByPath` so forks can still attach to a primary.
    if (!primary->isLocal())
      canonicalByPath.try_emplace(materializeNamepath(primary), primary);

    // Canonicalize the remaining forks (non-primary): dedup by path and mint
    // fresh names (origSym is already claimed).
    // A local fork is skipped:
    // an annotation on a local context simply drops the path.
    for (auto *v : group) {
      if (v == primary || v->isLocal())
        continue;
      canonicalize(v);
    }
  }
}

void Inliner::rewriteAnnotations() {
  // Cloned ops: owning contexts were recorded at clone time (I13).
  // Original ops (`recorded` == null): ownership alone selects them (I14).
  auto rewriteAnnos = [&](ArrayAttr annos, StringAttr modName,
                          const SmallVectorImpl<VirtualNLA *> *recorded,
                          SmallVectorImpl<Attribute> &newAnnos) {
    for (Attribute attr : annos) {
      Annotation anno(attr);
      auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
      if (!sym) {
        newAnnos.push_back(anno.getAttr());
        continue;
      }

      if (recorded) {
        // Cloned op: write exactly the recorded contexts for this symbol.
        // Anything unrecorded belongs to a different copy.
        for (auto *matched : *recorded)
          if (matched->origSym == sym.getAttr())
            appendContextAnno(anno, sym.getAttr(), matched, newAnnos);
        continue;
      }

      // Original op: annotations naming a nonexistent hierpath are dropped.
      // Otherwise rewrite per context this module owns.
      auto it = nlaPlanner.origToVNLAs.find(sym.getAttr());
      if (it == nlaPlanner.origToVNLAs.end())
        continue;
      for (auto *matched : it->second) {
        // Drop the annotation unless this module owns the context's leaf
        // (I14/I13); a path is never empty (I8); a local context is one hop,
        // not zero.
        if (matched->getPath().back().finalMod != modName)
          continue;
        appendContextAnno(anno, sym.getAttr(), matched, newAnnos);
      }
    }
  };

  auto rewriteOpAnnos = [&](Operation *op, StringAttr modName) {
    const SmallVectorImpl<VirtualNLA *> *recorded = nullptr;
    if (auto it = clonedAnnoContexts.find(op); it != clonedAnnoContexts.end())
      recorded = &it->second;

    // Update annotations on the op.
    // Skip ops without any, to avoid adding an empty annotations attribute.
    if (auto annos = getAnnotationsIfPresent(op); annos && !annos.empty()) {
      SmallVector<Attribute> newAnnotations;
      rewriteAnnos(annos, modName, recorded, newAnnotations);
      AnnotationSet(newAnnotations, context).applyToOperation(op);
    }

    // Update port annotations:
    // module ports and the per-port annotations of instances and memories.
    if (auto portAnnos = op->getAttrOfType<ArrayAttr>("portAnnotations")) {
      SmallVector<Attribute> newPortAnnotations;
      SmallVector<Attribute> newAnnotations;
      for (auto portArray : portAnnos.getAsRange<ArrayAttr>()) {
        newAnnotations.clear();
        rewriteAnnos(portArray, modName, recorded, newAnnotations);
        newPortAnnotations.push_back(ArrayAttr::get(context, newAnnotations));
      }
      op->setAttr("portAnnotations",
                  ArrayAttr::get(context, newPortAnnotations));
    }
  };
  auto rewriteModuleAnnos = [&](FModuleLike fmodule) {
    StringAttr modName = fmodule.getModuleNameAttr();
    fmodule.walk([&](Operation *op) { rewriteOpAnnos(op, modName); });
  };

  // Parallel per module: P2 state is read-only here (I2/I3).
  // Each context has a single writer (I14).
  // Only regular modules are worth a parallel task.
  // Other module-likes are each a trivial walk done serially, handled inline
  // (the same split as verifyInnerRefNamespace).
  SmallVector<FModuleOp> bodyModules;
  for (auto fmodule : circuit.getOps<FModuleLike>()) {
    if (auto regular = dyn_cast<FModuleOp>(*fmodule))
      bodyModules.push_back(regular);
    else
      rewriteModuleAnnos(fmodule);
  }
  mlir::parallelForEach(context, bodyModules, [&](FModuleOp fmodule) {
    rewriteModuleAnnos(fmodule);
  });
}

void Inliner::writebackHierPaths() {
  // Propagate usedness from duplicates onto their canonical, so a converged
  // path materializes even when only a duplicate's annotation kept it live.
  // Iterating `canonicalOf` unordered is safe: both effects are
  // order-independent (a count and a monotonic OR onto the canonical).
  for (auto &[dup, canon] : canonicalOf) {
    if (dup == canon)
      continue;
    ++stats.hierPathsMerged;
    if (dup->wasUsed)
      canon->wasUsed = true;
  }

#ifndef NDEBUG
  // I15: exactly one primary claimant per origSym group, emitted in place
  // below, so the symbol always survives for a user the inliner cannot see.
  // Supersedes the usedness-uniformity assert: emitted regardless of usedness,
  // the churn it guarded against cannot arise.
  for (size_t i = 0, e = nlaPlanner.allVNLAs.size(); i < e;) {
    StringAttr origSym = nlaPlanner.allVNLAs[i]->origSym;
    unsigned claimants = 0;
    for (; i < e && nlaPlanner.allVNLAs[i]->origSym == origSym; ++i) {
      auto *v = nlaPlanner.allVNLAs[i];
      if (v->realizedSym == origSym && canonicalOrSelf(v) == v)
        ++claimants;
    }
    assert(claimants == 1 &&
           "retention: each origSym must have exactly one primary claimant");
  }
#endif

  OpBuilder b(context);
  // Source symbol -> hierpath op, recorded by the planner.
  // (still valid: the ops are only mutated here)
  auto &existingPaths = nlaPlanner.hierPathOps;

  // Each used canonical context materializes next to its source op:
  //   realizedSym == origSym -> retarget the original op in place
  //   realizedSym is fresh   -> new private hw.hierpath beside the original
  //   local/duplicate/unused -> nothing; unreferenced originals are erased
  //
  // Iterate in creation order for determinism (contiguous per origSym).
  //
  // At a group boundary, set the insertion point after its hw.hierpath op
  // so forks land beside their source, not clumped at the block's end.
  StringAttr curGroup;

  // Ops kept alive by in-place reuse below.
  // A group can hold both a reusing context and later forks.
  // The forks still need the original's location, so keep its entry.
  DenseSet<StringAttr> retainedPaths;
  for (auto *vnla : nlaPlanner.allVNLAs) {
    if (vnla->origSym != curGroup) {
      curGroup = vnla->origSym;
      if (auto it = existingPaths.find(curGroup); it != existingPaths.end())
        b.setInsertionPointAfter(it->second);
    }

    // Duplicates are materialized by their canonical VNLA.
    if (canonicalOrSelf(vnla) != vnla)
      continue;
    // The primary is emitted unconditionally (I15).
    // A fork emits only when an annotation referenced it, and a local
    // non-primary has neither a minted symbol nor a path.
    bool isPrimary = vnla->realizedSym == vnla->origSym;
    if (!isPrimary && (vnla->isLocal() || !vnla->wasUsed))
      continue;

    auto arrayAttr = materializeNamepath(vnla);
    // The planner never invents an origSym; its source op is always present.
    auto origIt = existingPaths.find(vnla->origSym);
    assert(origIt != existingPaths.end() &&
           "origSym has no source hw.hierpath");

    // Same symbol survives -> mutate the original op in place.
    if (vnla->realizedSym == vnla->origSym) {
      // Count (and store) only real retargets, so a no-op run reports zero.
      if (arrayAttr != origIt->second.getNamepathAttr()) {
        origIt->second.setNamepathAttr(arrayAttr);
        ++stats.hierPathsUpdated;
      }
      retainedPaths.insert(vnla->origSym);
      continue;
    }

    // Forked into a fresh symbol: reuse the original op's location so the fork
    // keeps its provenance for diagnostics.
    auto hp = hw::HierPathOp::create(b, origIt->second.getLoc(),
                                     vnla->realizedSym, arrayAttr);
    hp.setPrivate();
    ++stats.hierPathsForked;
  }
  for (auto &[sym, deadPath] : existingPaths) {
    if (retainedPaths.contains(sym))
      continue;
    // Only a context-less (dead-rooted) origSym reaches here (I15).
    deadPath.erase();
    ++stats.hierPathsErased;
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
/// The FIRRTL inliner pass.
///
/// Runs InliningFacts (P1), NLAPlanner (P2), and Inliner (P3/P4) in sequence.
class InlinerPass : public circt::firrtl::impl::InlinerBase<InlinerPass> {
  using Base::Base;

  void runOnOperation() override {
    CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);
    auto circuit = getOperation();
    auto &symbolTable = getAnalysis<SymbolTable>();
    auto &instanceGraph = getAnalysis<InstanceGraph>();

    // Classify modules (P1).
    auto facts = InliningFacts::compute(circuit, instanceGraph, symbolTable);
    if (failed(facts))
      return signalPassFailure();
    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      debugHeader("InliningFacts results", 40) << "\n";
      facts->dump();
    });

    // Run NLA planning (P2).
    NLAPlanner nlaPlanner(circuit, symbolTable, instanceGraph, *facts);
    if (failed(nlaPlanner.run()))
      return signalPassFailure();
    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      debugHeader("NLA planner results", 40) << "\n";
      nlaPlanner.dump();
    });
    numHierPathsEndInnerSym += nlaPlanner.stats.endInnerSym;
    numHierPathsEndModule += nlaPlanner.stats.endModule;

    // Run Inlining: Clone (P3), and writeback (P4).
    CircuitNamespace circuitNamespace(circuit);
    Inliner inliner(circuit, symbolTable, circuitNamespace, *facts, nlaPlanner);
    if (failed(inliner.run()))
      signalPassFailure();

    numInstancesInlined += inliner.stats.instancesInlined;
    numInstancesFlattened += inliner.stats.instancesFlattened;
    numDeadModules += inliner.stats.deadModules;
    numHierPathsUpdated += inliner.stats.hierPathsUpdated;
    numHierPathsForked += inliner.stats.hierPathsForked;
    numHierPathsMerged += inliner.stats.hierPathsMerged;
    numHierPathsErased += inliner.stats.hierPathsErased;
  }
};
} // namespace
