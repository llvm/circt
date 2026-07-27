//===- FullReset.cpp - Implement full reset domains ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FullReset pass.
//
//===----------------------------------------------------------------------===//

#include "MemToRegOfVec.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/InstanceGraphInterface.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "full-reset"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_FULLRESET
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using circt::igraph::InstanceOpInterface;
using circt::igraph::InstancePath;
using circt::igraph::InstancePathCache;
using llvm::MapVector;
using llvm::SmallDenseSet;
using llvm::SmallSetVector;
using mlir::FailureOr;

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Return the name and parent module of a reset. The reset value must either be
/// a module port or a wire/node operation.
static std::pair<StringAttr, FModuleOp> getResetNameAndModule(Value reset) {
  if (auto arg = dyn_cast<BlockArgument>(reset)) {
    auto module = cast<FModuleOp>(arg.getParentRegion()->getParentOp());
    return {module.getPortNameAttr(arg.getArgNumber()), module};
  }
  auto *op = reset.getDefiningOp();
  return {op->getAttrOfType<StringAttr>("name"),
          op->getParentOfType<FModuleOp>()};
}

/// Return the name of a reset. The reset value must either be a module port or
/// a wire/node operation.
static StringAttr getResetName(Value reset) {
  return getResetNameAndModule(reset).first;
}

namespace {
/// A reset domain.
struct ResetDomain {
  /// Whether this is the root of the reset domain.
  bool isTop = false;

  /// The reset signal for this domain. A null value indicates that this domain
  /// explicitly has no reset.
  Value rootReset;

  /// The name of this reset signal.
  StringAttr resetName;
  /// The type of this reset signal.
  Type resetType;

  /// Implementation details for this domain. This will be the module local
  /// signal for this domain.
  Value localReset;
  /// If this module already has a port with the matching name, this holds the
  /// index of the port.
  std::optional<unsigned> existingPort;

  /// Create a reset domain without any reset.
  ResetDomain() = default;

  /// Create a reset domain associated with the root reset.
  ResetDomain(Value rootReset)
      : rootReset(rootReset), resetName(getResetName(rootReset)),
        resetType(rootReset.getType()) {}

  /// Returns true if this is in a reset domain, false if this is not a domain.
  explicit operator bool() const { return static_cast<bool>(rootReset); }
};
} // namespace

inline bool operator==(const ResetDomain &a, const ResetDomain &b) {
  return (a.isTop == b.isTop && a.resetName == b.resetName &&
          a.resetType == b.resetType);
}
inline bool operator!=(const ResetDomain &a, const ResetDomain &b) {
  return !(a == b);
}

/// Construct a zero value of the given type using the given builder.
static Value createZeroValue(ImplicitLocOpBuilder &builder, FIRRTLBaseType type,
                             SmallDenseMap<FIRRTLBaseType, Value> &cache) {
  // The zero value's type is a const version of `type`.
  type = type.getConstType(true);
  auto it = cache.find(type);
  if (it != cache.end())
    return it->second;
  auto nullBit = [&]() {
    return createZeroValue(
        builder, UIntType::get(builder.getContext(), 1, /*isConst=*/true),
        cache);
  };
  auto value =
      FIRRTLTypeSwitch<FIRRTLBaseType, Value>(type)
          .Case<ClockType>([&](auto type) {
            return AsClockPrimOp::create(builder, nullBit());
          })
          .Case<AsyncResetType>([&](auto type) {
            return AsAsyncResetPrimOp::create(builder, nullBit());
          })
          .Case<SIntType, UIntType>([&](auto type) {
            return ConstantOp::create(
                builder, type, APInt::getZero(type.getWidth().value_or(1)));
          })
          .Case<FEnumType>([&](auto type) -> Value {
            // There might not be a variant that corresponds to 0, in which case
            // we have to create a 0 value and bitcast it to the enum.
            if (type.getNumElements() != 0 &&
                type.getElement(0).value.getValue().isZero()) {
              const auto &element = type.getElement(0);
              auto value = createZeroValue(builder, element.type, cache);
              return FEnumCreateOp::create(builder, type, element.name, value);
            }
            auto value = ConstantOp::create(builder,
                                            UIntType::get(builder.getContext(),
                                                          type.getBitWidth(),
                                                          /*isConst=*/true),
                                            APInt::getZero(type.getBitWidth()));
            return BitCastOp::create(builder, type, value);
          })
          .Case<BundleType>([&](auto type) {
            auto wireOp = WireOp::create(builder, type);
            for (unsigned i = 0, e = type.getNumElements(); i < e; ++i) {
              auto fieldType = type.getElementTypePreservingConst(i);
              auto zero = createZeroValue(builder, fieldType, cache);
              auto acc =
                  SubfieldOp::create(builder, fieldType, wireOp.getResult(), i);
              emitConnect(builder, acc, zero);
            }
            return wireOp.getResult();
          })
          .Case<FVectorType>([&](auto type) {
            auto wireOp = WireOp::create(builder, type);
            auto zero = createZeroValue(
                builder, type.getElementTypePreservingConst(), cache);
            for (unsigned i = 0, e = type.getNumElements(); i < e; ++i) {
              auto acc = SubindexOp::create(builder, zero.getType(),
                                            wireOp.getResult(), i);
              emitConnect(builder, acc, zero);
            }
            return wireOp.getResult();
          })
          .Case<ResetType, AnalogType>(
              [&](auto type) { return InvalidValueOp::create(builder, type); })
          .Default([](auto) {
            llvm_unreachable("switch handles all types");
            return Value{};
          });
  cache.insert({type, value});
  return value;
}

/// Construct a null value of the given type using the given builder.
static Value createZeroValue(ImplicitLocOpBuilder &builder,
                             FIRRTLBaseType type) {
  SmallDenseMap<FIRRTLBaseType, Value> cache;
  return createZeroValue(builder, type, cache);
}

/// Helper function that inserts reset multiplexer into all `ConnectOp`s
/// with the given target. Looks through `SubfieldOp`, `SubindexOp`,
/// and `SubaccessOp`, and inserts multiplexers into connects to
/// these subaccesses as well. Modifies the insertion location of the builder.
/// Returns true if the `resetValue` was used in any way, false otherwise.
static bool insertResetMux(ImplicitLocOpBuilder &builder, Value target,
                           Value reset, Value resetValue) {
  // Indicates whether the `resetValue` was assigned to in some way. We use this
  // to erase unused subfield/subindex/subaccess ops on the reset value if they
  // end up unused.
  bool resetValueUsed = false;

  for (auto &use : target.getUses()) {
    Operation *useOp = use.getOwner();
    builder.setInsertionPoint(useOp);
    TypeSwitch<Operation *>(useOp)
        // Insert a mux on the value connected to the target:
        // connect(dst, src) -> connect(dst, mux(reset, resetValue, src))
        .Case<ConnectOp, MatchingConnectOp>([&](auto op) {
          if (op.getDest() != target)
            return;
          LLVM_DEBUG(llvm::dbgs() << "  - Insert mux into " << op << "\n");
          auto muxOp =
              MuxPrimOp::create(builder, reset, resetValue, op.getSrc());
          op.getSrcMutable().assign(muxOp);
          resetValueUsed = true;
        })
        // Look through subfields.
        .Case<SubfieldOp>([&](auto op) {
          auto resetSubValue =
              SubfieldOp::create(builder, resetValue, op.getFieldIndexAttr());
          if (insertResetMux(builder, op, reset, resetSubValue))
            resetValueUsed = true;
          else
            resetSubValue.erase();
        })
        // Look through subindices.
        .Case<SubindexOp>([&](auto op) {
          auto resetSubValue =
              SubindexOp::create(builder, resetValue, op.getIndexAttr());
          if (insertResetMux(builder, op, reset, resetSubValue))
            resetValueUsed = true;
          else
            resetSubValue.erase();
        })
        // Look through subaccesses.
        .Case<SubaccessOp>([&](auto op) {
          if (op.getInput() != target)
            return;
          auto resetSubValue =
              SubaccessOp::create(builder, resetValue, op.getIndex());
          if (insertResetMux(builder, op, reset, resetSubValue))
            resetValueUsed = true;
          else
            resetSubValue.erase();
        });
  }
  return resetValueUsed;
}


namespace {
/// Whether a reset is sync or async.
enum class ResetKind { Async, Sync };

static StringRef resetKindToStringRef(const ResetKind &kind) {
  switch (kind) {
  case ResetKind::Async:
    return "async";
  case ResetKind::Sync:
    return "sync";
  }
  llvm_unreachable("unhandled reset kind");
}
} // namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
/// Implement full reset domains annotated on the design.
///
/// Builds reset domains from FullReset / ExcludeFromFullReset annotations,
/// converts combinational memories in asynchronous domains to register
/// vectors, then implements full reset on registers and instance ports.
struct FullResetPass : public circt::firrtl::impl::FullResetBase<FullResetPass> {
  void runOnOperation() override;
  void runOnOperationInner();

  using FullResetBase::FullResetBase;
  FullResetPass(const FullResetPass &other) : FullResetBase(other) {}

  LogicalResult collectAnnos(CircuitOp circuit);
  // Collect reset annotations in the module and return a reset signal.
  // Return failure() if there was an error in the annotation processing.
  // Return nullopt if there was no reset annotation.
  // Return nullptr if there was ignore annotation.
  // Return a non-null Value if the reset was actually provided.
  FailureOr<std::optional<Value>> collectAnnos(FModuleOp module);

  LogicalResult buildDomains(CircuitOp circuit);
  void buildDomains(FModuleOp module, const InstancePath &instPath,
                    Value parentReset, InstanceGraph &instGraph,
                    unsigned indent = 0);

  void convertMemsInAsyncDomains();

  LogicalResult determineImpl();
  LogicalResult determineImpl(FModuleOp module, ResetDomain &domain);

  LogicalResult implementFullReset();
  LogicalResult implementFullReset(FModuleOp module, ResetDomain &domain);
  void implementFullReset(Operation *op, FModuleOp module, Value actualReset);

  // Helper to implement full reset for instance-like operations
  void implementFullReset(FInstanceLike inst, StringAttr moduleName,
                          Value actualReset);

  /// The annotated reset for a module. A null value indicates that the module
  /// is explicitly annotated with ignore. Otherwise the port/wire/node
  /// annotated as reset within the module is stored.
  DenseMap<Operation *, Value> annotatedResets;

  /// The reset domain for a module. In case of conflicting domain membership,
  /// the vector for a module contains multiple elements.
  MapVector<FModuleOp, SmallVector<std::pair<ResetDomain, InstancePath>, 1>>
      domains;

  /// Cache of modules symbols
  InstanceGraph *instanceGraph = nullptr;

  /// Cache of instance paths.
  std::unique_ptr<InstancePathCache> instancePathCache;
};
} // namespace

void FullResetPass::runOnOperation() {
  runOnOperationInner();
  annotatedResets.clear();
  domains.clear();
  instancePathCache.reset(nullptr);
  // InstanceGraph is kept up to date via replaceInstance during implementation.
  markAnalysesPreserved<InstanceGraph>();
}

void FullResetPass::runOnOperationInner() {
  instanceGraph = &getAnalysis<InstanceGraph>();
  instancePathCache = std::make_unique<InstancePathCache>(*instanceGraph);

  // Gather the reset annotations throughout the modules.
  if (failed(collectAnnos(getOperation())))
    return signalPassFailure();

  // Nothing to do if no FullReset / Exclude annotations were found.
  if (annotatedResets.empty())
    return;

  // Build the reset domains in the design.
  if (failed(buildDomains(getOperation())))
    return signalPassFailure();

  // Convert combinational memories in async full-reset domains so that FART
  // can attach resets to the resulting registers.
  convertMemsInAsyncDomains();

  // Determine how each reset shall be implemented.
  if (failed(determineImpl()))
    return signalPassFailure();

  // Implement the full resets.
  if (failed(implementFullReset()))
    return signalPassFailure();
}

void FullResetPass::convertMemsInAsyncDomains() {
  SmallVector<FModuleOp> asyncDomainMods;
  for (auto &[mod, entries] : domains) {
    if (entries.empty())
      continue;
    auto &domain = entries.back().first;
    if (!domain.rootReset)
      continue;
    if (!type_isa<AsyncResetType>(domain.resetType))
      continue;
    asyncDomainMods.push_back(mod);
  }
  if (asyncDomainMods.empty())
    return;

  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Convert comb mems in async full-reset domains") << "\n\n";
    for (auto mod : asyncDomainMods)
      llvm::dbgs() << "- " << mod.getName() << "\n";
  });

  unsigned converted = 0;
  convertCombMemsToRegOfVec(asyncDomainMods, ignoreReadEnable, converted);
  numConvertedMems = converted;
}

LogicalResult FullResetPass::collectAnnos(CircuitOp circuit) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Gather reset annotations") << "\n\n";
  });
  SmallVector<std::pair<FModuleOp, std::optional<Value>>> results;
  for (auto module : circuit.getOps<FModuleOp>())
    results.push_back({module, {}});
  // Collect annotations parallelly.
  if (failed(mlir::failableParallelForEach(
          circuit.getContext(), results, [&](auto &moduleAndResult) {
            auto result = collectAnnos(moduleAndResult.first);
            if (failed(result))
              return failure();
            moduleAndResult.second = *result;
            return success();
          })))
    return failure();

  for (auto [module, reset] : results)
    if (reset.has_value())
      annotatedResets.insert({module, *reset});
  return success();
}

FailureOr<std::optional<Value>>
FullResetPass::collectAnnos(FModuleOp module) {
  bool anyFailed = false;
  SmallSetVector<std::pair<Annotation, Location>, 4> conflictingAnnos;

  // Consume a possible "ignore" annotation on the module itself, which
  // explicitly assigns it no reset domain.
  bool ignore = false;
  AnnotationSet::removeAnnotations(module, [&](Annotation anno) {
    if (anno.isClass(excludeFromFullResetAnnoClass)) {
      ignore = true;
      conflictingAnnos.insert({anno, module.getLoc()});
      return true;
    }
    if (anno.isClass(fullResetAnnoClass)) {
      anyFailed = true;
      module.emitError("''FullResetAnnotation' cannot target module; must "
                       "target port or wire/node instead");
      return true;
    }
    return false;
  });
  if (anyFailed)
    return failure();

  // Consume any reset annotations on module ports.
  Value reset;
  // Helper for checking annotations and determining the reset
  auto checkAnnotations = [&](Annotation anno, Value arg) {
    if (anno.isClass(fullResetAnnoClass)) {
      ResetKind expectedResetKind;
      if (auto rt = anno.getMember<StringAttr>("resetType")) {
        if (rt == "sync") {
          expectedResetKind = ResetKind::Sync;
        } else if (rt == "async") {
          expectedResetKind = ResetKind::Async;
        } else {
          mlir::emitError(arg.getLoc(),
                          "'FullResetAnnotation' requires resetType == 'sync' "
                          "| 'async', but got resetType == ")
              << rt;
          anyFailed = true;
          return true;
        }
      } else {
        mlir::emitError(arg.getLoc(),
                        "'FullResetAnnotation' requires resetType == "
                        "'sync' | 'async', but got no resetType");
        anyFailed = true;
        return true;
      }
      // Check that the type is well-formed
      bool isAsync = expectedResetKind == ResetKind::Async;
      bool validUint = false;
      if (auto uintT = dyn_cast<UIntType>(arg.getType()))
        validUint = uintT.getWidth() == 1;
      if ((isAsync && !isa<AsyncResetType>(arg.getType())) ||
          (!isAsync && !validUint)) {
        auto kind = resetKindToStringRef(expectedResetKind);
        mlir::emitError(arg.getLoc(),
                        "'FullResetAnnotation' with resetType == '")
            << kind << "' must target " << kind << " reset, but targets "
            << arg.getType();
        anyFailed = true;
        return true;
      }

      reset = arg;
      conflictingAnnos.insert({anno, reset.getLoc()});

      return false;
    }
    if (anno.isClass(excludeFromFullResetAnnoClass)) {
      anyFailed = true;
      mlir::emitError(arg.getLoc(),
                      "'ExcludeFromFullResetAnnotation' cannot "
                      "target port/wire/node; must target module instead");
      return true;
    }
    return false;
  };

  AnnotationSet::removePortAnnotations(module,
                                       [&](unsigned argNum, Annotation anno) {
                                         Value arg = module.getArgument(argNum);
                                         return checkAnnotations(anno, arg);
                                       });
  if (anyFailed)
    return failure();

  // Consume any reset annotations on wires in the module body.
  module.getBody().walk([&](Operation *op) {
    // Reset annotations must target wire/node ops.
    if (!isa<WireOp, NodeOp>(op)) {
      if (AnnotationSet::hasAnnotation(op, fullResetAnnoClass,
                                       excludeFromFullResetAnnoClass)) {
        anyFailed = true;
        op->emitError(
            "reset annotations must target module, port, or wire/node");
      }
      return;
    }

    // At this point we know that we have a WireOp/NodeOp. Process the reset
    // annotations.
    AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
      auto arg = op->getResult(0);
      return checkAnnotations(anno, arg);
    });
  });
  if (anyFailed)
    return failure();

  // If we have found no annotations, there is nothing to do. We just leave
  // this module unannotated, which will cause it to inherit a reset domain
  // from its instantiation sites.
  if (!ignore && !reset) {
    LLVM_DEBUG(llvm::dbgs()
               << "No reset annotation for " << module.getName() << "\n");
    return std::optional<Value>();
  }

  // If we have found multiple annotations, emit an error and abort.
  if (conflictingAnnos.size() > 1) {
    auto diag = module.emitError("multiple reset annotations on module '")
                << module.getName() << "'";
    for (auto &annoAndLoc : conflictingAnnos)
      diag.attachNote(annoAndLoc.second)
          << "conflicting " << annoAndLoc.first.getClassAttr() << ":";
    return failure();
  }

  // Dump some information in debug builds.
  LLVM_DEBUG({
    llvm::dbgs() << "Annotated reset for " << module.getName() << ": ";
    if (ignore)
      llvm::dbgs() << "no domain\n";
    else if (auto arg = dyn_cast<BlockArgument>(reset))
      llvm::dbgs() << "port " << module.getPortName(arg.getArgNumber()) << "\n";
    else
      llvm::dbgs() << "wire "
                   << reset.getDefiningOp()->getAttrOfType<StringAttr>("name")
                   << "\n";
  });

  // Store the annotated reset for this module.
  assert(ignore || reset);
  return std::optional<Value>(reset);
}

//===----------------------------------------------------------------------===//
// Domain Construction
//===----------------------------------------------------------------------===//

/// Gather the reset domains present in a circuit. This traverses the instance
/// hierarchy of the design, making instances either live in a new reset
/// domain if so annotated, or inherit their parent's domain. This can go
/// wrong in some cases, mainly when a module is instantiated multiple times
/// within different reset domains.
LogicalResult FullResetPass::buildDomains(CircuitOp circuit) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Build full reset domains") << "\n\n";
  });

  // Gather the domains.
  auto &instGraph = getAnalysis<InstanceGraph>();
  // Walk all top-level modules.
  instGraph.walkPostOrder([&](igraph::InstanceGraphNode &node) {
    if (!node.noUses())
      return;
    if (auto module =
            dyn_cast_or_null<FModuleOp>(node.getModule().getOperation()))
      buildDomains(module, InstancePath{}, Value{}, instGraph);
  });

  // Report any domain conflicts among the modules.
  bool anyFailed = false;
  for (auto &it : domains) {
    auto module = cast<FModuleOp>(it.first);
    auto &domainConflicts = it.second;
    if (domainConflicts.size() <= 1)
      continue;

    anyFailed = true;
    SmallDenseSet<Value> printedDomainResets;
    auto diag = module.emitError("module '")
                << module.getName()
                << "' instantiated in different reset domains";
    for (auto &it : domainConflicts) {
      ResetDomain &domain = it.first;
      const auto &path = it.second;
      auto inst = path.leaf();
      auto loc = path.empty() ? module.getLoc() : inst.getLoc();
      auto &note = diag.attachNote(loc);

      // Describe the instance itself.
      if (path.empty())
        note << "root instance";
      else {
        note << "instance '";
        llvm::interleave(
            path,
            [&](InstanceOpInterface inst) { note << inst.getInstanceName(); },
            [&]() { note << "/"; });
        note << "'";
      }

      // Describe the reset domain the instance is in.
      note << " is in";
      if (domain.rootReset) {
        auto nameAndModule = getResetNameAndModule(domain.rootReset);
        note << " reset domain rooted at '" << nameAndModule.first.getValue()
             << "' of module '" << nameAndModule.second.getName() << "'";

        // Show where the domain reset is declared (once per reset).
        if (printedDomainResets.insert(domain.rootReset).second) {
          diag.attachNote(domain.rootReset.getLoc())
              << "reset domain '" << nameAndModule.first.getValue()
              << "' of module '" << nameAndModule.second.getName()
              << "' declared here:";
        }
      } else
        note << " no reset domain";
    }
  }
  return failure(anyFailed);
}

void FullResetPass::buildDomains(FModuleOp module,
                                   const InstancePath &instPath,
                                   Value parentReset, InstanceGraph &instGraph,
                                   unsigned indent) {
  LLVM_DEBUG({
    llvm::dbgs().indent(indent * 2) << "Visiting ";
    if (instPath.empty())
      llvm::dbgs() << "$root";
    else
      llvm::dbgs() << instPath.leaf().getInstanceName();
    llvm::dbgs() << " (" << module.getName() << ")\n";
  });

  // Assemble the domain for this module.
  ResetDomain domain;
  auto it = annotatedResets.find(module);
  if (it != annotatedResets.end()) {
    // If there is an actual reset, use it for our domain. Otherwise, our
    // module is explicitly marked to have no domain.
    if (auto localReset = it->second)
      domain = ResetDomain(localReset);
    domain.isTop = true;
  } else if (parentReset) {
    // Otherwise, we default to using the reset domain of our parent.
    domain = ResetDomain(parentReset);
  }

  // Associate the domain with this module. Only record non-null reset domains;
  // the `domains[module]` entry is created regardless, so modules in no-domain
  // contexts will have an empty entries list. If the module already has an
  // entry for this domain, don't add a duplicate.
  auto &entries = domains[module];
  if (domain.rootReset)
    if (llvm::all_of(entries,
                     [&](const auto &entry) { return entry.first != domain; }))
      entries.push_back({domain, instPath});

  // Traverse the child instances.
  for (auto *record : *instGraph[module]) {
    auto submodule = dyn_cast<FModuleOp>(*record->getTarget()->getModule());
    if (!submodule)
      continue;
    auto childPath =
        instancePathCache->appendInstance(instPath, record->getInstance());
    buildDomains(submodule, childPath, domain.rootReset, instGraph, indent + 1);
  }
}

/// Determine how the reset for each module shall be implemented.
LogicalResult FullResetPass::determineImpl() {
  auto anyFailed = false;
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Determine implementation") << "\n\n";
  });
  for (auto &it : domains) {
    auto module = cast<FModuleOp>(it.first);
    auto &entries = it.second;
    // Skip modules with no reset domain (empty entries).
    if (entries.empty())
      continue;
    auto &domain = entries.back().first;
    if (failed(determineImpl(module, domain)))
      anyFailed = true;
  }
  return failure(anyFailed);
}

/// Determine how the reset for a module shall be implemented. This function
/// fills in the `localReset` and `existingPort` fields of the given reset
/// domain.
///
/// Generally it does the following:
/// - If the domain has explicitly no reset ("ignore"), leaves everything
/// empty.
/// - If the domain is the place where the reset is defined ("top"), fills in
///   the existing port/wire/node as reset.
/// - If the module already has a port with the reset's name:
///   - If the port has the same name and type as the reset domain, reuses that
///   port.
///   - Otherwise errors out.
/// - Otherwise indicates that a port with the reset's name should be created.
///
LogicalResult FullResetPass::determineImpl(FModuleOp module,
                                             ResetDomain &domain) {
  // Nothing to do if the module needs no reset.
  if (!domain)
    return success();
  LLVM_DEBUG(llvm::dbgs() << "Planning reset for " << module.getName() << "\n");

  // If this is the root of a reset domain, we don't need to add any ports
  // and can just simply reuse the existing values.
  if (domain.isTop) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Rooting at local value " << domain.resetName << "\n");
    domain.localReset = domain.rootReset;
    if (auto blockArg = dyn_cast<BlockArgument>(domain.rootReset))
      domain.existingPort = blockArg.getArgNumber();
    return success();
  }

  // Otherwise, check if a port with this name and type already exists and
  // reuse that where possible.
  auto neededName = domain.resetName;
  auto neededType = domain.resetType;
  LLVM_DEBUG(llvm::dbgs() << "- Looking for existing port " << neededName
                          << "\n");
  auto portNames = module.getPortNames();
  auto *portIt = llvm::find(portNames, neededName);

  // If this port does not yet exist, record that we need to create it.
  if (portIt == portNames.end()) {
    LLVM_DEBUG(llvm::dbgs() << "- Creating new port " << neededName << "\n");
    domain.resetName = neededName;
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "- Reusing existing port " << neededName << "\n");

  // If this port has the wrong type, then error out.
  auto portNo = std::distance(portNames.begin(), portIt);
  auto portType = module.getPortType(portNo);
  if (portType != neededType) {
    auto diag = emitError(module.getPortLocation(portNo), "module '")
                << module.getName() << "' is in reset domain requiring port '"
                << domain.resetName.getValue() << "' to have type "
                << domain.resetType << ", but has type " << portType;
    diag.attachNote(domain.rootReset.getLoc()) << "reset domain rooted here";
    return failure();
  }

  // We have a pre-existing port which we should use.
  domain.existingPort = portNo;
  domain.localReset = module.getArgument(portNo);
  return success();
}

//===----------------------------------------------------------------------===//
// Full Reset Implementation
//===----------------------------------------------------------------------===//

/// Implement the annotated resets gathered in the pass' `domains` map.
LogicalResult FullResetPass::implementFullReset() {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Implement full resets") << "\n\n";
  });
  for (auto &it : domains) {
    auto module = cast<FModuleOp>(it.first);
    auto &entries = it.second;
    // For modules with a real domain, use that domain. For no-domain modules,
    // use a default empty domain but still process for tie-off.
    ResetDomain domain;
    if (!entries.empty())
      domain = entries.back().first;
    if (failed(implementFullReset(module, domain)))
      return failure();
  }
  return success();
}

/// Implement the async resets for a specific module.
///
/// This will add ports to the module as appropriate, update the register ops
/// in the module, and update any instantiated submodules with their
/// corresponding reset implementation details.
LogicalResult FullResetPass::implementFullReset(FModuleOp module,
                                                  ResetDomain &domain) {
  // For modules in no-domain contexts, we skip local transformations (adding
  // reset ports, converting registers) but still process instances to tie off
  // reset ports of children that have a real reset domain.
  if (!domain) {
    SmallVector<FInstanceLike> instances;
    module.walk([&](FInstanceLike instOp) { instances.push_back(instOp); });
    LLVM_DEBUG({
      if (!instances.empty())
        llvm::dbgs() << "Tie off instances in " << module.getName() << "\n";
    });
    for (auto instOp : instances)
      implementFullReset(instOp, module, Value());
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "Implementing full reset for " << module.getName()
                          << "\n");

  // Add an annotation indicating that this module belongs to a reset domain.
  auto *context = module.getContext();
  AnnotationSet annotations(module);
  annotations.addAnnotations(DictionaryAttr::get(
      context, NamedAttribute(StringAttr::get(context, "class"),
                              StringAttr::get(context, fullResetAnnoClass))));
  annotations.applyToOperation(module);

  // If needed, add a reset port to the module.
  auto actualReset = domain.localReset;
  if (!domain.localReset) {
    PortInfo portInfo{domain.resetName,
                      domain.resetType,
                      Direction::In,
                      {},
                      domain.rootReset.getLoc()};
    module.insertPorts({{0, portInfo}});
    actualReset = module.getArgument(0);
    LLVM_DEBUG(llvm::dbgs() << "- Inserted port " << domain.resetName << "\n");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "- Using ";
    if (auto blockArg = dyn_cast<BlockArgument>(actualReset))
      llvm::dbgs() << "port #" << blockArg.getArgNumber() << " ";
    else
      llvm::dbgs() << "wire/node ";
    llvm::dbgs() << getResetName(actualReset) << "\n";
  });

  // Gather a list of operations in the module that need to be updated with
  // the new reset.
  SmallVector<Operation *> opsToUpdate;
  module.walk([&](Operation *op) {
    if (isa<FInstanceLike, RegOp, RegResetOp>(op))
      opsToUpdate.push_back(op);
  });

  // If the reset is a local wire or node, move it upwards such that it
  // dominates all the operations that it will need to attach to. In the case
  // of a node this might not be easily possible, so we just spill into a wire
  // in that case.
  if (!isa<BlockArgument>(actualReset)) {
    mlir::DominanceInfo dom(module);
    // The first op in `opsToUpdate` is the top-most op in the module, since
    // the ops and blocks are traversed in a depth-first, top-to-bottom order
    // in `walk`. So we can simply check if the local reset declaration is
    // before the first op to find out if we need to move anything.
    auto *resetOp = actualReset.getDefiningOp();
    if (!opsToUpdate.empty() && !dom.dominates(resetOp, opsToUpdate[0])) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- Reset doesn't dominate all uses, needs to be moved\n");

      // If the node can't be moved because its input doesn't dominate the
      // target location, convert it to a wire.
      auto nodeOp = dyn_cast<NodeOp>(resetOp);
      if (nodeOp && !dom.dominates(nodeOp.getInput(), opsToUpdate[0])) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Promoting node to wire for move: " << nodeOp << "\n");
        auto builder = ImplicitLocOpBuilder::atBlockBegin(nodeOp.getLoc(),
                                                          nodeOp->getBlock());
        auto wireOp = WireOp::create(
            builder, nodeOp.getResult().getType(), nodeOp.getNameAttr(),
            nodeOp.getNameKindAttr(), nodeOp.getAnnotationsAttr(),
            nodeOp.getInnerSymAttr(), nodeOp.getForceableAttr());
        // Don't delete the node, since it might be in use in worklists.
        nodeOp->replaceAllUsesWith(wireOp);
        nodeOp->removeAttr(nodeOp.getInnerSymAttrName());
        nodeOp.setName("");
        // Leave forcable alone, since we cannot remove a result.  It will be
        // cleaned up in canonicalization since it is dead.  As will this node.
        nodeOp.setNameKind(NameKindEnum::DroppableName);
        nodeOp.setAnnotationsAttr(ArrayAttr::get(builder.getContext(), {}));
        builder.setInsertionPointAfter(nodeOp);
        emitConnect(builder, wireOp.getResult(), nodeOp.getResult());
        resetOp = wireOp;
        actualReset = wireOp.getResult();
        domain.localReset = wireOp.getResult();
      }

      // Determine the block into which the reset declaration needs to be
      // moved.
      Block *targetBlock = dom.findNearestCommonDominator(
          resetOp->getBlock(), opsToUpdate[0]->getBlock());
      LLVM_DEBUG({
        if (targetBlock != resetOp->getBlock())
          llvm::dbgs() << "- Needs to be moved to different block\n";
      });

      // At this point we have to figure out in front of which operation in
      // the target block the reset declaration has to be moved. The reset
      // declaration and the first op it needs to dominate may be buried
      // inside blocks of other operations (e.g. `WhenOp`), so we have to look
      // through their parent operations until we find the one that lies
      // within the target block.
      auto getParentInBlock = [](Operation *op, Block *block) {
        while (op && op->getBlock() != block)
          op = op->getParentOp();
        return op;
      };
      auto *resetOpInTarget = getParentInBlock(resetOp, targetBlock);
      auto *firstOpInTarget = getParentInBlock(opsToUpdate[0], targetBlock);

      // Move the operation upwards. Since there are situations where the
      // reset declaration does not dominate the first use, but the `WhenOp`
      // it is nested within actually *does* come before that use, we have to
      // consider moving the reset declaration in front of its parent op.
      if (resetOpInTarget->isBeforeInBlock(firstOpInTarget))
        resetOp->moveBefore(resetOpInTarget);
      else
        resetOp->moveBefore(firstOpInTarget);
    }
  }

  // Update the operations.
  for (auto *op : opsToUpdate)
    implementFullReset(op, module, actualReset);

  return success();
}

/// Helper to implement full reset for instance-like operations.
/// This handles the common logic of adding reset ports and connecting them.
void FullResetPass::implementFullReset(FInstanceLike inst,
                                         StringAttr moduleName,
                                         Value actualReset) {
  // Lookup the reset domain of the default target module. If there is no
  // reset domain associated with that module, as indicated by an empty list
  // of domains, simply skip it.
  auto *node = instanceGraph->lookup(moduleName);
  auto refModule = dyn_cast<FModuleOp>(*node->getModule());
  if (!refModule)
    return;
  auto *domainIt = domains.find(refModule);
  if (domainIt == domains.end() || domainIt->second.empty())
    return;
  auto &domain = domainIt->second.back().first;
  assert(domain && "null domains should not be listed");

  ImplicitLocOpBuilder builder(inst.getLoc(), inst);

  LLVM_DEBUG(llvm::dbgs() << (actualReset ? "- Update " : "- Tie-off ")
                          << inst->getName() << " '" << inst.getInstanceName()
                          << "'\n");

  // If needed, add a reset port to the instance.
  Value instReset;
  if (!domain.localReset) {
    LLVM_DEBUG(llvm::dbgs() << "  - Adding new result as reset\n");
    auto newInstOp = inst.cloneWithInsertedPortsAndReplaceUses(
        {{/*portIndex=*/0,
          {domain.resetName, domain.resetType, Direction::In}}});
    instReset = newInstOp->getResult(0);
    instanceGraph->replaceInstance(inst, newInstOp);
    inst->erase();
    inst = newInstOp;
  } else if (domain.existingPort.has_value()) {
    auto idx = *domain.existingPort;
    instReset = inst->getResult(idx);
    LLVM_DEBUG(llvm::dbgs() << "  - Using result #" << idx << " as reset\n");
  }

  // If there's no reset port on the instance to connect, we're done. This
  // can happen if the instantiated module has a reset domain, but that
  // domain is e.g. rooted at an internal wire.
  if (!instReset)
    return;

  builder.setInsertionPointAfter(inst);

  // If the module that contains the instance is not in a reset domain, as
  // indicated by actualReset being null, create a tie-off constant which
  // effectively turns the no-reset registers that had full resets added back
  // into no-reset registers.
  if (!actualReset) {
    LLVM_DEBUG(llvm::dbgs() << "  - Tying off reset to constant 0\n");
    if (type_isa<AsyncResetType>(domain.resetType))
      actualReset = SpecialConstantOp::create(builder, domain.resetType, false);
    else
      actualReset = ConstantOp::create(
          builder, UIntType::get(builder.getContext(), 1), APInt(1, 0));
  }

  // Connect the instance's reset to the actual reset or tie-off.
  assert(instReset && actualReset);
  emitConnect(builder, instReset, actualReset);
}

/// Modify an operation in a module to implement an full reset for that
/// module. If actualReset is null and op is an `InstanceOp`, creates a tie-off
/// constant for added reset ports. If the op is not an instance, aborts.
void FullResetPass::implementFullReset(Operation *op, FModuleOp module,
                                         Value actualReset) {
  ImplicitLocOpBuilder builder(op->getLoc(), op);

  // Handle instances.
  if (auto instOp = dyn_cast<FInstanceLike>(op))
    return implementFullReset(
        instOp, cast<StringAttr>(instOp.getReferencedModuleNamesAttr()[0]),
        actualReset);

  // All other ops require an actual reset. We only ever call this function with
  // null actualReset to create tie-offs on instance ops.
  assert(actualReset);

  // Handle reset-less registers.
  if (auto regOp = dyn_cast<RegOp>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "- Adding full reset to " << regOp << "\n");
    auto zero = createZeroValue(builder, regOp.getResult().getType());
    auto newRegOp = RegResetOp::create(
        builder, regOp.getResult().getType(), regOp.getClockVal(), actualReset,
        zero, regOp.getNameAttr(), regOp.getNameKindAttr(),
        regOp.getAnnotations(), regOp.getInnerSymAttr(),
        regOp.getForceableAttr());
    regOp.getResult().replaceAllUsesWith(newRegOp.getResult());
    if (regOp.getForceable())
      regOp.getRef().replaceAllUsesWith(newRegOp.getRef());
    regOp->erase();
    return;
  }

  // Handle registers with reset.
  if (auto regOp = dyn_cast<RegResetOp>(op)) {
    // If the register already has an async reset or if the type of the added
    // reset is sync, leave it alone.
    if (type_isa<AsyncResetType>(regOp.getResetSignal().getType()) ||
        type_isa<UIntType>(actualReset.getType())) {
      LLVM_DEBUG(llvm::dbgs() << "- Skipping (has reset) " << regOp << "\n");
      // The following performs the logic of `CheckResets` in the original
      // Scala source code.
      if (failed(regOp.verifyInvariants()))
        signalPassFailure();
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "- Updating reset of " << regOp << "\n");

    auto reset = regOp.getResetSignal();
    auto value = regOp.getResetValue();

    // If we arrive here, the register has a sync reset and the added reset is
    // async. In order to add an async reset, we have to move the sync reset
    // into a mux in front of the register.
    insertResetMux(builder, regOp.getResult(), reset, value);
    builder.setInsertionPointAfterValue(regOp.getResult());
    auto mux = MuxPrimOp::create(builder, reset, value, regOp.getResult());
    emitConnect(builder, regOp.getResult(), mux);

    // Replace the existing reset with the async reset.
    builder.setInsertionPoint(regOp);
    auto zero = createZeroValue(builder, regOp.getResult().getType());
    regOp.getResetSignalMutable().assign(actualReset);
    regOp.getResetValueMutable().assign(zero);
  }
}

