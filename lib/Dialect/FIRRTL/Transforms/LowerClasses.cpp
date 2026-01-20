//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerClasses pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/OwningModuleCache.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERCLASSES
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::firrtl;
using namespace circt::om;

namespace {

static bool shouldCreateClassImpl(igraph::InstanceGraphNode *node) {
  auto moduleLike = node->getModule<FModuleLike>();
  if (!moduleLike)
    return false;

  if (isa<firrtl::ClassLike>(moduleLike.getOperation()))
    return true;

  // Always create a class for public modules.
  if (moduleLike.isPublic())
    return true;

  // Create a class for modules with property ports.
  bool hasClassPorts = llvm::any_of(moduleLike.getPorts(), [](PortInfo port) {
    return isa<PropertyType>(port.type);
  });

  if (hasClassPorts)
    return true;

  // Create a class for modules that instantiate classes or modules with
  // property ports.
  for (auto *instance : *node) {
    if (auto op = instance->getInstance<FInstanceLike>())
      for (auto result : op->getResults())
        if (type_isa<PropertyType>(result.getType()))
          return true;
  }

  return false;
}

/// Helper class which holds a hierarchical path op reference and a pointer to
/// to the targeted operation.
struct PathInfo {
  PathInfo() = default;
  PathInfo(Location loc, bool canBeInstanceTarget, FlatSymbolRefAttr symRef,
           StringAttr altBasePathModule)
      : loc(loc), canBeInstanceTarget(canBeInstanceTarget), symRef(symRef),
        altBasePathModule(altBasePathModule) {
    assert(symRef && "symRef must not be null");
  }

  /// The Location of the hardware component targeted by this path.
  std::optional<Location> loc = std::nullopt;

  /// Flag to indicate if the hardware component can be targeted as an instance.
  bool canBeInstanceTarget = false;

  /// A reference to the hierarchical path targeting the op.
  FlatSymbolRefAttr symRef = nullptr;

  /// The module name of the root module from which we take an alternative base
  /// path.
  StringAttr altBasePathModule = nullptr;
};

/// Maps a FIRRTL path id to the lowered PathInfo.
struct PathInfoTable {
  // Add an alternative base path root module. The default base path from this
  // module will be passed through to where it is needed.
  void addAltBasePathRoot(StringAttr rootModuleName) {
    altBasePathRoots.insert(rootModuleName);
  }

  // Add a passthrough module for a given root module. The default base path
  // from the root module will be passed through the passthrough module.
  void addAltBasePathPassthrough(StringAttr passthroughModuleName,
                                 StringAttr rootModuleName) {
    auto &rootSequence = altBasePathsPassthroughs[passthroughModuleName];
    rootSequence.push_back(rootModuleName);
  }

  // Get an iterator range over the alternative base path root module names.
  llvm::iterator_range<SmallPtrSetImpl<StringAttr>::iterator>
  getAltBasePathRoots() const {
    return llvm::make_range(altBasePathRoots.begin(), altBasePathRoots.end());
  }

  // Get the number of alternative base paths passing through the given
  // passthrough module.
  size_t getNumAltBasePaths(StringAttr passthroughModuleName) const {
    return altBasePathsPassthroughs.lookup(passthroughModuleName).size();
  }

  // Get the root modules that are passing an alternative base path through the
  // given passthrough module.
  llvm::iterator_range<const StringAttr *>
  getRootsForPassthrough(StringAttr passthroughModuleName) const {
    auto it = altBasePathsPassthroughs.find(passthroughModuleName);
    assert(it != altBasePathsPassthroughs.end() &&
           "expected passthrough module to already exist");
    return llvm::make_range(it->second.begin(), it->second.end());
  }

  // Collect alternative base paths passing through `instance`, by looking up
  // its associated `moduleNameAttr`. The results are collected in `result`.
  void collectAltBasePaths(Operation *instance, StringAttr moduleNameAttr,
                           SmallVectorImpl<Value> &result) const {
    auto altBasePaths = altBasePathsPassthroughs.lookup(moduleNameAttr);
    auto parent = instance->getParentOfType<om::ClassOp>();

    // Handle each alternative base path for instances of this module-like.
    for (auto [i, altBasePath] : llvm::enumerate(altBasePaths)) {
      if (parent.getName().starts_with(altBasePath)) {
        // If we are passing down from the root, take the root base path.
        result.push_back(instance->getBlock()->getArgument(0));
      } else {
        // Otherwise, pass through the appropriate base path from above.
        // + 1 to skip default base path
        auto basePath = instance->getBlock()->getArgument(1 + i);
        assert(isa<om::BasePathType>(basePath.getType()) &&
               "expected a passthrough base path");
        result.push_back(basePath);
      }
    }
  }

  // The table mapping DistinctAttrs to PathInfo structs.
  DenseMap<DistinctAttr, PathInfo> table;

private:
  // Module name attributes indicating modules whose base path input should
  // be used as alternate base paths.
  SmallPtrSet<StringAttr, 16> altBasePathRoots;

  // Module name attributes mapping from modules who pass through alternative
  // base paths from their parents to a sequence of the parents' module names.
  DenseMap<StringAttr, SmallVector<StringAttr>> altBasePathsPassthroughs;
};

/// The suffix to append to lowered module names.
static constexpr StringRef kClassNameSuffix = "_Class";

/// Helper class to capture details about a property.
struct Property {
  size_t index;
  StringRef name;
  Type type;
  Location loc;
};

/// Helper class to capture state about a Class being lowered.
struct ClassLoweringState {
  FModuleLike moduleLike;
  std::vector<hw::HierPathOp> paths;
};

struct LoweringState {
  PathInfoTable pathInfoTable;
  DenseMap<om::ClassLike, ClassLoweringState> classLoweringStateTable;
};

/// Helper struct to capture state about an object that needs RtlPorts added.
struct RtlPortsInfo {
  firrtl::PathOp containingModuleRef;
  Value basePath;
  om::ObjectOp object;
};

struct LowerClassesPass
    : public circt::firrtl::impl::LowerClassesBase<LowerClassesPass> {
  void runOnOperation() override;

private:
  LogicalResult processPaths(InstanceGraph &instanceGraph,
                             hw::InnerSymbolNamespaceCollection &namespaces,
                             HierPathCache &cache, PathInfoTable &pathInfoTable,
                             SymbolTable &symbolTable);

  // Predicate to check if a module-like needs a Class to be created.
  bool shouldCreateClass(StringAttr modName);

  // Create an OM Class op from a FIRRTL Class op.
  om::ClassLike createClass(FModuleLike moduleLike,
                            const PathInfoTable &pathInfoTable,
                            std::mutex &intraPassMutex);

  // Lower the FIRRTL Class to OM Class.
  void lowerClassLike(FModuleLike moduleLike, om::ClassLike classLike,
                      const PathInfoTable &pathInfoTable);
  void lowerClass(om::ClassOp classOp, FModuleLike moduleLike,
                  const PathInfoTable &pathInfoTable);
  void lowerClassExtern(ClassExternOp classExternOp, FModuleLike moduleLike);

  // Update Object instantiations in a FIRRTL Module or OM Class.
  LogicalResult updateInstances(Operation *op, InstanceGraph &instanceGraph,
                                const LoweringState &state,
                                const PathInfoTable &pathInfoTable,
                                std::mutex &intraPassMutex);

  /// Create and add all 'ports' lists of RtlPort objects for each object.
  void createAllRtlPorts(const PathInfoTable &pathInfoTable,
                         hw::InnerSymbolNamespaceCollection &namespaces,
                         HierPathCache &hierPathCache);

  // Convert to OM ops and types in Classes or Modules.
  LogicalResult dialectConversion(
      Operation *op, const PathInfoTable &pathInfoTable,
      const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable);

  // State to memoize repeated calls to shouldCreateClass.
  DenseMap<StringAttr, bool> shouldCreateClassMemo;

  // State used while creating the optional 'ports' list of RtlPort objects.
  SmallVector<RtlPortsInfo> rtlPortsToCreate;

  // Store of already created external classes.  External modules are not
  // modules, but specific instantiations of a module with a given
  // parameterization.  When this is happening, two external modules will have
  // the same `defname`.  This is a mechanism to ensure we don't create the same
  // external class twice.
  DenseMap<StringAttr, om::ClassLike> externalClassMap;
};

struct PathTracker {
  // An entry point for parallely tracking paths in the circuit and apply
  // changes to `pathInfoTable`.
  static LogicalResult
  run(CircuitOp circuit, InstanceGraph &instanceGraph,
      hw::InnerSymbolNamespaceCollection &namespaces, HierPathCache &cache,
      PathInfoTable &pathInfoTable, const SymbolTable &symbolTable,
      const DenseMap<DistinctAttr, FModuleOp> &owningModules);

  PathTracker(FModuleLike module,
              hw::InnerSymbolNamespaceCollection &namespaces,
              InstanceGraph &instanceGraph, const SymbolTable &symbolTable,
              const DenseMap<DistinctAttr, FModuleOp> &owningModules)
      : module(module), moduleNamespace(namespaces[module]),
        namespaces(namespaces), instanceGraph(instanceGraph),
        symbolTable(symbolTable), owningModules(owningModules) {}

private:
  struct PathInfoTableEntry {
    Operation *op;
    DistinctAttr id;
    StringAttr altBasePathModule;
    // This is null if the path has no owning module.
    ArrayAttr pathAttr;
  };

  // Run the main logic.
  LogicalResult runOnModule();

  // Return updated annotations for a given AnnoTarget if success.
  FailureOr<AnnotationSet> processPathTrackers(const AnnoTarget &target);

  LogicalResult updatePathInfoTable(PathInfoTable &pathInfoTable,
                                    HierPathCache &cache) const;

  // Determine it is necessary to use an alternative base path for `moduleName`
  // and `owningModule`.
  FailureOr<bool> getOrComputeNeedsAltBasePath(Location loc,
                                               StringAttr moduleName,
                                               FModuleOp owningModule,
                                               bool isNonLocal);
  FModuleLike module;

  // Local data structures.
  hw::InnerSymbolNamespace &moduleNamespace;
  hw::InnerSymbolNamespaceCollection &namespaces;
  DenseMap<std::pair<StringAttr, FModuleOp>, bool> needsAltBasePathCache;

  // Thread-unsafe global data structure. Don't mutate.
  InstanceGraph &instanceGraph;
  const SymbolTable &symbolTable;
  const DenseMap<DistinctAttr, FModuleOp> &owningModules;

  // Result.
  SmallVector<PathInfoTableEntry> entries;
  SetVector<StringAttr> altBasePathRoots;
};

/// Constants and helpers for creating the RtlPorts on the fly.

static constexpr StringRef kContainingModuleName = "containingModule";
static constexpr StringRef kPortsName = "ports";
static constexpr StringRef kRtlPortClassName = "RtlPort";

static Type getRtlPortsType(MLIRContext *context) {
  return om::ListType::get(om::ClassType::get(
      context, FlatSymbolRefAttr::get(context, kRtlPortClassName)));
}

/// Create and add the 'ports' list of RtlPort objects for an object.
static void createRtlPorts(const RtlPortsInfo &rtlPortToCreate,
                           const PathInfoTable &pathInfoTable,
                           hw::InnerSymbolNamespaceCollection &namespaces,
                           HierPathCache &hierPathCache, OpBuilder &builder) {
  firrtl::PathOp containingModuleRef = rtlPortToCreate.containingModuleRef;
  Value basePath = rtlPortToCreate.basePath;
  om::ObjectOp object = rtlPortToCreate.object;

  // Set the builder to just before the object.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(object);

  // Look up the module from the containingModuleRef.

  FlatSymbolRefAttr containingModulePathRef =
      pathInfoTable.table.at(containingModuleRef.getTarget()).symRef;

  const SymbolTable &symbolTable = hierPathCache.getSymbolTable();

  hw::HierPathOp containingModulePath =
      symbolTable.lookup<hw::HierPathOp>(containingModulePathRef.getAttr());

  assert(containingModulePath.isModule() &&
         "expected containing module path to target a module");

  StringAttr moduleName = containingModulePath.leafMod();

  FModuleLike mod = symbolTable.lookup<FModuleLike>(moduleName);
  MLIRContext *ctx = mod.getContext();
  Location loc = mod.getLoc();

  // Create the per-port information.

  auto portClassName = StringAttr::get(ctx, kRtlPortClassName);
  auto portClassType =
      om::ClassType::get(ctx, FlatSymbolRefAttr::get(portClassName));

  SmallVector<Value> ports;
  for (unsigned i = 0, e = mod.getNumPorts(); i < e; ++i) {
    // Only process ports that are not zero-width.
    auto portType = type_dyn_cast<FIRRTLBaseType>(mod.getPortType(i));
    if (!portType || portType.getBitWidthOrSentinel() == 0)
      continue;

    // Get a path to the port. This may modify port attributes or the global
    // namespace of hierpaths, so use the mutex around those operations.

    auto portTarget = PortAnnoTarget(mod, i);

    auto portSym =
        getInnerRefTo({portTarget.getPortNo(), portTarget.getOp(), 0},
                      [&](FModuleLike m) -> hw::InnerSymbolNamespace & {
                        return namespaces[m];
                      });

    FlatSymbolRefAttr portPathRef =
        hierPathCache.getRefFor(ArrayAttr::get(ctx, {portSym}));

    auto portPath = om::PathCreateOp::create(
        builder, loc, om::PathType::get(ctx),
        om::TargetKindAttr::get(ctx, om::TargetKind::DontTouch), basePath,
        portPathRef);

    // Get a direction attribute.

    StringRef portDirectionName =
        mod.getPortDirection(i) == Direction::Out ? "Output" : "Input";

    auto portDirection = om::ConstantOp::create(
        builder, loc, om::StringType::get(ctx),
        StringAttr::get(portDirectionName, om::StringType::get(ctx)));

    // Get a width attribute.

    auto portWidth = om::ConstantOp::create(
        builder, loc, om::OMIntegerType::get(ctx),
        om::IntegerAttr::get(
            ctx, mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                        portType.getBitWidthOrSentinel())));

    // Create an RtlPort object for this port, and add it to the list.

    auto portObj = om::ObjectOp::create(
        builder, loc, portClassType, portClassName,
        ArrayRef<Value>{portPath, portDirection, portWidth});

    ports.push_back(portObj);
  }

  // Create a list of RtlPort objects to be included with the containingModule.

  auto portsList =
      om::ListCreateOp::create(builder, UnknownLoc::get(builder.getContext()),
                               getRtlPortsType(builder.getContext()), ports);

  object.getActualParamsMutable().append({portsList});
}

} // namespace

LogicalResult
PathTracker::run(CircuitOp circuit, InstanceGraph &instanceGraph,
                 hw::InnerSymbolNamespaceCollection &namespaces,
                 HierPathCache &cache, PathInfoTable &pathInfoTable,
                 const SymbolTable &symbolTable,
                 const DenseMap<DistinctAttr, FModuleOp> &owningModules) {
  // First allocate module namespaces. Don't capture a namespace reference at
  // this point since they could be invalidated when DenseMap grows.
  for (auto *node : instanceGraph)
    if (auto module = node->getModule<FModuleLike>())
      (void)namespaces.get(module);

  for (auto *node : instanceGraph)
    if (auto module = node->getModule<FModuleLike>()) {
      // Classes do not have path trackers on them.
      if (isa<firrtl::ClassOp, firrtl::ExtClassOp>(module))
        continue;
      PathTracker tracker(module, namespaces, instanceGraph, symbolTable,
                          owningModules);
      if (failed(tracker.runOnModule()))
        return failure();
      if (failed(tracker.updatePathInfoTable(pathInfoTable, cache)))
        return failure();
    }

  return success();
}

LogicalResult PathTracker::runOnModule() {
  auto processAndUpdateAnnoTarget = [&](AnnoTarget target) -> LogicalResult {
    auto anno = processPathTrackers(target);
    if (failed(anno))
      return failure();
    target.setAnnotations(*anno);
    return success();
  };

  // Process the module annotations.
  if (failed(processAndUpdateAnnoTarget(OpAnnoTarget(module))))
    return failure();

  // Process module port annotations.
  SmallVector<Attribute> portAnnotations;
  portAnnotations.reserve(module.getNumPorts());
  for (unsigned i = 0, e = module.getNumPorts(); i < e; ++i) {
    auto annos = processPathTrackers(PortAnnoTarget(module, i));
    if (failed(annos))
      return failure();
    portAnnotations.push_back(annos->getArrayAttr());
  }
  // Batch update port annotations.
  module.setPortAnnotationsAttr(
      ArrayAttr::get(module.getContext(), portAnnotations));

  // Process ops in the module body.
  auto result = module.walk([&](hw::InnerSymbolOpInterface op) {
    if (failed(processAndUpdateAnnoTarget(OpAnnoTarget(op))))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  // Process paththrough.
  return success();
}

FailureOr<bool>
PathTracker::getOrComputeNeedsAltBasePath(Location loc, StringAttr moduleName,
                                          FModuleOp owningModule,
                                          bool isNonLocal) {

  auto it = needsAltBasePathCache.find({moduleName, owningModule});
  if (it != needsAltBasePathCache.end())
    return it->second;
  bool needsAltBasePath = false;
  auto *node = instanceGraph.lookup(moduleName);
  while (true) {
    // If the path is rooted at the owning module, we're done.
    if (node->getModule() == owningModule)
      break;
    // If there are no more parents, then the path op lives in a different
    // hierarchy than the HW object it references, which needs to handled
    // specially. Flag this, so we know to create an alternative base path
    // below.
    if (node->noUses()) {
      needsAltBasePath = true;
      break;
    }
    // If there is more than one instance of this module, and the target is
    // non-local, then the path operation is ambiguous, which is an error.
    if (isNonLocal && !node->hasOneUse()) {
      auto diag = mlir::emitError(loc)
                  << "unable to uniquely resolve target due "
                     "to multiple instantiation";
      for (auto *use : node->uses())
        diag.attachNote(use->getInstance().getLoc()) << "instance here";
      return diag;
    }
    node = (*node->usesBegin())->getParent();
  }
  needsAltBasePathCache[{moduleName, owningModule}] = needsAltBasePath;
  return needsAltBasePath;
}

FailureOr<AnnotationSet>
PathTracker::processPathTrackers(const AnnoTarget &target) {
  auto error = false;
  auto annotations = target.getAnnotations();
  auto *op = target.getOp();
  annotations.removeAnnotations([&](Annotation anno) {
    // If there has been an error, just skip this annotation.
    if (error)
      return false;

    // We are looking for OMIR tracker annotations.
    if (!anno.isClass("circt.tracker"))
      return false;

    // The token must have a valid ID.
    auto id = anno.getMember<DistinctAttr>("id");
    if (!id) {
      op->emitError("circt.tracker annotation missing id field");
      error = true;
      return false;
    }

    // Get the fieldID.  If there is none, it is assumed to be 0.
    uint64_t fieldID = anno.getFieldID();

    // Attach an inner sym to the operation.
    Attribute targetSym;
    if (auto portTarget = dyn_cast<PortAnnoTarget>(target)) {
      targetSym =
          getInnerRefTo({portTarget.getPortNo(), portTarget.getOp(), fieldID},
                        [&](FModuleLike module) -> hw::InnerSymbolNamespace & {
                          return moduleNamespace;
                        });
    } else if (auto module = dyn_cast<FModuleLike>(op)) {
      assert(!fieldID && "field not valid for modules");
      targetSym = FlatSymbolRefAttr::get(module.getModuleNameAttr());
    } else {
      targetSym =
          getInnerRefTo({target.getOp(), fieldID},
                        [&](FModuleLike module) -> hw::InnerSymbolNamespace & {
                          return moduleNamespace;
                        });
    }

    // Create the hierarchical path.
    SmallVector<Attribute> path;

    // Copy the trailing final target part of the path.
    path.push_back(targetSym);

    auto moduleName = target.getModule().getModuleNameAttr();

    // Verify a nonlocal annotation refers to a HierPathOp.
    hw::HierPathOp hierPathOp;
    if (auto hierName = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
      hierPathOp =
          dyn_cast<hw::HierPathOp>(symbolTable.lookup(hierName.getAttr()));
      if (!hierPathOp) {
        op->emitError("annotation does not point at a HierPathOp");
        error = true;
        return false;
      }
    }

    // Get the owning module. If there is no owning module, then this
    // declaration does not have a use, and we can return early.
    auto owningModule = owningModules.lookup(id);
    if (!owningModule)
      return true;

    // Copy the middle part from the annotation's NLA.
    if (hierPathOp) {
      // Get the original path.
      auto oldPath = hierPathOp.getNamepath().getValue();

      // Set the moduleName and path based on the hierarchical path. If the
      // owningModule is in the hierarichal path, start the hierarchical path
      // there. Otherwise use the top of the hierarchical path.
      bool pathContainsOwningModule = false;
      size_t owningModuleIndex = 0;
      for (auto [idx, pathFramgent] : llvm::enumerate(oldPath)) {
        if (auto innerRef = dyn_cast<hw::InnerRefAttr>(pathFramgent)) {
          if (innerRef.getModule() == owningModule.getModuleNameAttr()) {
            pathContainsOwningModule = true;
            owningModuleIndex = idx;
          }
        } else if (auto symRef = dyn_cast<FlatSymbolRefAttr>(pathFramgent)) {
          if (symRef.getAttr() == owningModule.getModuleNameAttr()) {
            pathContainsOwningModule = true;
            owningModuleIndex = idx;
          }
        }
      }

      if (pathContainsOwningModule) {
        // Set the path root module name to the owning module.
        moduleName = owningModule.getModuleNameAttr();

        // Copy the old path, dropping the module name and the prefix to the
        // owning module.
        llvm::append_range(path, llvm::reverse(oldPath.drop_back().drop_front(
                                     owningModuleIndex)));
      } else {
        // Set the path root module name to the start of the path.
        moduleName = cast<hw::InnerRefAttr>(oldPath.front()).getModule();

        // Copy the old path, dropping the module name.
        llvm::append_range(path, llvm::reverse(oldPath.drop_back()));
      }
    }

    // Check if we need an alternative base path.
    auto needsAltBasePath = getOrComputeNeedsAltBasePath(
        op->getLoc(), moduleName, owningModule, hierPathOp);
    if (failed(needsAltBasePath)) {
      error = true;
      return false;
    }

    // Copy the leading part of the hierarchical path from the owning module
    // to the start of the annotation's NLA.
    InstanceGraphNode *node = instanceGraph.lookup(moduleName);
    while (true) {
      // If it's not a non-local target, we don't have to append anything,
      // unless it needs an alternative base path, in which case we do need to
      // make a hierarchical path.
      if (!hierPathOp && !needsAltBasePath.value())
        break;

      // If we get to the owning module or the top, we're done.
      if (node->getModule() == owningModule || node->noUses())
        break;

      // Append the next level of hierarchy to the path.
      assert(node->hasOneUse() && "expected single instantiation");
      InstanceRecord *inst = *node->usesBegin();
      path.push_back(
          OpAnnoTarget(inst->getInstance<InstanceOp>())
              .getNLAReference(namespaces[inst->getParent()->getModule()]));

      node = inst->getParent();
    }

    // Create the HierPathOp.
    std::reverse(path.begin(), path.end());
    auto pathAttr = ArrayAttr::get(op->getContext(), path);

    // If we need an alternative base path, save the top module from the
    // path. We will plumb in the basepath from this module.
    StringAttr altBasePathModule;
    if (*needsAltBasePath) {
      altBasePathModule =
          TypeSwitch<Attribute, StringAttr>(path.front())
              .Case<FlatSymbolRefAttr>([](auto a) { return a.getAttr(); })
              .Case<hw::InnerRefAttr>([](auto a) { return a.getModule(); });

      altBasePathRoots.insert(altBasePathModule);
    }

    // Record the path operation associated with the path op.
    entries.push_back({op, id, altBasePathModule, pathAttr});

    // Remove this annotation from the operation.
    return true;
  });

  if (error)
    return {};

  return annotations;
}

LogicalResult PathTracker::updatePathInfoTable(PathInfoTable &pathInfoTable,
                                               HierPathCache &cache) const {
  for (auto root : altBasePathRoots)
    pathInfoTable.addAltBasePathRoot(root);

  for (const auto &entry : entries) {
    assert(entry.pathAttr &&
           "expected all PathInfoTableEntries to have a pathAttr");

    // Record the path operation associated with the path op.
    auto [it, inserted] = pathInfoTable.table.try_emplace(entry.id);
    auto &pathInfo = it->second;
    if (!inserted) {
      // If this DistinctAttr has been seen before, check if it actually points
      // to the same thing. This can happen in Dedup, where multiple trackers
      // end up getting created for different paths, which ultimately coalesce
      // into the same path after Dedup completes. Unfortunately, we can't
      // easily detect this in the middle of Dedup, so allow duplicate
      // DistinctAttrs here.
      assert(pathInfo.symRef && "expected all PathInfos to have a symRef");
      auto existingHierpath =
          symbolTable.lookup<hw::HierPathOp>(pathInfo.symRef.getValue());
      auto existingPathAttr = existingHierpath.getNamepath();
      if (existingPathAttr == entry.pathAttr)
        continue;

      assert(pathInfo.loc.has_value() && "all PathInfo should have a Location");
      auto diag = emitError(pathInfo.loc.value(),
                            "path identifier already found, paths must resolve "
                            "to a unique target");
      diag.attachNote(entry.op->getLoc()) << "other path identifier here";
      return failure();
    }

    // Check if the op is targetable by an instance target. The op pointer may
    // be invalidated later, so this is the last time we want to access it here.
    bool canBeInstanceTarget = isa<InstanceOp, FModuleLike>(entry.op);

    pathInfo = {entry.op->getLoc(), canBeInstanceTarget,
                cache.getRefFor(entry.pathAttr), entry.altBasePathModule};
  }
  return success();
}

/// This pass removes the OMIR tracker annotations from operations, and ensures
/// that each thing that was targeted has a hierarchical path targeting it. It
/// builds a table which maps the original OMIR tracker annotation IDs to the
/// corresponding hierarchical paths. We use this table to convert FIRRTL path
/// ops to OM. FIRRTL paths refer to their target using a target ID, while OM
/// paths refer to their target using hierarchical paths.
LogicalResult LowerClassesPass::processPaths(
    InstanceGraph &instanceGraph,
    hw::InnerSymbolNamespaceCollection &namespaces, HierPathCache &cache,
    PathInfoTable &pathInfoTable, SymbolTable &symbolTable) {
  auto circuit = getOperation();

  // Collect the path declarations and owning modules.
  OwningModuleCache owningModuleCache(instanceGraph);
  DenseMap<DistinctAttr, FModuleOp> owningModules;
  std::vector<Operation *> declarations;
  auto result = circuit.walk([&](Operation *op) {
    if (auto pathOp = dyn_cast<PathOp>(op)) {
      // Find the owning module of this path reference.
      auto owningModule = owningModuleCache.lookup(pathOp);
      // If this reference does not have a single owning module, it is an error.
      if (!owningModule) {
        pathOp->emitError("path does not have a single owning module");
        return WalkResult::interrupt();
      }
      auto target = pathOp.getTargetAttr();
      auto [it, inserted] = owningModules.try_emplace(target, owningModule);
      // If this declaration already has a reference, both references must have
      // the same owning module.
      if (!inserted && it->second != owningModule) {
        pathOp->emitError()
            << "path reference " << target << " has conflicting owning modules "
            << it->second.getModuleNameAttr() << " and "
            << owningModule.getModuleNameAttr();
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  if (failed(PathTracker::run(circuit, instanceGraph, namespaces, cache,
                              pathInfoTable, symbolTable, owningModules)))
    return failure();

  // For each module that will be passing through a base path, compute its
  // descendants that need this base path passed through.
  for (auto rootModule : pathInfoTable.getAltBasePathRoots()) {
    InstanceGraphNode *node = instanceGraph.lookup(rootModule);

    // Do a depth first traversal of the instance graph from rootModule,
    // looking for descendants that need to be passed through.
    auto start = llvm::df_begin(node);
    auto end = llvm::df_end(node);
    auto it = start;
    while (it != end) {
      // Nothing to do for the root module.
      if (it == start) {
        ++it;
        continue;
      }

      // If we aren't creating a class for this child, skip this hierarchy.
      if (!shouldCreateClass(
              it->getModule<FModuleLike>().getModuleNameAttr())) {
        it = it.skipChildren();
        continue;
      }

      // If we are at a leaf, nothing to do.
      if (it->begin() == it->end()) {
        ++it;
        continue;
      }

      // Track state for this passthrough.
      StringAttr passthroughModule = it->getModule().getModuleNameAttr();
      pathInfoTable.addAltBasePathPassthrough(passthroughModule, rootModule);

      ++it;
    }
  }

  return success();
}

/// Lower FIRRTL Class and Object ops to OM Class and Object ops
void LowerClassesPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto intraPassMutex = std::mutex();

  // Get the CircuitOp.
  CircuitOp circuit = getOperation();

  // Get the InstanceGraph and SymbolTable.
  InstanceGraph &instanceGraph = getAnalysis<InstanceGraph>();
  SymbolTable &symbolTable = getAnalysis<SymbolTable>();

  hw::InnerSymbolNamespaceCollection namespaces;
  HierPathCache cache(circuit, symbolTable);

  // Fill `shouldCreateClassMemo`.
  for (auto *node : instanceGraph)
    if (auto moduleLike = node->getModule<firrtl::FModuleLike>())
      shouldCreateClassMemo.insert({moduleLike.getModuleNameAttr(), false});

  parallelForEach(circuit.getContext(), instanceGraph,
                  [&](igraph::InstanceGraphNode *node) {
                    if (auto moduleLike = node->getModule<FModuleLike>())
                      shouldCreateClassMemo[moduleLike.getModuleNameAttr()] =
                          shouldCreateClassImpl(node);
                  });

  // Rewrite all path annotations into inner symbol targets.
  PathInfoTable pathInfoTable;
  if (failed(processPaths(instanceGraph, namespaces, cache, pathInfoTable,
                          symbolTable))) {
    signalPassFailure();
    return;
  }

  LoweringState loweringState;

  // Create new OM Class ops serially while tracking modules that need port
  // erasure.
  DenseMap<StringAttr, firrtl::ClassType> classTypeTable;
  SmallVector<FModuleLike> modulesToErasePortsFrom;
  for (auto *node : instanceGraph) {
    auto moduleLike = node->getModule<firrtl::FModuleLike>();
    if (!moduleLike)
      continue;

    if (shouldCreateClass(moduleLike.getModuleNameAttr())) {
      auto omClass = createClass(moduleLike, pathInfoTable, intraPassMutex);
      auto &classLoweringState = loweringState.classLoweringStateTable[omClass];
      // For external modules with the same defname, the same omClass is reused.
      // Only set moduleLike if not already set, to avoid overwriting.
      if (!classLoweringState.moduleLike)
        classLoweringState.moduleLike = moduleLike;

      // Track this module for port erasure if it's not a ClassLike.
      if (!isa<firrtl::ClassLike>(moduleLike.getOperation()))
        modulesToErasePortsFrom.push_back(moduleLike);

      // Find the module instances under the current module with metadata. These
      // ops will be converted to om objects by this pass. Create a hierarchical
      // path for each of these instances, which will be used to rebase path
      // operations. Hierarchical paths must be created serially to ensure their
      // order in the circuit is deterministc.
      for (auto *instance : *node) {
        auto inst = instance->getInstance<firrtl::InstanceOp>();
        if (!inst)
          continue;
        // Get the referenced module.
        auto module = instance->getTarget()->getModule<FModuleLike>();
        if (module && shouldCreateClass(module.getModuleNameAttr())) {
          auto targetSym = getInnerRefTo(
              {inst, 0}, [&](FModuleLike module) -> hw::InnerSymbolNamespace & {
                return namespaces[module];
              });
          SmallVector<Attribute> path = {targetSym};
          auto pathAttr = ArrayAttr::get(ctx, path);
          auto hierPath = cache.getOpFor(pathAttr);
          classLoweringState.paths.push_back(hierPath);
        }
      }

      if (auto classLike =
              dyn_cast<firrtl::ClassLike>(moduleLike.getOperation()))
        classTypeTable[classLike.getNameAttr()] = classLike.getInstanceType();
    }
  }

  // Move ops from FIRRTL Class to OM Class in parallel.
  mlir::parallelForEach(ctx, loweringState.classLoweringStateTable,
                        [this, &pathInfoTable](auto &entry) {
                          const auto &[classLike, state] = entry;
                          lowerClassLike(state.moduleLike, classLike,
                                         pathInfoTable);
                        });

  // Erase property ports from all modules that had classes created.  This must
  // be done separately because multiple modules can share the same class (e.g.,
  // external modules with the same defname).
  for (auto moduleLike : modulesToErasePortsFrom) {
    BitVector portsToErase(moduleLike.getNumPorts());
    for (unsigned i = 0, e = moduleLike.getNumPorts(); i < e; ++i)
      if (isa<PropertyType>(moduleLike.getPortType(i)))
        portsToErase.set(i);
    moduleLike.erasePorts(portsToErase);
  }

  // Completely erase Class module-likes, and remove from the InstanceGraph.
  for (auto &[omClass, state] : loweringState.classLoweringStateTable) {
    if (isa<firrtl::ClassLike>(state.moduleLike.getOperation())) {
      InstanceGraphNode *node = instanceGraph.lookup(state.moduleLike);
      for (auto *use : llvm::make_early_inc_range(node->uses()))
        use->erase();
      instanceGraph.erase(node);
      state.moduleLike.erase();
    }
  }

  // Collect ops where Objects can be instantiated.
  SmallVector<Operation *> objectContainers;
  for (auto &op : circuit.getOps())
    if (isa<FModuleOp, om::ClassLike>(op))
      objectContainers.push_back(&op);

  // Update Object creation ops in Classes or Modules in parallel.
  if (failed(
          mlir::failableParallelForEach(ctx, objectContainers, [&](auto *op) {
            return updateInstances(op, instanceGraph, loweringState,
                                   pathInfoTable, intraPassMutex);
          })))
    return signalPassFailure();

  // If needed, create and add 'ports' lists of RtlPort objects.
  if (!rtlPortsToCreate.empty())
    createAllRtlPorts(pathInfoTable, namespaces, cache);

  // Convert to OM ops and types in Classes or Modules in parallel.
  if (failed(
          mlir::failableParallelForEach(ctx, objectContainers, [&](auto *op) {
            return dialectConversion(op, pathInfoTable, classTypeTable);
          })))
    return signalPassFailure();

  // We keep the instance graph up to date, so mark that analysis preserved.
  markAnalysesPreserved<InstanceGraph>();

  // Reset pass state.
  rtlPortsToCreate.clear();
}

// Predicate to check if a module-like needs a Class to be created.
bool LowerClassesPass::shouldCreateClass(StringAttr modName) {
  // Return a memoized result.
  return shouldCreateClassMemo.at(modName);
}

void checkAddContainingModulePorts(bool hasContainingModule, OpBuilder builder,
                                   SmallVector<Attribute> &fieldNames,
                                   SmallVector<NamedAttribute> &fieldTypes) {
  if (hasContainingModule) {
    auto name = builder.getStringAttr(kPortsName);
    fieldNames.push_back(name);
    fieldTypes.push_back(NamedAttribute(
        name, TypeAttr::get(getRtlPortsType(builder.getContext()))));
  }
}

static om::ClassLike convertExtClass(FModuleLike moduleLike, OpBuilder builder,
                                     Twine name,
                                     ArrayRef<StringRef> formalParamNames,
                                     bool hasContainingModule) {
  SmallVector<Attribute> fieldNames;
  SmallVector<NamedAttribute> fieldTypes;
  for (unsigned i = 0, e = moduleLike.getNumPorts(); i < e; ++i) {
    auto type = moduleLike.getPortType(i);
    if (!isa<PropertyType>(type))
      continue;

    auto direction = moduleLike.getPortDirection(i);
    if (direction != Direction::In) {
      auto name = moduleLike.getPortNameAttr(i);
      fieldNames.push_back(name);
      fieldTypes.push_back(NamedAttribute(name, TypeAttr::get(type)));
    }
  }
  checkAddContainingModulePorts(hasContainingModule, builder, fieldNames,
                                fieldTypes);
  return om::ClassExternOp::create(builder, moduleLike.getLoc(), name,
                                   formalParamNames, fieldNames, fieldTypes);
}

static om::ClassLike convertClass(FModuleLike moduleLike, OpBuilder builder,
                                  Twine name,
                                  ArrayRef<StringRef> formalParamNames,
                                  bool hasContainingModule) {
  // Collect output property assignments to get field names and types.
  SmallVector<Attribute> fieldNames;
  SmallVector<NamedAttribute> fieldTypes;
  for (auto op : llvm::make_early_inc_range(
           moduleLike->getRegion(0).getOps<PropAssignOp>())) {
    auto outputPort = dyn_cast<BlockArgument>(op.getDest());
    if (!outputPort)
      continue;

    StringAttr name = moduleLike.getPortNameAttr(outputPort.getArgNumber());

    fieldNames.push_back(name);
    fieldTypes.push_back(
        NamedAttribute(name, TypeAttr::get(op.getSrc().getType())));
  }

  checkAddContainingModulePorts(hasContainingModule, builder, fieldNames,
                                fieldTypes);
  return om::ClassOp::create(builder, moduleLike.getLoc(), name,
                             formalParamNames, fieldNames, fieldTypes);
}

// Create an OM Class op from a FIRRTL Class op or Module op with properties.
om::ClassLike LowerClassesPass::createClass(FModuleLike moduleLike,
                                            const PathInfoTable &pathInfoTable,
                                            std::mutex &intraPassMutex) {
  // Collect the parameter names from input properties.
  SmallVector<StringRef> formalParamNames;
  // Every class gets a base path as its first parameter.
  formalParamNames.emplace_back("basepath");

  // If this class is passing through base paths from above, add those.
  size_t nAltBasePaths =
      pathInfoTable.getNumAltBasePaths(moduleLike.getModuleNameAttr());
  for (size_t i = 0; i < nAltBasePaths; ++i)
    formalParamNames.push_back(StringAttr::get(
        moduleLike->getContext(), "alt_basepath_" + llvm::Twine(i)));

  // Collect the input parameters.
  bool hasContainingModule = false;
  for (auto [index, port] : llvm::enumerate(moduleLike.getPorts())) {
    if (port.isInput() && isa<PropertyType>(port.type)) {
      formalParamNames.push_back(port.name);

      // Check if we have a 'containingModule' field.
      if (port.name.strref().contains(kContainingModuleName))
        hasContainingModule = true;
    }
  }

  OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBodyBlock());

  // If there is a 'containingModule', add a parameter for 'ports'.
  if (hasContainingModule)
    formalParamNames.push_back(kPortsName);

  // Take the name from the FIRRTL Class or Module to create the OM Class name.
  StringRef className = moduleLike.getName();
  StringAttr baseClassNameAttr = moduleLike.getNameAttr();

  // Use the defname for external modules.
  if (auto externMod = dyn_cast<FExtModuleOp>(moduleLike.getOperation())) {
    className = externMod.getExtModuleName();
    baseClassNameAttr = externMod.getExtModuleNameAttr();
  }

  // If the op is a Module or ExtModule, the OM Class would conflict with the HW
  // Module, so give it a suffix. There is no formal ABI for this yet.
  StringRef suffix =
      isa<FModuleOp, FExtModuleOp>(moduleLike) ? kClassNameSuffix : "";

  // Construct the OM Class with the FIRRTL Class name and parameter names.
  om::ClassLike loweredClassOp;
  if (isa<firrtl::ExtClassOp, firrtl::FExtModuleOp>(
          moduleLike.getOperation())) {
    // External modules are "deduplicated" via their defname.  Don't create a
    // new external class if we've already created one for this defname.
    auto [it, inserted] = externalClassMap.insert({baseClassNameAttr, {}});
    if (inserted)
      it->getSecond() = convertExtClass(moduleLike, builder, className + suffix,
                                        formalParamNames, hasContainingModule);
    loweredClassOp = it->getSecond();
  } else {
    loweredClassOp = convertClass(moduleLike, builder, className + suffix,
                                  formalParamNames, hasContainingModule);
  }

  return loweredClassOp;
}

void LowerClassesPass::lowerClassLike(FModuleLike moduleLike,
                                      om::ClassLike classLike,
                                      const PathInfoTable &pathInfoTable) {

  if (auto classOp = dyn_cast<om::ClassOp>(classLike.getOperation())) {
    return lowerClass(classOp, moduleLike, pathInfoTable);
  }
  if (auto classExternOp =
          dyn_cast<om::ClassExternOp>(classLike.getOperation())) {
    return lowerClassExtern(classExternOp, moduleLike);
  }
  llvm_unreachable("unhandled class-like op");
}

void LowerClassesPass::lowerClass(om::ClassOp classOp, FModuleLike moduleLike,
                                  const PathInfoTable &pathInfoTable) {
  // Collect information about property ports.
  SmallVector<Property> inputProperties;
  BitVector portsToErase(moduleLike.getNumPorts());
  bool hasContainingModule = false;
  for (auto [index, port] : llvm::enumerate(moduleLike.getPorts())) {
    // For Module ports that aren't property types, move along.
    if (!isa<PropertyType>(port.type))
      continue;

    // Remember input properties to create the OM Class formal parameters.
    if (port.isInput()) {
      inputProperties.push_back({index, port.name, port.type, port.loc});

      // Check if we have a 'containingModule' field.
      if (port.name.strref().contains(kContainingModuleName))
        hasContainingModule = true;
    }

    // In case this is a Module, remember to erase this port.
    portsToErase.set(index);
  }

  // Construct the OM Class body with block arguments for each input property,
  // updating the mapping to map from the input property to the block argument.
  Block *moduleBody = &moduleLike->getRegion(0).front();
  Block *classBody = &classOp->getRegion(0).emplaceBlock();
  // Every class created from a module gets a base path as its first parameter.
  auto basePathType = BasePathType::get(&getContext());
  auto unknownLoc = UnknownLoc::get(&getContext());
  classBody->addArgument(basePathType, unknownLoc);

  // If this class is passing through base paths from above, add those.
  size_t nAltBasePaths =
      pathInfoTable.getNumAltBasePaths(moduleLike.getModuleNameAttr());
  for (size_t i = 0; i < nAltBasePaths; ++i)
    classBody->addArgument(basePathType, unknownLoc);

  // Move operations from the modulelike to the OM class.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*moduleBody))) {
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      if (!shouldCreateClass(instance.getReferencedModuleNameAttr()))
        continue;
      auto *clone = OpBuilder::atBlockBegin(classBody).clone(op);
      for (auto result : instance.getResults()) {
        if (isa<PropertyType>(result.getType()))
          result.replaceAllUsesWith(clone->getResult(result.getResultNumber()));
      }
      continue;
    }

    auto isProperty = [](auto x) { return isa<PropertyType>(x.getType()); };
    if (llvm::any_of(op.getOperands(), isProperty) ||
        llvm::any_of(op.getResults(), isProperty))
      op.moveBefore(classBody, classBody->begin());
  }

  // Move property ports from the module to the class.
  for (auto input : inputProperties) {
    auto arg = classBody->addArgument(input.type, input.loc);
    moduleBody->getArgument(input.index).replaceAllUsesWith(arg);
  }

  llvm::SmallVector<mlir::Location> fieldLocs;
  llvm::SmallVector<mlir::Value> fieldValues;
  for (Operation &op :
       llvm::make_early_inc_range(classOp.getBodyBlock()->getOperations())) {
    if (auto propAssign = dyn_cast<PropAssignOp>(op)) {
      if (auto blockArg = dyn_cast<BlockArgument>(propAssign.getDest())) {
        // Store any output property assignments into fields op inputs.
        fieldLocs.push_back(op.getLoc());
        fieldValues.push_back(propAssign.getSrc());
        propAssign.erase();
      }
    }
  }

  // If there is a 'containingModule', add an argument for 'ports', and a field.
  if (hasContainingModule) {
    BlockArgument argumentValue = classBody->addArgument(
        getRtlPortsType(&getContext()), UnknownLoc::get(&getContext()));
    fieldLocs.push_back(argumentValue.getLoc());
    fieldValues.push_back(argumentValue);
  }

  OpBuilder builder = OpBuilder::atBlockEnd(classOp.getBodyBlock());
  classOp.addNewFieldsOp(builder, fieldLocs, fieldValues);

  // Port erasure for the `moduleLike` is handled centrally in `runOnOperation`.
}

void LowerClassesPass::lowerClassExtern(ClassExternOp classExternOp,
                                        FModuleLike moduleLike) {
  // Construct the OM Class body.
  // Add a block arguments for each input property.
  // Add a class.extern.field op for each output.
  Block *classBody = &classExternOp.getRegion().emplaceBlock();

  // Every class gets a base path as its first parameter.
  classBody->addArgument(BasePathType::get(&getContext()),
                         UnknownLoc::get(&getContext()));

  for (unsigned i = 0, e = moduleLike.getNumPorts(); i < e; ++i) {
    auto type = moduleLike.getPortType(i);
    if (!isa<PropertyType>(type))
      continue;

    auto loc = moduleLike.getPortLocation(i);
    auto direction = moduleLike.getPortDirection(i);
    if (direction == Direction::In)
      classBody->addArgument(type, loc);
  }

  // Port erasure for the `moduleLike` is handled centrally in `runOnOperation`.
}

// Helper to update an Object instantiation. FIRRTL Object instances are
// converted to OM Object instances.
static LogicalResult updateObjectInClass(
    firrtl::ObjectOp firrtlObject, const PathInfoTable &pathInfoTable,
    SmallVectorImpl<RtlPortsInfo> &rtlPortsToCreate, std::mutex &intraPassMutex,
    SmallVectorImpl<Operation *> &opsToErase) {
  // The 0'th argument is the base path.
  auto basePath = firrtlObject->getBlock()->getArgument(0);
  // build a table mapping the indices of input ports to their position in the
  // om class's parameter list.
  auto firrtlClassType = firrtlObject.getType();
  auto numElements = firrtlClassType.getNumElements();
  llvm::SmallVector<unsigned> argIndexTable;
  argIndexTable.resize(numElements);

  // Get any alternative base paths passing through this module.
  SmallVector<Value> altBasePaths;
  pathInfoTable.collectAltBasePaths(
      firrtlObject, firrtlClassType.getNameAttr().getAttr(), altBasePaths);

  // Account for the default base path and any alternatives.
  unsigned nextArgIndex = 1 + altBasePaths.size();

  for (unsigned i = 0; i < numElements; ++i) {
    auto direction = firrtlClassType.getElement(i).direction;
    if (direction == Direction::In)
      argIndexTable[i] = nextArgIndex++;
  }

  // Collect its input actual parameters by finding any subfield ops that are
  // assigned to. Take the source of the assignment as the actual parameter.

  llvm::SmallVector<Value> args;
  args.resize(nextArgIndex);
  args[0] = basePath;

  // Collect any alternative base paths passing through.
  for (auto [i, altBasePath] : llvm::enumerate(altBasePaths))
    args[1 + i] = altBasePath; // + 1 to skip default base path

  firrtl::PathOp containingModuleRef;
  for (auto *user : llvm::make_early_inc_range(firrtlObject->getUsers())) {
    if (auto subfield = dyn_cast<ObjectSubfieldOp>(user)) {
      auto index = subfield.getIndex();
      auto direction = firrtlClassType.getElement(index).direction;

      // We only lower "writes to input ports" here. Reads from output
      // ports will be handled using the conversion framework.
      if (direction == Direction::Out)
        continue;

      for (auto *subfieldUser :
           llvm::make_early_inc_range(subfield->getUsers())) {
        if (auto propassign = dyn_cast<PropAssignOp>(subfieldUser)) {
          // the operands of the propassign may have already been converted to
          // om. Use the generic operand getters to get the operands as
          // untyped values.
          auto dst = propassign.getOperand(0);
          auto src = propassign.getOperand(1);
          if (dst == subfield.getResult()) {
            args[argIndexTable[index]] = src;
            opsToErase.push_back(propassign);

            // Check if we have a 'containingModule' field.
            if (firrtlClassType.getElement(index).name.strref().contains(
                    kContainingModuleName)) {
              assert(!containingModuleRef &&
                     "expected exactly one containingModule");
              assert(isa_and_nonnull<firrtl::PathOp>(src.getDefiningOp()) &&
                     "expected containingModule to be a PathOp");
              containingModuleRef = src.getDefiningOp<firrtl::PathOp>();
            }
          }
        }
      }

      opsToErase.push_back(subfield);
    }
  }

  // Check that all input ports have been initialized.
  for (unsigned i = 0; i < numElements; ++i) {
    auto element = firrtlClassType.getElement(i);
    if (element.direction == Direction::Out)
      continue;

    auto argIndex = argIndexTable[i];
    if (!args[argIndex])
      return emitError(firrtlObject.getLoc())
             << "uninitialized input port " << element.name;
  }

  // Convert the FIRRTL Class type to an OM Class type.
  auto className = firrtlObject.getType().getNameAttr();
  auto classType = om::ClassType::get(firrtlObject->getContext(), className);

  // Create the new Object op.
  OpBuilder builder(firrtlObject);

  auto object = om::ObjectOp::create(builder, firrtlObject.getLoc(), classType,
                                     firrtlObject.getClassNameAttr(), args);

  // If there is a 'containingModule', track that we need to add 'ports'.
  if (containingModuleRef) {
    std::lock_guard<std::mutex> guard(intraPassMutex);
    rtlPortsToCreate.push_back({containingModuleRef, basePath, object});
  }

  // Replace uses of the FIRRTL Object with the OM Object. The later dialect
  // conversion will take care of converting the types.
  auto cast = UnrealizedConversionCastOp::create(
      builder, object.getLoc(), firrtlObject.getType(), object.getResult());
  firrtlObject.replaceAllUsesWith(cast.getResult(0));

  // Erase the original Object, now that we're done with it.
  opsToErase.push_back(firrtlObject);
  return success();
}

// Helper to update a Module instantiation in a Class. Module instances within a
// Class are converted to OM Object instances of the Class derived from the
// Module.
static LogicalResult
updateInstanceInClass(InstanceOp firrtlInstance, hw::HierPathOp hierPath,
                      InstanceGraph &instanceGraph,
                      const PathInfoTable &pathInfoTable,
                      SmallVectorImpl<Operation *> &opsToErase) {

  // Set the insertion point right before the instance op.
  OpBuilder builder(firrtlInstance);

  // Collect the FIRRTL instance inputs to form the Object instance actual
  // parameters. The order of the SmallVector needs to match the order the
  // formal parameters are declared on the corresponding Class.
  SmallVector<Value> actualParameters;
  // The 0'th argument is the base path.
  auto basePath = firrtlInstance->getBlock()->getArgument(0);
  auto symRef = FlatSymbolRefAttr::get(hierPath.getSymNameAttr());
  auto rebasedPath = om::BasePathCreateOp::create(
      builder, firrtlInstance->getLoc(), basePath, symRef);

  actualParameters.push_back(rebasedPath);

  // Add any alternative base paths passing through this instance.
  pathInfoTable.collectAltBasePaths(
      firrtlInstance, firrtlInstance.getModuleNameAttr().getAttr(),
      actualParameters);

  for (auto result : firrtlInstance.getResults()) {
    // If the port is an output, continue.
    if (firrtlInstance.getPortDirection(result.getResultNumber()) ==
        Direction::Out)
      continue;

    // If the port is not a property type, continue.
    auto propertyResult = dyn_cast<FIRRTLPropertyValue>(result);
    if (!propertyResult)
      continue;

    // Get the property assignment to the input, and track the assigned
    // Value as an actual parameter to the Object instance.
    auto propertyAssignment = getPropertyAssignment(propertyResult);
    assert(propertyAssignment && "properties require single assignment");
    actualParameters.push_back(propertyAssignment.getSrcMutable().get());

    // Erase the property assignment.
    opsToErase.push_back(propertyAssignment);
  }

  // Get the referenced module to get its name.
  auto referencedModule =
      firrtlInstance.getReferencedModule<FModuleLike>(instanceGraph);

  StringRef moduleName = referencedModule.getName();

  // Use the defname for external modules.
  if (auto externMod = dyn_cast<FExtModuleOp>(referencedModule.getOperation()))
    moduleName = externMod.getExtModuleName();

  // Convert the FIRRTL Module name to an OM Class type.
  auto className = FlatSymbolRefAttr::get(
      builder.getStringAttr(moduleName + kClassNameSuffix));

  auto classType = om::ClassType::get(firrtlInstance->getContext(), className);

  // Create the new Object op.
  auto object =
      om::ObjectOp::create(builder, firrtlInstance.getLoc(), classType,
                           className.getAttr(), actualParameters);

  // Replace uses of the FIRRTL instance outputs with field access into
  // the OM Object. The later dialect conversion will take care of
  // converting the types.
  for (auto result : firrtlInstance.getResults()) {
    // If the port isn't an output, continue.
    if (firrtlInstance.getPortDirection(result.getResultNumber()) !=
        Direction::Out)
      continue;

    // If the port is not a property type, continue.
    if (!isa<PropertyType>(result.getType()))
      continue;

    // The path to the field is just this output's name.
    auto objectFieldPath = builder.getArrayAttr({FlatSymbolRefAttr::get(
        firrtlInstance.getPortNameAttr(result.getResultNumber()))});

    // Create the field access.
    auto objectField = ObjectFieldOp::create(
        builder, object.getLoc(), result.getType(), object, objectFieldPath);

    result.replaceAllUsesWith(objectField);
  }

  // Erase the original instance, now that we're done with it.
  opsToErase.push_back(firrtlInstance);
  return success();
}

// Helper to update a Module instantiation in a Module. Module instances within
// a Module are updated to remove the property typed ports.
static LogicalResult
updateInstanceInModule(InstanceOp firrtlInstance, InstanceGraph &instanceGraph,
                       SmallVectorImpl<Operation *> &opsToErase) {
  // Collect property typed ports to erase.
  BitVector portsToErase(firrtlInstance.getNumResults());
  for (auto result : firrtlInstance.getResults())
    if (isa<PropertyType>(result.getType()))
      portsToErase.set(result.getResultNumber());

  // If there are none, nothing to do.
  if (portsToErase.none())
    return success();

  // Create a new instance with the property ports removed.
  InstanceOp newInstance =
      firrtlInstance.cloneWithErasedPortsAndReplaceUses(portsToErase);

  // Replace the instance in the instance graph. This is called from multiple
  // threads, but because the instance graph data structure is not mutated, and
  // only one thread ever sets the instance pointer for a given instance, this
  // should be safe.
  instanceGraph.replaceInstance(firrtlInstance, newInstance);

  // Erase the original instance, which is now replaced.
  opsToErase.push_back(firrtlInstance);
  return success();
}

static LogicalResult
updateInstancesInModule(FModuleOp moduleOp, InstanceGraph &instanceGraph,
                        SmallVectorImpl<Operation *> &opsToErase) {
  OpBuilder builder(moduleOp);
  for (auto &op : moduleOp->getRegion(0).getOps()) {
    if (auto objectOp = dyn_cast<firrtl::ObjectOp>(op)) {
      assert(0 && "should be no objects in modules");
    } else if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      if (failed(updateInstanceInModule(instanceOp, instanceGraph, opsToErase)))
        return failure();
    }
  }
  return success();
}

static LogicalResult updateObjectsAndInstancesInClass(
    om::ClassOp classOp, InstanceGraph &instanceGraph,
    const LoweringState &state, const PathInfoTable &pathInfoTable,
    SmallVectorImpl<RtlPortsInfo> &rtlPortsToCreate, std::mutex &intraPassMutex,
    SmallVectorImpl<Operation *> &opsToErase) {
  OpBuilder builder(classOp);
  auto &classState = state.classLoweringStateTable.at(classOp);
  auto it = classState.paths.begin();
  for (auto &op : classOp->getRegion(0).getOps()) {
    if (auto objectOp = dyn_cast<firrtl::ObjectOp>(op)) {
      if (failed(updateObjectInClass(objectOp, pathInfoTable, rtlPortsToCreate,
                                     intraPassMutex, opsToErase)))
        return failure();
    } else if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      if (failed(updateInstanceInClass(instanceOp, *it++, instanceGraph,
                                       pathInfoTable, opsToErase)))
        return failure();
    }
  }
  return success();
}

// Update Object or Module instantiations in a FIRRTL Module or OM Class.
LogicalResult LowerClassesPass::updateInstances(
    Operation *op, InstanceGraph &instanceGraph, const LoweringState &state,
    const PathInfoTable &pathInfoTable, std::mutex &intraPassMutex) {

  // Track ops to erase at the end. We can't do this eagerly, since we want to
  // loop over each op in the container's body, and we may end up removing some
  // ops later in the body when we visit instances earlier in the body.
  SmallVector<Operation *> opsToErase;
  auto result =
      TypeSwitch<Operation *, LogicalResult>(op)

          .Case([&](FModuleOp moduleOp) {
            // Convert FIRRTL Module instance within a Module to
            // remove property ports if necessary.
            return updateInstancesInModule(moduleOp, instanceGraph, opsToErase);
          })
          .Case([&](om::ClassOp classOp) {
            // Convert FIRRTL Module instance within a Class to OM
            // Object instance.
            return updateObjectsAndInstancesInClass(
                classOp, instanceGraph, state, pathInfoTable, rtlPortsToCreate,
                intraPassMutex, opsToErase);
          })
          .Default([](auto *op) { return success(); });
  if (failed(result))
    return result;
  // Erase the ops marked to be erased.
  for (auto *op : opsToErase)
    op->erase();

  return success();
}

// Create and add all 'ports' lists of RtlPort objects for each object.
void LowerClassesPass::createAllRtlPorts(
    const PathInfoTable &pathInfoTable,
    hw::InnerSymbolNamespaceCollection &namespaces,
    HierPathCache &hierPathCache) {
  MLIRContext *ctx = &getContext();

  // Get a builder initialized to the end of the top-level module.
  OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBodyBlock());

  // Declare an RtlPort class on the fly.
  om::ClassOp::buildSimpleClassOp(
      builder, UnknownLoc::get(ctx), kRtlPortClassName,
      {"ref", "direction", "width"}, {"ref", "direction", "width"},
      {om::PathType::get(ctx), om::StringType::get(ctx),
       om::OMIntegerType::get(ctx)});

  // Sort the collected rtlPortsToCreate and process each.
  llvm::stable_sort(rtlPortsToCreate, [](auto lhs, auto rhs) {
    return lhs.object.getClassName() < rhs.object.getClassName();
  });

  // Create each 'ports' list.
  for (auto rtlPortToCreate : rtlPortsToCreate)
    createRtlPorts(rtlPortToCreate, pathInfoTable, namespaces, hierPathCache,
                   builder);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

struct FIntegerConstantOpConversion
    : public OpConversionPattern<FIntegerConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FIntegerConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::ConstantOp>(
        op, om::OMIntegerType::get(op.getContext()),
        om::IntegerAttr::get(op.getContext(), adaptor.getValueAttr()));
    return success();
  }
};

struct BoolConstantOpConversion : public OpConversionPattern<BoolConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BoolConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::ConstantOp>(
        op, rewriter.getBoolAttr(adaptor.getValue()));
    return success();
  }
};

struct DoubleConstantOpConversion
    : public OpConversionPattern<DoubleConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DoubleConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::ConstantOp>(op, adaptor.getValue());
    return success();
  }
};

struct StringConstantOpConversion
    : public OpConversionPattern<StringConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringType = om::StringType::get(op.getContext());
    rewriter.replaceOpWithNewOp<om::ConstantOp>(
        op, stringType, StringAttr::get(op.getValue(), stringType));
    return success();
  }
};

struct ListCreateOpConversion
    : public OpConversionPattern<firrtl::ListCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(firrtl::ListCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto listType = getTypeConverter()->convertType<om::ListType>(op.getType());
    if (!listType)
      return failure();
    rewriter.replaceOpWithNewOp<om::ListCreateOp>(op, listType,
                                                  adaptor.getElements());
    return success();
  }
};

struct ListConcatOpConversion
    : public OpConversionPattern<firrtl::ListConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(firrtl::ListConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto listType = getTypeConverter()->convertType<om::ListType>(op.getType());
    if (!listType)
      return failure();
    rewriter.replaceOpWithNewOp<om::ListConcatOp>(op, listType,
                                                  adaptor.getSubLists());
    return success();
  }
};

struct IntegerAddOpConversion
    : public OpConversionPattern<firrtl::IntegerAddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(firrtl::IntegerAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::IntegerAddOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    return success();
  }
};

struct IntegerMulOpConversion
    : public OpConversionPattern<firrtl::IntegerMulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(firrtl::IntegerMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::IntegerMulOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    return success();
  }
};

struct IntegerShrOpConversion
    : public OpConversionPattern<firrtl::IntegerShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(firrtl::IntegerShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::IntegerShrOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    return success();
  }
};

struct IntegerShlOpConversion
    : public OpConversionPattern<firrtl::IntegerShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(firrtl::IntegerShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::IntegerShlOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    return success();
  }
};

struct PathOpConversion : public OpConversionPattern<firrtl::PathOp> {

  PathOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   const PathInfoTable &pathInfoTable,
                   PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        pathInfoTable(pathInfoTable) {}

  LogicalResult
  matchAndRewrite(firrtl::PathOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    auto pathType = om::PathType::get(context);
    auto pathInfoIt = pathInfoTable.table.find(op.getTarget());

    // The 0'th argument is the base path by default.
    auto basePath = op->getBlock()->getArgument(0);

    // If the target was optimized away, then replace the path operation with
    // a deleted path.
    if (pathInfoIt == pathInfoTable.table.end()) {
      if (op.getTargetKind() == firrtl::TargetKind::DontTouch)
        return emitError(op.getLoc(), "DontTouch target was deleted");
      if (op.getTargetKind() == firrtl::TargetKind::Instance)
        return emitError(op.getLoc(), "Instance target was deleted");
      rewriter.replaceOpWithNewOp<om::EmptyPathOp>(op);
      return success();
    }

    auto pathInfo = pathInfoIt->second;
    auto symbol = pathInfo.symRef;

    // Convert the target kind to an OMIR target.  Member references are updated
    // to reflect the current kind of reference.
    om::TargetKind targetKind;
    switch (op.getTargetKind()) {
    case firrtl::TargetKind::DontTouch:
      targetKind = om::TargetKind::DontTouch;
      break;
    case firrtl::TargetKind::Reference:
      targetKind = om::TargetKind::Reference;
      break;
    case firrtl::TargetKind::Instance:
      if (!pathInfo.canBeInstanceTarget)
        return emitError(op.getLoc(), "invalid target for instance path")
                   .attachNote(pathInfo.loc)
               << "target not instance or module";
      targetKind = om::TargetKind::Instance;
      break;
    case firrtl::TargetKind::MemberInstance:
    case firrtl::TargetKind::MemberReference:
      if (pathInfo.canBeInstanceTarget)
        targetKind = om::TargetKind::MemberInstance;
      else
        targetKind = om::TargetKind::MemberReference;
      break;
    }

    // If we are using an alternative base path for this path, get it from the
    // passthrough port on the enclosing class.
    if (auto altBasePathModule = pathInfo.altBasePathModule) {
      // Get the original name of the parent. At this point both FIRRTL classes
      // and modules have been converted to OM classes, but we need to look up
      // based on the parent's original name.
      auto parent = op->getParentOfType<om::ClassOp>();
      auto parentName = parent.getName();
      if (parentName.ends_with(kClassNameSuffix))
        parentName = parentName.drop_back(kClassNameSuffix.size());
      auto originalParentName = StringAttr::get(op->getContext(), parentName);

      // Get the base paths passing through the parent.
      auto altBasePaths =
          pathInfoTable.getRootsForPassthrough(originalParentName);
      assert(!altBasePaths.empty() && "expected passthrough base paths");

      // Find the base path passthrough that was associated with this path.
      for (auto [i, altBasePath] : llvm::enumerate(altBasePaths)) {
        if (altBasePathModule == altBasePath) {
          // + 1 to skip default base path
          auto basePathArg = op->getBlock()->getArgument(1 + i);
          assert(isa<om::BasePathType>(basePathArg.getType()) &&
                 "expected a passthrough base path");
          basePath = basePathArg;
        }
      }
    }

    rewriter.replaceOpWithNewOp<om::PathCreateOp>(
        op, pathType, om::TargetKindAttr::get(op.getContext(), targetKind),
        basePath, symbol);
    return success();
  }

  const PathInfoTable &pathInfoTable;
};

struct WireOpConversion : public OpConversionPattern<WireOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WireOp wireOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto wireValue = dyn_cast<FIRRTLPropertyValue>(wireOp.getResult());

    // If the wire isn't a Property, not much we can do here.
    if (!wireValue)
      return failure();

    // If the wire isn't inside a graph region, we can't trivially remove it. In
    // practice, this pattern does run for wires in graph regions, so this check
    // should pass and we can proceed with the trivial rewrite.
    auto regionKindInterface = wireOp->getParentOfType<RegionKindInterface>();
    if (!regionKindInterface)
      return failure();
    if (regionKindInterface.getRegionKind(0) != RegionKind::Graph)
      return failure();

    // Find the assignment to the wire.
    PropAssignOp propAssign = getPropertyAssignment(wireValue);
    if (!propAssign)
      return failure();

    // Use the source of the assignment instead of the wire.
    rewriter.replaceOp(wireOp, propAssign.getSrc());

    // Erase the source of the assignment.
    rewriter.eraseOp(propAssign);

    return success();
  }
};

struct AnyCastOpConversion : public OpConversionPattern<ObjectAnyRefCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ObjectAnyRefCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AnyCastOp>(op, adaptor.getInput());
    return success();
  }
};

struct ObjectSubfieldOpConversion
    : public OpConversionPattern<firrtl::ObjectSubfieldOp> {
  using OpConversionPattern::OpConversionPattern;

  ObjectSubfieldOpConversion(
      const TypeConverter &typeConverter, MLIRContext *context,
      const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable)
      : OpConversionPattern(typeConverter, context),
        classTypeTable(classTypeTable) {}

  LogicalResult
  matchAndRewrite(firrtl::ObjectSubfieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto omClassType = dyn_cast<om::ClassType>(adaptor.getInput().getType());
    if (!omClassType)
      return failure();

    // Convert the field-index used by the firrtl implementation, to a symbol,
    // as used by the om implementation.
    auto firrtlClassType =
        classTypeTable.lookup(omClassType.getClassName().getAttr());
    if (!firrtlClassType)
      return failure();

    const auto &element = firrtlClassType.getElement(op.getIndex());
    // We cannot convert input ports to fields.
    if (element.direction == Direction::In)
      return failure();

    auto field = FlatSymbolRefAttr::get(element.name);
    auto path = rewriter.getArrayAttr({field});
    auto type = typeConverter->convertType(element.type);
    rewriter.replaceOpWithNewOp<om::ObjectFieldOp>(op, type, adaptor.getInput(),
                                                   path);
    return success();
  }

  const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable;
};

struct ClassFieldsOpConversion : public OpConversionPattern<ClassFieldsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassFieldsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ClassFieldsOp>(op, adaptor.getOperands(),
                                               adaptor.getFieldLocsAttr());
    return success();
  }
};

struct ObjectOpConversion : public OpConversionPattern<om::ObjectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(om::ObjectOp objectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the object with a new object using the converted actual parameter
    // types from the adaptor.
    rewriter.replaceOpWithNewOp<om::ObjectOp>(objectOp, objectOp.getType(),
                                              adaptor.getClassNameAttr(),
                                              adaptor.getActualParams());
    return success();
  }
};

static LogicalResult convertClassLike(om::ClassLike classOp,
                                      TypeConverter typeConverter,
                                      ConversionPatternRewriter &rewriter) {
  Block *body = classOp.getBodyBlock();
  TypeConverter::SignatureConversion result(body->getNumArguments());

  // Convert block argument types.
  if (failed(
          typeConverter.convertSignatureArgs(body->getArgumentTypes(), result)))
    return failure();

  // Convert the body.
  if (failed(rewriter.convertRegionTypes(body->getParent(), typeConverter,
                                         &result)))
    return failure();

  rewriter.modifyOpInPlace(classOp, [&]() {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](TypeAttr typeAttr) {
      return mlir::TypeAttr::get(
          typeConverter.convertType(typeAttr.getValue()));
    });
    classOp.replaceFieldTypes(replacer);
  });

  return success();
}

struct ClassOpSignatureConversion : public OpConversionPattern<om::ClassOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(om::ClassOp classOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return convertClassLike(classOp, *typeConverter, rewriter);
  }
};

struct ClassExternOpSignatureConversion
    : public OpConversionPattern<om::ClassExternOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(om::ClassExternOp classOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return convertClassLike(classOp, *typeConverter, rewriter);
  }
};

struct ObjectFieldOpConversion : public OpConversionPattern<ObjectFieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ObjectFieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the object field with a new object field of the appropriate
    // result type based on the type converter.
    auto type = typeConverter->convertType(op.getType());
    if (!type)
      return failure();

    rewriter.replaceOpWithNewOp<ObjectFieldOp>(op, type, adaptor.getObject(),
                                               adaptor.getFieldPathAttr());

    return success();
  }
};

/// Replace OM-to-FIRRTL casts with the OM value.
struct UnrealizedConversionCastOpConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();
    auto type = typeConverter->convertType(op.getResult(0));
    if (!type || type != adaptor.getOperands()[0].getType())
      return failure();
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Setup
//===----------------------------------------------------------------------===//

static void populateConversionTarget(ConversionTarget &target) {
  // FIRRTL dialect operations inside ClassOps or not using only OM types must
  // be legalized.
  target.addDynamicallyLegalDialect<FIRRTLDialect>(
      [](Operation *op) { return !op->getParentOfType<om::ClassLike>(); });

  // OM dialect operations are legal if they don't use FIRRTL types.
  target.addDynamicallyLegalDialect<OMDialect>([](Operation *op) {
    auto containsFIRRTLType = [](Type type) {
      return type
          .walk([](Type type) {
            return failure(isa<FIRRTLDialect>(type.getDialect()));
          })
          .wasInterrupted();
    };
    auto noFIRRTLOperands =
        llvm::none_of(op->getOperandTypes(), [&containsFIRRTLType](Type type) {
          return containsFIRRTLType(type);
        });
    auto noFIRRTLResults =
        llvm::none_of(op->getResultTypes(), [&containsFIRRTLType](Type type) {
          return containsFIRRTLType(type);
        });
    return noFIRRTLOperands && noFIRRTLResults;
  });

  // OM Class ops are legal if they don't use FIRRTL types for block arguments.
  target.addDynamicallyLegalOp<om::ClassOp, om::ClassExternOp>(
      [](Operation *op) -> std::optional<bool> {
        auto classLike = dyn_cast<om::ClassLike>(op);
        if (!classLike)
          return std::nullopt;
        auto fieldNames = classLike.getFieldNames();
        if (!llvm::all_of(fieldNames, [&](auto field) {
              std::optional<Type> type =
                  classLike.getFieldType(cast<StringAttr>(field));
              return type.has_value() && !isa<FIRRTLType>(type.value());
            }))
          return false;

        return llvm::none_of(
            classLike.getBodyBlock()->getArgumentTypes(),
            [](Type type) { return isa<FIRRTLDialect>(type.getDialect()); });
      });
}

static void populateTypeConverter(TypeConverter &converter) {
  // Convert FIntegerType to IntegerType.
  converter.addConversion(
      [](IntegerType type) { return OMIntegerType::get(type.getContext()); });
  converter.addConversion([](FIntegerType type) {
    // The actual width of the IntegerType doesn't actually get used; it will be
    // folded away by the dialect conversion infrastructure to the type of the
    // APSIntAttr used in the FIntegerConstantOp.
    return OMIntegerType::get(type.getContext());
  });

  // Convert FIRRTL StringType to OM StringType.
  converter.addConversion([](om::StringType type) { return type; });
  converter.addConversion([](firrtl::StringType type) {
    return om::StringType::get(type.getContext());
  });

  // Convert FIRRTL PathType to OM PathType.
  converter.addConversion([](om::PathType type) { return type; });
  converter.addConversion([](om::BasePathType type) { return type; });
  converter.addConversion([](om::FrozenPathType type) { return type; });
  converter.addConversion([](om::FrozenBasePathType type) { return type; });
  converter.addConversion([](firrtl::PathType type) {
    return om::PathType::get(type.getContext());
  });

  // Convert FIRRTL Class type to OM Class type.
  converter.addConversion([](om::ClassType type) { return type; });
  converter.addConversion([](firrtl::ClassType type) {
    return om::ClassType::get(type.getContext(), type.getNameAttr());
  });

  // Convert FIRRTL AnyRef type to OM Any type.
  converter.addConversion([](om::AnyType type) { return type; });
  converter.addConversion([](firrtl::AnyRefType type) {
    return om::AnyType::get(type.getContext());
  });

  // Convert FIRRTL List type to OM List type.
  auto convertListType = [&converter](auto type) -> std::optional<mlir::Type> {
    // If the element type is already in the OM dialect, there's nothing to do.
    if (isa<om::OMDialect>(type.getElementType().getDialect()))
      return type;
    auto elementType = converter.convertType(type.getElementType());
    if (!elementType)
      return {};
    return om::ListType::get(elementType);
  };

  converter.addConversion(
      [convertListType](om::ListType type) -> std::optional<mlir::Type> {
        // Convert any om.list<firrtl> -> om.list<om>
        return convertListType(type);
      });

  converter.addConversion(
      [convertListType](firrtl::ListType type) -> std::optional<mlir::Type> {
        // Convert any firrtl.list<firrtl> -> om.list<om>
        return convertListType(type);
      });

  // Convert FIRRTL Bool type to OM
  converter.addConversion(
      [](BoolType type) { return IntegerType::get(type.getContext(), 1); });

  // Convert FIRRTL double type to OM.
  converter.addConversion(
      [](DoubleType type) { return Float64Type::get(type.getContext()); });

  // Add a target materialization such that the conversion does not fail when a
  // type conversion could not be reconciled automatically by the framework.
  converter.addTargetMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        return UnrealizedConversionCastOp::create(builder, loc, type, values[0])
            ->getResult(0);
      });

  // Add a source materialization such that the conversion does not fail when a
  // type conversion could not be reconciled automatically by the framework.
  converter.addSourceMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        return UnrealizedConversionCastOp::create(builder, loc, type, values[0])
            ->getResult(0);
      });
}

static void populateRewritePatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    const PathInfoTable &pathInfoTable,
    const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable) {
  patterns.add<FIntegerConstantOpConversion>(converter, patterns.getContext());
  patterns.add<StringConstantOpConversion>(converter, patterns.getContext());
  patterns.add<PathOpConversion>(converter, patterns.getContext(),
                                 pathInfoTable);
  patterns.add<WireOpConversion>(converter, patterns.getContext());
  patterns.add<AnyCastOpConversion>(converter, patterns.getContext());
  patterns.add<ObjectSubfieldOpConversion>(converter, patterns.getContext(),
                                           classTypeTable);
  patterns.add<ClassFieldsOpConversion>(converter, patterns.getContext());
  patterns.add<ClassOpSignatureConversion>(converter, patterns.getContext());
  patterns.add<ClassExternOpSignatureConversion>(converter,
                                                 patterns.getContext());
  patterns.add<ObjectOpConversion>(converter, patterns.getContext());
  patterns.add<ObjectFieldOpConversion>(converter, patterns.getContext());
  patterns.add<ListCreateOpConversion>(converter, patterns.getContext());
  patterns.add<ListConcatOpConversion>(converter, patterns.getContext());
  patterns.add<BoolConstantOpConversion>(converter, patterns.getContext());
  patterns.add<DoubleConstantOpConversion>(converter, patterns.getContext());
  patterns.add<IntegerAddOpConversion>(converter, patterns.getContext());
  patterns.add<IntegerMulOpConversion>(converter, patterns.getContext());
  patterns.add<IntegerShrOpConversion>(converter, patterns.getContext());
  patterns.add<IntegerShlOpConversion>(converter, patterns.getContext());
  patterns.add<UnrealizedConversionCastOpConversion>(converter,
                                                     patterns.getContext());
}

// Convert to OM ops and types in Classes or Modules.
LogicalResult LowerClassesPass::dialectConversion(
    Operation *op, const PathInfoTable &pathInfoTable,
    const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable) {
  ConversionTarget target(getContext());
  populateConversionTarget(target);

  TypeConverter typeConverter;
  populateTypeConverter(typeConverter);

  RewritePatternSet patterns(&getContext());
  populateRewritePatterns(patterns, typeConverter, pathInfoTable,
                          classTypeTable);

  return applyPartialConversion(op, target, std::move(patterns));
}
