//===- ResolvePaths.cpp - Resolve path operations ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the ResolvePathsPass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/OwningModuleCache.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace firrtl;

namespace {
struct PathResolver {
  PathResolver(CircuitOp circuit, InstanceGraph &instanceGraph)
      : circuit(circuit), symbolTable(circuit), instanceGraph(instanceGraph),
        hierPathCache(circuit, symbolTable),
        builder(OpBuilder::atBlockBegin(circuit->getBlock())) {}

  /// This function will find the operation targeted and create a hierarchical
  /// path operation if needed. If the target is resolved, the op will either
  /// be a reference to the HierPathOp, or null if no HierPathOp was needed.
  LogicalResult resolveHierPath(Location loc, FModuleOp owningModule,
                                const AnnoPathValue &target,
                                FlatSymbolRefAttr &op) {

    // We want to root this path at the top level module, or in the case of an
    // unreachable module, we settle for as high as we can get.
    auto module = target.ref.getModule();
    if (!target.instances.empty())
      module = target.instances.front()->getParentOfType<FModuleLike>();
    auto *node = instanceGraph[module];
    while (true) {
      // If the path is rooted at the owning module, we're done.
      if (node->getModule() == owningModule)
        break;
      // If there are no more parents, then the path op lives in a different
      // hierarchy than the HW object it references, which is an error.
      if (node->noUses())
        return emitError(loc)
               << "unable to resolve path relative to owning module "
               << owningModule.getModuleNameAttr();
      // If there is more than one instance of this module, then the path
      // operation is ambiguous, which is an error.
      if (!node->hasOneUse()) {
        auto diag = emitError(loc) << "unable to uniquely resolve target due "
                                      "to multiple instantiation";
        for (auto *use : node->uses())
          diag.attachNote(use->getInstance().getLoc()) << "instance here";
        return diag;
      }
      node = (*node->usesBegin())->getParent();
    }

    // Find the minimal uniquely-identifying path to the operation.  We scan
    // through the list of instances looking for the first module which is
    // multiply instantiated.  We will start our HierPathOp at this instance.
    auto *it = llvm::find_if(target.instances, [&](InstanceOp instance) {
      auto *node = instanceGraph.lookup(instance.getReferencedModuleNameAttr());
      return !node->hasOneUse();
    });

    // If the path is empty, then this is a local reference and we should not
    // construct a HierPathOp.
    auto pathLength = std::distance(it, target.instances.end());
    if (pathLength == 0) {
      op = nullptr;
      return success();
    }

    // Transform the instances into a list of FlatSymbolRefs.
    SmallVector<Attribute> insts;
    insts.reserve(pathLength);
    std::transform(it, target.instances.end(), std::back_inserter(insts),
                   [&](InstanceOp instance) {
                     return OpAnnoTarget(instance).getNLAReference(
                         namespaces[instance->getParentOfType<FModuleLike>()]);
                   });

    // Push a reference to the current module.
    insts.push_back(
        FlatSymbolRefAttr::get(target.ref.getModule().getModuleNameAttr()));
    auto instAttr = ArrayAttr::get(circuit.getContext(), insts);

    // Return the hierchical path.
    op = hierPathCache.getRefFor(instAttr);
    return success();
  }

  LogicalResult resolve(OwningModuleCache &cache, UnresolvedPathOp unresolved) {
    auto loc = unresolved.getLoc();
    ImplicitLocOpBuilder b(loc, unresolved);
    auto *context = b.getContext();

    /// Spelling takes the form
    /// "OMReferenceTarget:~Circuit|Foo/bar:Bar>member".
    auto target = unresolved.getTarget();

    // OMDeleted nodes do not have a target, so it is impossible to resolve
    // them to a real path.  We create a special constant for these path
    // values.
    if (target.consume_front("OMDeleted")) {
      if (!target.empty())
        return emitError(loc, "OMDeleted references can not have targets");
      // Deleted targets are turned into OMReference targets with a dangling
      // id
      // - i.e. the id is not attached to any target.
      auto targetKind = TargetKindAttr::get(context, TargetKind::Reference);
      auto id = DistinctAttr::create(UnitAttr::get(context));
      auto resolved = b.create<PathOp>(targetKind, id);
      unresolved->replaceAllUsesWith(resolved);
      unresolved.erase();
      return success();
    }

    // Parse the OM target kind.
    TargetKind targetKind;
    if (target.consume_front("OMDontTouchedReferenceTarget")) {
      targetKind = TargetKind::DontTouch;
    } else if (target.consume_front("OMInstanceTarget")) {
      targetKind = TargetKind::Instance;
    } else if (target.consume_front("OMMemberInstanceTarget")) {
      targetKind = TargetKind::MemberInstance;
    } else if (target.consume_front("OMMemberReferenceTarget")) {
      targetKind = TargetKind::MemberReference;
    } else if (target.consume_front("OMReferenceTarget")) {
      targetKind = TargetKind::Reference;
    } else {
      return emitError(loc)
             << "unknown or missing OM reference type in target string: \""
             << target << "\"";
    }
    auto targetKindAttr = TargetKindAttr::get(context, targetKind);

    // Parse the target.
    if (!target.consume_front(":"))
      return emitError(loc, "expected ':' in target string");

    auto token = tokenizePath(target);
    if (!token)
      return emitError(loc)
             << "cannot tokenize annotation path \"" << target << "\"";

    // Resolve the target to a target.
    auto path = resolveEntities(*token, circuit, symbolTable, targetCache);
    if (!path)
      return failure();

    // Make sure that we are targeting a leaf of the operation. That way lower
    // types can't split a single reference into many, and cause ambiguity. If
    // we are targeting a module, the type will be null.
    if (Type targetType = path->ref.getType()) {
      auto fieldId = path->fieldIdx;
      auto baseType = dyn_cast<FIRRTLBaseType>(targetType);
      if (!baseType)
        return emitError(loc, "unable to target non-hardware type ")
               << targetType;
      targetType = hw::FieldIdImpl::getFinalTypeByFieldID(baseType, fieldId);
      if (isa<BundleType, FVectorType>(targetType))
        return emitError(loc, "unable to target aggregate type ") << targetType;
    }

    // Create a unique ID.
    auto id = DistinctAttr::create(UnitAttr::get(context));

    auto owningModule = cache.lookup(unresolved);
    StringRef moduleName = "nullptr";
    if (owningModule)
      moduleName = owningModule.getModuleName();
    if (!owningModule)
      return unresolved->emitError("path does not have a single owning module");

    // Resolve a unique path to the operation in question.
    FlatSymbolRefAttr hierPathName;
    if (failed(resolveHierPath(loc, owningModule, *path, hierPathName)))
      return failure();

    // Create the annotation.
    NamedAttrList fields;
    fields.append("id", id);
    fields.append("class", StringAttr::get(context, "circt.tracker"));
    if (hierPathName)
      fields.append("circt.nonlocal", hierPathName);
    if (path->fieldIdx != 0)
      fields.append("circt.fieldID", b.getI64IntegerAttr(path->fieldIdx));
    auto annotation = DictionaryAttr::get(context, fields);

    // Attach the annotation to the target.
    auto annoTarget = path->ref;
    auto annotations = annoTarget.getAnnotations();
    annotations.addAnnotations(annotation);
    if (targetKindAttr.getValue() == TargetKind::DontTouch)
      annotations.addDontTouch();
    annoTarget.setAnnotations(annotations);

    // Create the path operation.
    auto resolved = b.create<PathOp>(targetKindAttr, id);
    unresolved->replaceAllUsesWith(resolved);
    unresolved.erase();
    return success();
  }

  CircuitOp circuit;
  SymbolTable symbolTable;
  CircuitTargetCache targetCache;
  InstanceGraph &instanceGraph;
  hw::InnerSymbolNamespaceCollection namespaces;
  HierPathCache hierPathCache;
  OpBuilder builder;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct ResolvePathsPass : public ResolvePathsBase<ResolvePathsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void ResolvePathsPass::runOnOperation() {
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  PathResolver resolver(circuit, instanceGraph);
  OwningModuleCache cache(instanceGraph);
  auto result = circuit.walk([&](UnresolvedPathOp unresolved) {
    if (failed(resolver.resolve(cache, unresolved))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    signalPassFailure();
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createResolvePathsPass() {
  return std::make_unique<ResolvePathsPass>();
}
