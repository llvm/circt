//===- FIRRTLInliningInfo.cpp - Inlining classification --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FIRRTLInliningInfo.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace firrtl;

LogicalResult InliningInfo::run() {
  auto inlineAnnoClassAttr =
      StringAttr::get(circuit.getContext(), inlineAnnoClass);
  auto flattenAnnoClassAttr =
      StringAttr::get(circuit.getContext(), flattenAnnoClass);
  for (auto &op : circuit.getOps()) {
    // Initialize module information.  Not order-dependent.
    if (auto module = dyn_cast<FModuleLike>(op)) {
      auto &info = modInfoMap[module];
      AnnotationSet anno(module);
      info.hasInline = anno.hasAnnotation(inlineAnnoClassAttr);
      info.hasFlatten = anno.hasAnnotation(flattenAnnoClassAttr);

      // Reject inline/flatten on anything but a regular module.
      // LowerAnnotations restricts these to FModuleOp; this catches raw IR.
      if (!isa<FModuleOp>(module) && (info.hasInline || info.hasFlatten))
        return emitError(module.getLoc())
               << "inline/flatten annotations are only valid on a regular "
                  "module";

      // Does anything other than an InstanceOp instantiate this?
      auto instantiators = instanceGraph.lookup(module)->uses();
      auto nonInstanceRecIt =
          llvm::find_if(instantiators, [](InstanceRecord *rec) {
            return !isa<InstanceOp>(rec->getInstance());
          });
      bool hasNonInstanceUse = nonInstanceRecIt != instantiators.end();

      if (!module.canDiscardOnUseEmpty() || hasNonInstanceUse)
        info.isLive = info.hasUnflattenedPath = true;
      if (info.hasInline && hasNonInstanceUse) {
        auto diag = mlir::emitWarning(module.getLoc())
                    << "module marked inline is also instantiated by an "
                       "operation that cannot be inlined; it is inlined only "
                       "into its 'firrtl.instance' parents and retained";
        diag.attachNote((*nonInstanceRecIt)->getInstance()->getLoc())
            << "instantiated here";
      }
      continue;
    }

    // Symbol use analysis:

    // Ignore symbol uses in NLAs.
    if (isa<hw::HierPathOp>(op))
      continue;

    // Mark modules live whose symbols are referenced in other ops.
    auto symbolUses = SymbolTable::getSymbolUses(&op);
    if (!symbolUses)
      continue;
    for (const auto &use : *symbolUses) {
      if (auto flat = dyn_cast<FlatSymbolRefAttr>(use.getSymbolRef()))
        if (auto moduleLike = symbolTable.lookup<FModuleLike>(flat.getAttr())) {
          auto &info = modInfoMap[moduleLike];
          info.isLive = info.hasUnflattenedPath = true;
        }
    }
  }

  // Calculate inlining info top-down.
  instanceGraph.walkInversePostOrder([&](igraph::InstanceGraphNode &node) {
    auto *mod = node.getModule().getOperation();
    // Save IPO over FModuleOp's for later.
    if (auto fmod = dyn_cast<FModuleOp>(mod))
      ipoModules.push_back(fmod);
    auto &modInfo = modInfoMap[mod];

    // Skip if no non-inlined path to this module.
    if (!modInfo.hasUnflattenedPath && !modInfo.underFlatten)
      return;

    bool childrenInstantiated =
        modInfo.hasUnflattenedPath && !modInfo.hasFlatten;
    bool moduleMayBeFlattened = modInfo.underFlatten || modInfo.hasFlatten;
    for (auto *edge : node) {
      auto *childMod = edge->getTarget()->getModule().getOperation();
      auto &childInfo = modInfoMap[childMod];
      bool isRegularModule = isa<FModuleOp>(childMod);
      assert(
          (isRegularModule || !(childInfo.hasInline || childInfo.hasFlatten)) &&
          "non-fmoduleop with inline/flatten annotation");
      if (isRegularModule && moduleMayBeFlattened)
        childInfo.underFlatten = true;

      // A non-regular child is never absorbed, so it stays instantiated.
      // A regular child does iff this module does and isn't flattening it.
      if (childrenInstantiated || !isRegularModule)
        childInfo.hasUnflattenedPath = true;

      // Set liveness.
      if (childInfo.hasUnflattenedPath &&
          (!isRegularModule || !childInfo.hasInline))
        childInfo.isLive = true;
    }
  });

  return success();
}

void InliningInfo::print(raw_ostream &os) const {
  // Op accessors are non-const; ops are value handles, so copy.
  CircuitOp circuitOp = circuit;
  for (auto module : circuitOp.getBodyBlock()->getOps<FModuleLike>()) {
    auto info = modInfoMap.lookup(module);
    os << "@" << module.getModuleName() << "\n";
    os << "  hasInline: " << llvm::toStringRef(info.hasInline) << "\n";
    os << "  hasFlatten: " << llvm::toStringRef(info.hasFlatten) << "\n";
    os << "  underFlatten: " << llvm::toStringRef(info.underFlatten) << "\n";
    os << "  hasUnflattenedPath: " << llvm::toStringRef(info.hasUnflattenedPath)
       << "\n";
    os << "  isLive: " << llvm::toStringRef(info.isLive) << "\n";
  }
  os << "inverse post-order:";
  for (auto module : ipoModules)
    os << " @" << module.getModuleName();
  os << "\n";
}

void InliningInfo::dump() const { print(llvm::dbgs()); }
