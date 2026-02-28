//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AnnotateInputOnlyModules pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "firrtl-annotate-input-only-modules"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_ANNOTATEINPUTONLYMODULES
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct AnnotateInputOnlyModulesPass
    : public circt::firrtl::impl::AnnotateInputOnlyModulesBase<
          AnnotateInputOnlyModulesPass> {
  void runOnOperation() override;

  LogicalResult initialize(MLIRContext *context) override {
    // Cache the inline annotation.
    inlineAnno = DictionaryAttr::getWithSorted(
        context, {{StringAttr::get(context, "class"),
                   StringAttr::get(context, inlineAnnoClass)}});
    return success();
  }

  mlir::DictionaryAttr inlineAnno;
};

} // end anonymous namespace

void AnnotateInputOnlyModulesPass::runOnOperation() {
  auto circuit = getOperation();
  bool changed = false;
  auto &instanceInfo = getAnalysis<InstanceInfo>();
  for (auto module : circuit.getOps<FModuleOp>()) {
    // Inline input only modules in design.
    // Don't inline if the module is public or has a layer enabled.
    if (!instanceInfo.anyInstanceInEffectiveDesign(module) ||
        module.isPublic() || !module.getLayers().empty())
      continue;

    // Check if the module has only input ports (no output ports)
    bool hasHardwareOutputPort =
        llvm::any_of(module.getPorts(), [&](auto port) {
          return port.direction == Direction::Out &&
                 type_isa<FIRRTLBaseType>(port.type);
        });

    // If the module has only input ports, add InlineAnnotation
    if (hasHardwareOutputPort)
      continue;

    AnnotationSet annos(module);

    // Check if InlineAnnotation or DontTouchAnnotation exists
    if (annos.hasAnnotation(inlineAnnoClass) || annos.hasDontTouch())
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Annotating inline annotation "
                            << module.getModuleName() << "\n");

    // Create InlineAnnotation
    annos.addAnnotations(ArrayRef<Attribute>{inlineAnno});
    annos.applyToOperation(module);

    ++numAnnotated;
    changed = true;
  }

  if (!changed)
    return markAllAnalysesPreserved();

  markAnalysesPreserved<igraph::InstanceGraph, InstanceInfo>();
}
