//===- EmitMetadata.cpp - Emit various types of metadata --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the EmitMetadata pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace firrtl;

namespace {
class EmitMetadataPass : public EmitMetadataBase<EmitMetadataPass> {
  LogicalResult emitRetimeModulesMetadata();
  LogicalResult emitSitestBlackboxMetadata();
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;
};
} // end anonymous namespace

/// This will search for a target annotation and remove it from the operation.
/// If the annotation has a filename, it will be returned in the output
/// argument.  If the annotation is missing the filename member, or if more than
/// one matching annotation is attached, it will print an error and return
/// failure.
static LogicalResult removeAnnotationWithFilename(Operation *op,
                                                  StringRef annoClass,
                                                  StringRef &filename) {
  filename = "";
  bool error = false;
  AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
    // If there was a previous error or its not a match, continue.
    if (error || !anno.isClass(annoClass))
      return false;

    // If we have already found a matching annotation, error.
    if (!filename.empty()) {
      op->emitError("more than one ") << annoClass << " annotation attached";
      error = true;
      return false;
    }

    // Get the filename from the annotation.
    auto filenameAttr = anno.getMember<StringAttr>("filename");
    if (!filenameAttr) {
      op->emitError(annoClass) << " requires a filename";
      error = true;
      return false;
    }

    // Require a non-empty filename.
    filename = filenameAttr.getValue();
    if (filename.empty()) {
      op->emitError(annoClass) << " requires a non-empty filename";
      error = true;
      return false;
    }

    return true;
  });

  // If there was a problem above, return failure.
  return failure(error);
}

/// This function collects the name of each module annotated and prints them
/// all as a JSON array.
LogicalResult EmitMetadataPass::emitRetimeModulesMetadata() {

  // Circuit level annotation.
  auto *retimeModulesAnnoClass =
      "sifive.enterprise.firrtl.RetimeModulesAnnotation";
  // Per module annotation.
  auto *retimeModuleAnnoClass =
      "sifive.enterprise.firrtl.RetimeModuleAnnotation";

  auto *context = &getContext();
  auto circuitOp = getOperation();

  // Get the filename, removing the annotation from the circuit.
  StringRef filename;
  if (failed(removeAnnotationWithFilename(circuitOp, retimeModulesAnnoClass,
                                          filename)))
    return failure();

  if (filename.empty())
    return success();

  // Create a string buffer for the json data.
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  llvm::json::OStream j(os);

  // The output is a json array with each element a module name.
  unsigned index = 0;
  SmallVector<Attribute> symbols;
  SmallString<3> placeholder;
  j.array([&] {
    for (auto module : circuitOp.getBody()->getOps<FModuleLike>()) {
      // The annotation has no supplemental information, just remove it.
      if (!AnnotationSet::removeAnnotations(module, retimeModuleAnnoClass))
        continue;

      // We use symbol substitution to make sure we output the correct thing
      // when the module goes through renaming.
      j.value(("{{" + Twine(index++) + "}}").str());
      symbols.push_back(SymbolRefAttr::get(context, module.moduleName()));
    }
  });

  // Put the retime information in a verbatim operation.
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());
  auto verbatimOp = builder.create<sv::VerbatimOp>(
      circuitOp.getLoc(), buffer, ValueRange(), builder.getArrayAttr(symbols));
  auto fileAttr = hw::OutputFileAttr::getFromFilename(
      context, filename, /*excludeFromFilelist=*/true);
  verbatimOp->setAttr("output_file", fileAttr);
  return success();
}

/// This function finds all external modules which will need to be generated for
/// the test harness to run.
LogicalResult EmitMetadataPass::emitSitestBlackboxMetadata() {
  auto *dutBlackboxAnnoClass =
      "sifive.enterprise.firrtl.SitestBlackBoxAnnotation";
  auto *testBlackboxAnnoClass =
      "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation";
  auto *dutAnnoClass = "sifive.enterprise.firrtl.MarkDUTAnnotation";

  // Any extmodule with these annotations or one of these ScalaClass classes
  // should be excluded from the blackbox list.
  auto *scalaClassAnnoClass = "sifive.enterprise.firrtl.ScalaClassAnnotation";
  std::array<StringRef, 3> classBlackList = {
      "freechips.rocketchip.util.BlackBoxedROM", "chisel3.shim.CloneModule",
      "sifive.enterprise.grandcentral.MemTap"};
  std::array<StringRef, 5> blackListedAnnos = {
      "firrtl.transforms.BlackBoxInlineAnno",
      "firrtl.transforms.BlackBoxResourceAnno",
      "sifive.enterprise.grandcentral.DataTapsAnnotation",
      "sifive.enterprise.grandcentral.MemTapAnnotation",
      "sifive.enterprise.grandcentral.transforms.SignalMappingAnnotation"};

  auto *context = &getContext();
  auto circuitOp = getOperation();

  // Get the filenames from the annotations.
  StringRef dutFilename, testFilename;
  if (failed(removeAnnotationWithFilename(circuitOp, dutBlackboxAnnoClass,
                                          dutFilename)) ||
      failed(removeAnnotationWithFilename(circuitOp, testBlackboxAnnoClass,
                                          testFilename)))
    return failure();

  // If we don't have either annotation, no need to run this pass.
  if (dutFilename.empty() && testFilename.empty())
    return success();

  auto *body = circuitOp.getBody();

  // Find the device under test and create a set of all modules underneath it.
  DenseSet<Operation *> dutModuleSet;
  auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
    return AnnotationSet(&op).hasAnnotation(dutAnnoClass);
  });
  if (it != body->end()) {
    auto instanceGraph = getAnalysis<InstanceGraph>();
    auto *node = instanceGraph.lookup(&(*it));
    llvm::for_each(llvm::depth_first(node), [&](InstanceGraphNode *node) {
      dutModuleSet.insert(node->getModule());
    });
  }

  // Find all extmodules in the circuit. Check if they are black-listed from
  // being included in the list. If they are not, separate them into two groups
  // depending on if theyre in the DUT or the test harness.
  SmallVector<StringRef> dutModules;
  SmallVector<StringRef> testModules;
  for (auto extModule : circuitOp.getBody()->getOps<FExtModuleOp>()) {
    // If the module doesn't have a defname, then we can't record it properly.
    // Just skip it.
    if (!extModule.defname())
      continue;

    // If its a generated blackbox, skip it.
    AnnotationSet annos(extModule);
    if (llvm::any_of(blackListedAnnos, [&](auto blackListedAnno) {
          return annos.hasAnnotation(blackListedAnno);
        }))
      continue;

    // If its a blacklisted scala class, skip it.
    if (auto scalaAnnoDict = annos.getAnnotation(scalaClassAnnoClass)) {
      Annotation scalaAnno(scalaAnnoDict);
      auto scalaClass = scalaAnno.getMember<StringAttr>("className");
      if (scalaClass &&
          llvm::is_contained(classBlackList, scalaClass.getValue()))
        continue;
    }

    // Record the defname of the module.
    if (dutModuleSet.contains(extModule)) {
      dutModules.push_back(*extModule.defname());
    } else {
      testModules.push_back(*extModule.defname());
    }
  }

  // This is a helper to create the verbatim output operation.
  auto createOutput = [&](SmallVectorImpl<StringRef> &names,
                          StringRef filename) {
    if (filename.empty())
      return;

    // Sort and remove duplicates.
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());

    // The output is a json array with each element a module name. The
    // defname of a module can't change so we can output them verbatim.
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    llvm::json::OStream j(os);
    j.array([&] {
      for (auto &name : names)
        j.value(name);
    });

    // Put the information in a verbatim operation.
    auto builder = OpBuilder::atBlockEnd(body);
    auto verbatimOp =
        builder.create<sv::VerbatimOp>(circuitOp.getLoc(), buffer);
    auto fileAttr = hw::OutputFileAttr::getFromFilename(
        context, filename, /*excludeFromFilelist=*/true);
    verbatimOp->setAttr("output_file", fileAttr);
  };

  createOutput(testModules, testFilename);
  createOutput(dutModules, dutFilename);
  return success();
}

void EmitMetadataPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  // We need this for SV verbatim and HW attributes.
  registry.insert<hw::HWDialect, sv::SVDialect>();
}

void EmitMetadataPass::runOnOperation() {
  if (failed(emitRetimeModulesMetadata()) ||
      failed(emitSitestBlackboxMetadata()))
    return signalPassFailure();

  // This pass does not modify the hierarchy.
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createEmitMetadataPass() {
  return std::make_unique<EmitMetadataPass>();
}
