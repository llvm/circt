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
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace firrtl;

namespace {
class EmitMetadataPass : public EmitMetadataBase<EmitMetadataPass> {
  LogicalResult emitRetimeModulesMetadata();
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
      op->emitError("More than one ") << annoClass << " annotation attached";
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

void EmitMetadataPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  // We need this for SV verbatim and HW attributes.
  registry.insert<hw::HWDialect, sv::SVDialect>();
}

void EmitMetadataPass::runOnOperation() {
  if (failed(emitRetimeModulesMetadata()))
    return signalPassFailure();

  // This pass does not modify the hierarchy.
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createEmitMetadataPass() {
  return std::make_unique<EmitMetadataPass>();
}
