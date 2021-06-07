//===- BlackBoxReader.cpp - Ingest black box sources ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Read Verilog source files for black boxes based on corresponding black box
// annotations on the circuit and modules. Primarily based on:
//
// https://github.com/chipsalliance/firrtl/blob/master/src/main/scala/firrtl/
// transforms/BlackBoxSourceHelper.scala
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "firrtl-blackbox-reader"

using namespace circt;
using namespace firrtl;

using hw::OutputFileAttr;
using sv::VerbatimOp;

/// Append a path to an existing path, replacing it if the other path is
/// absolute. This mimicks the behaviour of `foo/bar` and `/foo/bar` being used
/// in a working directory `/home`, resulting in `/home/foo/bar` and `/foo/bar`,
/// respectively.
static void appendPossiblyAbsolutePath(SmallVectorImpl<char> &base,
                                       const Twine &suffix) {
  if (llvm::sys::path::is_absolute(suffix)) {
    base.clear();
    suffix.toVector(base);
  } else {
    llvm::sys::path::append(base, suffix);
  }
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct EmittedFile {
  VerbatimOp op;
  StringAttr fileName;
};

struct BlackBoxReaderPass : public BlackBoxReaderBase<BlackBoxReaderPass> {
  void runOnOperation() override;
  bool runOnAnnotation(Operation *op, Annotation anno, OpBuilder &builder);
  void loadFile(Operation *op, StringRef inputPath, OpBuilder &builder);

  using BlackBoxReaderBase::inputPrefix;
  using BlackBoxReaderBase::resourcePrefix;

private:
  /// The generated `sv.verbatim` nodes.
  SmallVector<EmittedFile, 8> emittedFiles;

  /// The target directory to output black boxes into. Can be changed
  /// through `firrtl.transforms.BlackBoxTargetDirAnno` annotations.
  SmallString<128> targetDir;

  /// The file list file name (sic) for black boxes. If set, generates a file
  /// that lists all non-header source files for black boxes. Can be changed
  /// through `firrtl.transforms.BlackBoxResourceFileNameAnno` annotations.
  SmallString<128> resourceFileName;
};
} // end anonymous namespace

/// Emit the annotated source code for black boxes in a circuit.
void BlackBoxReaderPass::runOnOperation() {
  CircuitOp circuitOp = getOperation();

  // Internalize some string attributes for easy reference later.
  auto cx = &getContext();
  StringRef targetDirAnnoClass = "firrtl.transforms.BlackBoxTargetDirAnno";
  StringRef resourceFileNameAnnoClass =
      "firrtl.transforms.BlackBoxResourceFileNameAnno";

  // Determine the target directory and resource file name from the
  // annotations present on the circuit operation.
  targetDir = ".";
  resourceFileName = "firrtl_black_box_resource_files.f";

  AnnotationSet circuitAnnos(circuitOp);
  SmallVector<Attribute, 4> filteredAnnos;
  for (auto annot : circuitAnnos) {
    // Handle target dir annotation.
    if (annot.isClass(targetDirAnnoClass)) {
      if (auto target = annot.getMember<StringAttr>("targetDir")) {
        targetDir = target.getValue();
        continue;
      }
      circuitOp->emitError(targetDirAnnoClass)
          << " annotation missing \"targetDir\" attribute";
      signalPassFailure();
      continue;
    }

    // Handle resource file name annotation.
    if (annot.isClass(resourceFileNameAnnoClass)) {
      if (auto resourceFN = annot.getMember<StringAttr>("resourceFileName")) {
        resourceFileName = resourceFN.getValue();
        continue;
      }

      circuitOp->emitError(resourceFileNameAnnoClass)
          << " annotation missing \"resourceFileName\" attribute";
      signalPassFailure();
      continue;
    }

    filteredAnnos.push_back(annot.getDict());
  }

  // Update the circuit annotations to exclude the ones we have consumed.
  if (filteredAnnos.empty())
    circuitOp->removeAttr("annotations");
  else
    circuitOp->setAttr("annotations", ArrayAttr::get(cx, filteredAnnos));

  LLVM_DEBUG(llvm::dbgs() << "Black box target directory: " << targetDir << "\n"
                          << "Black box resource file name: "
                          << resourceFileName << "\n");

  // Gather the relevant annotations on all modules in the circuit.
  OpBuilder builder(&getContext());
  builder.setInsertionPointToEnd(getOperation()->getBlock());

  for (auto &op : *circuitOp.getBody()) {
    if (!isa<FModuleOp>(op) && !isa<FExtModuleOp>(op))
      continue;

    SmallVector<Attribute, 4> filteredAnnos;
    for (auto anno : AnnotationSet(&op)) {
      if (!runOnAnnotation(&op, anno, builder))
        filteredAnnos.push_back(anno.getDict());
    }

    // Update the operation annotations to exclude the ones we have consumed.
    if (filteredAnnos.empty())
      op.removeAttr("annotations");
    else
      op.setAttr("annotations", ArrayAttr::get(cx, filteredAnnos));
  }

  // If we have emitted any files, generate a file list operation that
  // documents the additional annotation-controlled file listing to be
  // created.
  if (!emittedFiles.empty()) {
    auto trueAttr = BoolAttr::get(cx, true);
    std::string output;
    llvm::raw_string_ostream os(output);
    for (auto &file : emittedFiles) {
      const auto &fileName = file.fileName.getValue();
      // Exclude Verilog header files since we expect them to be included
      // explicitly by compiler directives in other source files.
      auto ext = llvm::sys::path::extension(fileName);
      bool exclude = (ext == ".h" || ext == ".vh" || ext == ".svh");
      file.op->setAttr(
          "output_file",
          OutputFileAttr::get(StringAttr::get(cx, targetDir), file.fileName,
                              BoolAttr::get(cx, exclude),
                              /*exclude_replicated_ops=*/trueAttr, cx));

      SmallString<32> filePath(targetDir);
      llvm::sys::path::append(filePath, fileName);
      if (!exclude)
        os << filePath << "\n";
    }
    auto op =
        builder.create<VerbatimOp>(circuitOp->getLoc(), std::move(output));
    op->setAttr("output_file",
                OutputFileAttr::get(StringAttr::get(cx, ""),
                                    StringAttr::get(cx, resourceFileName),
                                    /*exclude_from_filelist=*/trueAttr,
                                    /*exclude_replicated_ops=*/trueAttr, cx));
  }

  // Clean up.
  emittedFiles.clear();
}

/// Run on an operation-annotation pair. The annotation need not be a black box
/// annotation. Returns `true` if the annotation was indeed a black box
/// annotation (even if it was incomplete) and should be removed from the op.
bool BlackBoxReaderPass::runOnAnnotation(Operation *op, Annotation anno,
                                         OpBuilder &builder) {
  StringRef inlineAnnoClass = "firrtl.transforms.BlackBoxInlineAnno";
  StringRef pathAnnoClass = "firrtl.transforms.BlackBoxPathAnno";
  StringRef resourceAnnoClass = "firrtl.transforms.BlackBoxResourceAnno";

  // Handle inline annotation.
  if (anno.isClass(inlineAnnoClass)) {
    auto name = anno.getMember<StringAttr>("name");
    auto text = anno.getMember<StringAttr>("text");
    if (!name || !text) {
      op->emitError(inlineAnnoClass)
          << " annotation missing \"name\" or \"text\" attribute";
      signalPassFailure();
      return true;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Add black box source `" << name.getValue() << "` inline\n");

    // Create an IR node to hold the contents.
    emittedFiles.push_back({builder.create<VerbatimOp>(op->getLoc(), text),
                            builder.getStringAttr(name.getValue())});
    return true;
  }

  // Handle path annotation.
  if (anno.isClass(pathAnnoClass)) {
    auto path = anno.getMember<StringAttr>("path");
    if (!path) {
      op->emitError(pathAnnoClass) << " annotation missing \"path\" attribute";
      signalPassFailure();
      return true;
    }
    SmallString<128> inputPath(inputPrefix);
    appendPossiblyAbsolutePath(inputPath, path.getValue());
    loadFile(op, inputPath, builder);
    return true;
  }

  // Handle resource annotation.
  if (anno.isClass(resourceAnnoClass)) {
    auto resourceId = anno.getMember<StringAttr>("resourceId");
    if (!resourceId) {
      op->emitError(resourceAnnoClass)
          << " annotation missing \"resourceId\" attribute";
      signalPassFailure();
      return true;
    }
    SmallString<128> inputPath(resourcePrefix);
    // Note that we always treat `resourceId` as a relative path, as the
    // previous Scala implementation tended to emit `/foo.v` as resourceId.
    llvm::sys::path::append(inputPath, resourceId.getValue());
    loadFile(op, inputPath, builder);
    return true;
  }

  // Annotation was not concerned with black boxes.
  return false;
}

/// Copies a black box source file to the appropriate location in the target
/// directory.
void BlackBoxReaderPass::loadFile(Operation *op, StringRef inputPath,
                                  OpBuilder &builder) {
  auto fileName = llvm::sys::path::filename(inputPath);
  LLVM_DEBUG(llvm::dbgs() << "Add black box source  `" << fileName << "` from `"
                          << inputPath << "`\n");

  // Open and read the input file.
  std::string errorMessage;
  auto input = mlir::openInputFile(inputPath, &errorMessage);
  if (!input) {
    op->emitError(errorMessage);
    signalPassFailure();
    return;
  }

  // Create an IR node to hold the contents.
  emittedFiles.push_back(
      {builder.create<VerbatimOp>(op->getLoc(), input->getBuffer()),
       builder.getStringAttr(fileName)});
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::firrtl::createBlackBoxReaderPass(
    llvm::Optional<StringRef> inputPrefix,
    llvm::Optional<StringRef> resourcePrefix) {
  auto pass = std::make_unique<BlackBoxReaderPass>();
  if (inputPrefix)
    pass->inputPrefix = inputPrefix->str();
  if (resourcePrefix)
    pass->resourcePrefix = resourcePrefix->str();
  return pass;
}
