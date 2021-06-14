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
#include "llvm/ADT/SmallPtrSet.h"
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
struct BlackBoxReaderPass : public BlackBoxReaderBase<BlackBoxReaderPass> {
  void runOnOperation() override;
  bool runOnAnnotation(Operation *op, Annotation anno, OpBuilder &builder);
  bool loadFile(Operation *op, StringRef inputPath, OpBuilder &builder);
  void setOutputFile(VerbatimOp op, StringAttr fileNameAttr);

  using BlackBoxReaderBase::inputPrefix;
  using BlackBoxReaderBase::resourcePrefix;

private:
  /// A set of the files generated so far. This is used to prevent two
  /// annotations from generating the same file.
  SmallPtrSet<Attribute, 8> emittedFiles;

  /// A list of all files which will be included in the file list.  This is
  /// subset of all emitted files.
  SmallVector<StringRef> fileListFiles;

  /// The target directory to output black boxes into. Can be changed
  /// through `firrtl.transforms.BlackBoxTargetDirAnno` annotations.
  StringRef targetDir;

  /// The file list file name (sic) for black boxes. If set, generates a file
  /// that lists all non-header source files for black boxes. Can be changed
  /// through `firrtl.transforms.BlackBoxResourceFileNameAnno` annotations.
  StringRef resourceFileName;
};
} // end anonymous namespace

/// Emit the annotated source code for black boxes in a circuit.
void BlackBoxReaderPass::runOnOperation() {
  CircuitOp circuitOp = getOperation();
  auto context = &getContext();

  // Internalize some string attributes for easy reference later.
  StringRef targetDirAnnoClass = "firrtl.transforms.BlackBoxTargetDirAnno";
  StringRef resourceFileNameAnnoClass =
      "firrtl.transforms.BlackBoxResourceFileNameAnno";

  // Determine the target directory and resource file name from the
  // annotations present on the circuit operation.
  targetDir = ".";
  resourceFileName = "firrtl_black_box_resource_files.f";

  // Process black box annotations on the circuit.  Some of these annotations
  // will affect how the rest of the annotations are resolved.
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
    circuitOp->setAttr("annotations", ArrayAttr::get(context, filteredAnnos));

  LLVM_DEBUG(llvm::dbgs() << "Black box target directory: " << targetDir << "\n"
                          << "Black box resource file name: "
                          << resourceFileName << "\n");

  // Newly generated IR will be placed at the end of the circuit.
  auto builder = OpBuilder::atBlockEnd(circuitOp->getBlock());

  // Gather the relevant annotations on all modules in the circuit.
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
      op.setAttr("annotations", ArrayAttr::get(context, filteredAnnos));
  }

  // If we have emitted any files, generate a file list operation that
  // documents the additional annotation-controlled file listing to be
  // created.
  if (!fileListFiles.empty()) {
    // Output the file list in sorted order.
    llvm::sort(fileListFiles.begin(), fileListFiles.end());

    // Create the file list contents by prepending the file name with the target
    // directory, and putting each file on its own line.
    std::string output;
    llvm::raw_string_ostream os(output);
    llvm::interleave(
        fileListFiles, os,
        [&](StringRef fileName) {
          SmallString<32> filePath(targetDir);
          llvm::sys::path::append(filePath, fileName);
          os << filePath;
        },
        "\n");

    // Put the file list in to a verbatim op.
    auto op =
        builder.create<VerbatimOp>(circuitOp->getLoc(), std::move(output));

    // Attach the output file information to the verbatim op.
    auto trueAttr = BoolAttr::get(context, true);
    op->setAttr("output_file",
                OutputFileAttr::get(StringAttr::get(context, ""),
                                    StringAttr::get(context, resourceFileName),
                                    /*exclude_from_filelist=*/trueAttr,
                                    /*exclude_replicated_ops=*/trueAttr,
                                    context));
  }

  // Clean up.
  emittedFiles.clear();
  fileListFiles.clear();
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

    // Skip this inline annotation if the target is already generated.
    if (emittedFiles.count(name))
      return true;

    // Create an IR node to hold the contents.
    auto verbatim = builder.create<VerbatimOp>(op->getLoc(), text);
    setOutputFile(verbatim, name);
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
    if (loadFile(op, inputPath, builder))
      return true;
    op->emitError("Cannot find file ") << inputPath;
    signalPassFailure();
    return false;
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

    // Note that we always treat `resourceId` as a relative path, as the
    // previous Scala implementation tended to emit `/foo.v` as resourceId.
    auto relativeResourceId =
        llvm::sys::path::relative_path(resourceId.getValue());

    SmallVector<StringRef> roots;
    StringRef(resourcePrefix).split(roots, ':');
    for (auto root : roots) {
      SmallString<128> inputPath(root);
      llvm::sys::path::append(inputPath, relativeResourceId);
      if (loadFile(op, inputPath, builder))
        return true; // found file
    }
    op->emitError("Cannot find file ") << resourceId.getValue();
    signalPassFailure();
    return false;
  }

  // Annotation was not concerned with black boxes.
  return false;
}

/// Copies a black box source file to the appropriate location in the target
/// directory.
bool BlackBoxReaderPass::loadFile(Operation *op, StringRef inputPath,
                                  OpBuilder &builder) {
  auto fileName = llvm::sys::path::filename(inputPath);
  LLVM_DEBUG(llvm::dbgs() << "Add black box source  `" << fileName << "` from `"
                          << inputPath << "`\n");

  // Skip this annotation if the target is already loaded.
  auto fileNameAttr = builder.getStringAttr(fileName);
  if (emittedFiles.count(fileNameAttr))
    return true;

  // Open and read the input file.
  std::string errorMessage;
  auto input = mlir::openInputFile(inputPath, &errorMessage);
  if (!input)
    return false;

  // Create an IR node to hold the contents.
  auto verbatimOp =
      builder.create<VerbatimOp>(op->getLoc(), input->getBuffer());
  setOutputFile(verbatimOp, fileNameAttr);
  return true;
}

/// This function is called for every file generated.  It does the following
/// things:
///  1. Attaches the output file attribute to the VerbatimOp.
///  2. Record that the file has been generated to avoid duplicates.
///  3. Add each file name to the generated "file list" file.
void BlackBoxReaderPass::setOutputFile(VerbatimOp op, StringAttr fileNameAttr) {
  auto *context = &getContext();
  auto fileName = fileNameAttr.getValue();
  // Exclude Verilog header files since we expect them to be included
  // explicitly by compiler directives in other source files.
  auto ext = llvm::sys::path::extension(fileName);
  bool exclude = (ext == ".h" || ext == ".vh" || ext == ".svh");
  auto trueAttr = BoolAttr::get(context, true);
  op->setAttr("output_file",
              OutputFileAttr::get(
                  StringAttr::get(context, targetDir), fileNameAttr,
                  /*exclude_from_filelist=*/BoolAttr::get(context, exclude),
                  /*exclude_replicated_ops=*/trueAttr, context));

  // Record that this file has been generated.
  assert(!emittedFiles.count(fileNameAttr) &&
         "Can't generate the same file twice.");
  emittedFiles.insert(fileNameAttr);

  // Append this file to the file list if its not excluded.
  if (!exclude)
    fileListFiles.push_back(fileName);
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
