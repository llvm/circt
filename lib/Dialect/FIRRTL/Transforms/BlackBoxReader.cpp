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

#include "./PassDetails.h"
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
  void runOnAnnotation(Operation *op, DictionaryAttr anno, OpBuilder &builder);
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

  // Internalized annotation class names for fast comparison.
  StringAttr targetDirAnnoClass;
  StringAttr resourceFileNameAnnoClass;
  StringAttr inlineAnnoClass;
  StringAttr pathAnnoClass;
  StringAttr resourceAnnoClass;
};
} // end anonymous namespace

/// Emit the annotated source code for black boxes in a circuit.
void BlackBoxReaderPass::runOnOperation() {
  CircuitOp circuitOp = getOperation();

  // Internalize some string attributes for easy reference later.
  auto cx = &getContext();
  targetDirAnnoClass =
      StringAttr::get(cx, "firrtl.transforms.BlackBoxTargetDirAnno");
  resourceFileNameAnnoClass =
      StringAttr::get(cx, "firrtl.transforms.BlackBoxResourceFileNameAnno");
  inlineAnnoClass = StringAttr::get(cx, "firrtl.transforms.BlackBoxInlineAnno");
  pathAnnoClass = StringAttr::get(cx, "firrtl.transforms.BlackBoxPathAnno");
  resourceAnnoClass =
      StringAttr::get(cx, "firrtl.transforms.BlackBoxResourceAnno");

  // Determine the target directory and resource file name from the
  // annotations present on the circuit operation.
  targetDir = ".";
  resourceFileName = "firrtl_black_box_resource_files.f";

  if (auto attrs = circuitOp->getAttrOfType<ArrayAttr>("annotations")) {
    for (auto attr : attrs) {
      auto anno = attr.dyn_cast<DictionaryAttr>();
      if (!anno)
        continue;
      auto cls = anno.getAs<StringAttr>("class");

      // Handle target dir annotation.
      if (cls == targetDirAnnoClass) {
        auto x = anno.getAs<StringAttr>("targetDir");
        if (!x) {
          circuitOp->emitError(targetDirAnnoClass.getValue() +
                               " annotation missing \"targetDir\" attribute");
          signalPassFailure();
          continue;
        }
        targetDir = x.getValue();
        continue;
      }

      // Handle resource file name annotation.
      if (cls == resourceFileNameAnnoClass) {
        auto x = anno.getAs<StringAttr>("resourceFileName");
        if (!x) {
          circuitOp->emitError(
              resourceFileNameAnnoClass.getValue() +
              " annotation missing \"resourceFileName\" attribute");
          signalPassFailure();
          continue;
        }
        resourceFileName = x.getValue();
        continue;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Black box target directory: " << targetDir << "\n"
                          << "Black box resource file name: "
                          << resourceFileName << "\n");

  // Gather the relevant annotations on all modules in the circuit.
  OpBuilder builder(&getContext());
  builder.setInsertionPointToEnd(getOperation()->getBlock());

  for (auto &op : *circuitOp.getBody()) {
    if (!isa<FModuleOp>(op) && !isa<FExtModuleOp>(op))
      continue;
    if (auto attrs = op.getAttrOfType<ArrayAttr>("annotations")) {
      for (auto attr : attrs) {
        if (auto anno = attr.dyn_cast<DictionaryAttr>())
          runOnAnnotation(&op, anno, builder);
      }
    }
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
          OutputFileAttr::get(file.fileName, BoolAttr::get(cx, exclude),
                              /*exclude_replicated_ops=*/trueAttr, cx));
      if (!exclude)
        os << fileName << "\n";
    }
    auto op =
        builder.create<VerbatimOp>(circuitOp->getLoc(), std::move(output));
    op->setAttr("output_file",
                OutputFileAttr::get(StringAttr::get(cx, resourceFileName),
                                    /*exclude_from_filelist=*/trueAttr,
                                    /*exclude_replicated_ops=*/trueAttr, cx));
  }

  // Clean up.
  emittedFiles.clear();
}

/// Run on an operation annotated with a black box annotation.
void BlackBoxReaderPass::runOnAnnotation(Operation *op, DictionaryAttr anno,
                                         OpBuilder &builder) {
  auto cls = anno.getAs<StringAttr>("class");

  // Handle inline annotation.
  if (cls == inlineAnnoClass) {
    auto name = anno.getAs<StringAttr>("name");
    auto text = anno.getAs<StringAttr>("text");
    if (!name || !text) {
      op->emitError(inlineAnnoClass.getValue() +
                    " annotation missing \"name\" or \"text\" attribute");
      signalPassFailure();
      return;
    }

    // Determine output path.
    SmallString<128> outputPath(targetDir);
    llvm::sys::path::append(outputPath, name.getValue());
    LLVM_DEBUG(llvm::dbgs()
               << "Add black box source `" << outputPath << "` inline\n");

    // Create an IR node to hold the contents.
    emittedFiles.push_back({builder.create<VerbatimOp>(op->getLoc(), text),
                            builder.getStringAttr(outputPath)});
    return;
  }

  // Handle path annotation.
  if (cls == pathAnnoClass) {
    auto path = anno.getAs<StringAttr>("path");
    if (!path) {
      op->emitError(pathAnnoClass.getValue() +
                    " annotation missing \"path\" attribute");
      signalPassFailure();
      return;
    }
    SmallString<128> inputPath(inputPrefix);
    appendPossiblyAbsolutePath(inputPath, path.getValue());
    return loadFile(op, inputPath, builder);
  }

  // Handle resource annotation.
  if (cls == resourceAnnoClass) {
    auto resourceId = anno.getAs<StringAttr>("resourceId");
    if (!resourceId) {
      op->emitError(pathAnnoClass.getValue() +
                    " annotation missing \"resourceId\" attribute");
      signalPassFailure();
      return;
    }
    SmallString<128> inputPath(inputPrefix);
    appendPossiblyAbsolutePath(inputPath, resourcePrefix);
    // Note that we always treat `resourceId` as a relative path, as the
    // previous Scala implementation tended to emit `/foo.v` as resourceId.
    llvm::sys::path::append(inputPath, resourceId.getValue());
    return loadFile(op, inputPath, builder);
  }
}

/// Copies a black box source file to the appropriate location in the target
/// directory.
void BlackBoxReaderPass::loadFile(Operation *op, StringRef inputPath,
                                  OpBuilder &builder) {
  SmallString<128> outputPath(targetDir);
  llvm::sys::path::append(outputPath, llvm::sys::path::filename(inputPath));
  LLVM_DEBUG(llvm::dbgs() << "Add black box source `" << outputPath
                          << "` from `" << inputPath << "`\n");

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
       builder.getStringAttr(outputPath)});
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
