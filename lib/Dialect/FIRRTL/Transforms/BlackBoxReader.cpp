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

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/Debug.h"
#include "circt/Support/Path.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "firrtl-blackbox-reader"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_BLACKBOXREADER
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Data extracted from BlackBoxInlineAnno or BlackBoxPathAnno.
struct AnnotationInfo {
  /// The name of the file that should be created for this BlackBox.
  StringAttr name;
  /// The output directory information for this extmodule.
  hw::OutputFileAttr outputFileAttr;
  /// The body of the BlackBox.  (This should be Verilog text.)
  StringAttr inlineText;

#if !defined(NDEBUG)
  /// Pretty print the AnnotationInfo in a YAML-esque format.
  void print(raw_ostream &os, unsigned indent = 0) const {
    os << llvm::formatv("name: {1}\n"
                        "{0}outputFile: {2}\n"
                        "{0}exclude: {3}\n",
                        llvm::fmt_pad("", indent, 0), name,
                        outputFileAttr.getFilename(),
                        outputFileAttr.getExcludeFromFilelist().getValue());
  };
#endif
};

struct BlackBoxReaderPass
    : public circt::firrtl::impl::BlackBoxReaderBase<BlackBoxReaderPass> {
  using Base::Base;

  void runOnOperation() override;
  bool runOnAnnotation(Operation *op, Annotation anno, OpBuilder &builder,
                       bool isCover, AnnotationInfo &annotationInfo);
  StringAttr loadFile(Operation *op, StringRef inputPath, OpBuilder &builder);
  hw::OutputFileAttr getOutputFile(Operation *origOp, StringAttr fileNameAttr,
                                   bool isCover = false);

private:
  /// A list of all files which will be included in the file list.  This is
  /// subset of all emitted files.
  SmallVector<emit::FileOp> fileListFiles;

  /// The target directory to output black boxes into. Can be changed
  /// through `firrtl.transforms.BlackBoxTargetDirAnno` annotations.
  StringRef targetDir;

  /// The target directory for cover statements.
  StringRef coverDir;

  /// The target directory for testbench files.
  StringRef testBenchDir;

  /// Analyses used by this pass.
  InstanceGraph *instanceGraph;
  InstanceInfo *instanceInfo;

  /// A cache of the modules which have been marked as DUT or a testbench.
  /// This is used to determine the output directory.
  DenseMap<Operation *, bool> dutModuleMap;

  /// An ordered map of Verilog filenames to the annotation-derived information
  /// that will be used to create this file.  Due to situations where multiple
  /// external modules may not deduplicate (e.g., they have different
  /// parameters), multiple annotations may all want to write to the same file.
  /// This always tracks the actual annotation that will be used.  If a more
  /// appropriate annotation is found (e.g., which will cause the file to be
  /// written to the DUT directory and not the TestHarness directory), then this
  /// will map will be updated.
  llvm::MapVector<StringAttr, AnnotationInfo> emittedFileMap;
};
} // end anonymous namespace

/// Emit the annotated source code for black boxes in a circuit.
void BlackBoxReaderPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");
  CircuitOp circuitOp = getOperation();
  CircuitNamespace ns(circuitOp);

  instanceGraph = &getAnalysis<InstanceGraph>();
  instanceInfo = &getAnalysis<InstanceInfo>();
  auto context = &getContext();

  // If this pass has changed anything.
  bool anythingChanged = false;

  // Internalize some string attributes for easy reference later.

  // Determine the target directory and resource file name from the
  // annotations present on the circuit operation.
  targetDir = ".";

  // Process black box annotations on the circuit.  Some of these annotations
  // will affect how the rest of the annotations are resolved.
  SmallVector<Attribute, 4> filteredAnnos;
  for (auto annot : AnnotationSet(circuitOp)) {
    filteredAnnos.push_back(annot.getDict());

    // Get the testbench and cover directories.
    if (annot.isClass(extractCoverageAnnoClass))
      if (auto dir = annot.getMember<StringAttr>("directory")) {
        coverDir = dir.getValue();
        continue;
      }

    if (annot.isClass(testBenchDirAnnoClass))
      if (auto dir = annot.getMember<StringAttr>("dirname")) {
        testBenchDir = dir.getValue();
        continue;
      }

    // Handle target dir annotation.
    if (annot.isClass(blackBoxTargetDirAnnoClass)) {
      if (auto target = annot.getMember<StringAttr>("targetDir")) {
        targetDir = target.getValue();
        continue;
      }
      circuitOp->emitError(blackBoxTargetDirAnnoClass)
          << " annotation missing \"targetDir\" attribute";
      signalPassFailure();
      continue;
    }
  }
  // Apply the filtered annotations to the circuit.  If we updated the circuit
  // and record that they changed.
  anythingChanged |=
      AnnotationSet(filteredAnnos, context).applyToOperation(circuitOp);

  LLVM_DEBUG(llvm::dbgs() << "Black box target directory: " << targetDir
                          << "\n");

  // Newly generated IR will be placed at the end of the circuit.
  auto builder = circuitOp.getBodyBuilder();

  LLVM_DEBUG(llvm::dbgs() << "Visiting extmodules:\n");
  auto bboxAnno =
      builder.getDictionaryAttr({{builder.getStringAttr("class"),
                                  builder.getStringAttr(blackBoxAnnoClass)}});
  for (auto extmoduleOp : circuitOp.getBodyBlock()->getOps<FExtModuleOp>()) {
    LLVM_DEBUG({
      llvm::dbgs().indent(2)
          << "- name: " << extmoduleOp.getModuleNameAttr() << "\n";
      llvm::dbgs().indent(4) << "annotations:\n";
    });
    AnnotationSet annotations(extmoduleOp);
    bool isCover =
        !coverDir.empty() && annotations.hasAnnotation(verifBlackBoxAnnoClass);
    bool foundBBoxAnno = false;
    annotations.removeAnnotations([&](Annotation anno) {
      AnnotationInfo annotationInfo;
      if (!runOnAnnotation(extmoduleOp, anno, builder, isCover, annotationInfo))
        return false;

      LLVM_DEBUG(annotationInfo.print(llvm::dbgs().indent(6) << "- ", 8));

      // If we have seen a black box trying to create a blackbox with this
      // filename before, then compute the lowest commmon ancestor between the
      // two blackbox paths.  This is the same logic used in `AssignOutputDirs`.
      // However, this needs to incorporate filenames that are only available
      // _after_ output directories are assigned.
      auto [ptr, inserted] =
          emittedFileMap.try_emplace(annotationInfo.name, annotationInfo);
      if (inserted) {
        emittedFileMap[annotationInfo.name] = annotationInfo;
      } else {
        auto &fileAttr = ptr->second.outputFileAttr;
        SmallString<64> directory(fileAttr.getDirectory());
        makeCommonPrefix(directory,
                         annotationInfo.outputFileAttr.getDirectory());
        fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
            context, directory, annotationInfo.name.getValue(),
            /*excludeFromFileList=*/
            fileAttr.getExcludeFromFilelist().getValue());
        // TODO: Check that the new text is the _exact same_ as the prior best.
      }

      foundBBoxAnno = true;
      return true;
    });

    if (foundBBoxAnno) {
      annotations.addAnnotations({bboxAnno});
      anythingChanged = true;
    }
    annotations.applyToOperation(extmoduleOp);
  }

  LLVM_DEBUG(llvm::dbgs() << "emittedFiles:\n");
  Location loc = builder.getUnknownLoc();
  for (auto &[verilogName, annotationInfo] : emittedFileMap) {
    LLVM_DEBUG({
      llvm::dbgs().indent(2) << "verilogName: " << verilogName << "\n";
      llvm::dbgs().indent(2) << "annotationInfo:\n";
      annotationInfo.print(llvm::dbgs().indent(4) << "- ", 6);
    });

    auto fileName = ns.newName("blackbox_" + verilogName.getValue());

    auto fileOp = builder.create<emit::FileOp>(
        loc, annotationInfo.outputFileAttr.getFilename(), fileName,
        [&, text = annotationInfo.inlineText] {
          builder.create<emit::VerbatimOp>(loc, text);
        });

    if (!annotationInfo.outputFileAttr.getExcludeFromFilelist().getValue())
      fileListFiles.push_back(fileOp);
  }

  // If we have emitted any files, generate a file list operation that
  // documents the additional annotation-controlled file listing to be
  // created.
  if (!fileListFiles.empty()) {
    // Output the file list in sorted order.
    llvm::sort(fileListFiles.begin(), fileListFiles.end(),
               [](emit::FileOp fileA, emit::FileOp fileB) {
                 return fileA.getFileName() < fileB.getFileName();
               });

    // Create the file list contents by enumerating the symbols to the files.
    SmallVector<Attribute> symbols;
    for (emit::FileOp file : fileListFiles)
      symbols.push_back(FlatSymbolRefAttr::get(file.getSymNameAttr()));
  }

  // If nothing has changed we can preserve the analysis.
  if (!anythingChanged)
    markAllAnalysesPreserved();
  markAnalysesPreserved<InstanceGraph, InstanceInfo>();

  // Clean up.
  emittedFileMap.clear();
  fileListFiles.clear();
  LLVM_DEBUG(debugFooter() << "\n");
}

/// Run on an operation-annotation pair. The annotation need not be a black box
/// annotation. Returns `true` if the annotation was indeed a black box
/// annotation (even if it was incomplete) and should be removed from the op.
bool BlackBoxReaderPass::runOnAnnotation(Operation *op, Annotation anno,
                                         OpBuilder &builder, bool isCover,
                                         AnnotationInfo &annotationInfo) {
  // Handle inline annotation.
  if (anno.isClass(blackBoxInlineAnnoClass)) {
    auto name = anno.getMember<StringAttr>("name");
    auto text = anno.getMember<StringAttr>("text");
    if (!name || !text) {
      op->emitError(blackBoxInlineAnnoClass)
          << " annotation missing \"name\" or \"text\" attribute";
      signalPassFailure();
      return true;
    }

    annotationInfo.outputFileAttr = getOutputFile(op, name, isCover);
    annotationInfo.name = name;
    annotationInfo.inlineText = text;
    return true;
  }

  // Handle path annotation.
  if (anno.isClass(blackBoxPathAnnoClass)) {
    auto path = anno.getMember<StringAttr>("path");
    if (!path) {
      op->emitError(blackBoxPathAnnoClass)
          << " annotation missing \"path\" attribute";
      signalPassFailure();
      return true;
    }
    SmallString<128> inputPath(inputPrefix);
    appendPossiblyAbsolutePath(inputPath, path.getValue());
    auto text = loadFile(op, inputPath, builder);
    if (!text) {
      op->emitError("Cannot find file ") << inputPath;
      signalPassFailure();
      return false;
    }
    auto name = builder.getStringAttr(llvm::sys::path::filename(path));
    annotationInfo.outputFileAttr = getOutputFile(op, name, isCover);
    annotationInfo.name = name;
    annotationInfo.inlineText = text;
    return true;
  }

  // Annotation was not concerned with black boxes.
  return false;
}

/// Copies a black box source file to the appropriate location in the target
/// directory.
StringAttr BlackBoxReaderPass::loadFile(Operation *op, StringRef inputPath,
                                        OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs() << "Add black box source  `"
                          << llvm::sys::path::filename(inputPath) << "` from `"
                          << inputPath << "`\n");

  // Open and read the input file.
  std::string errorMessage;
  auto input = mlir::openInputFile(inputPath, &errorMessage);
  if (!input)
    return {};

  // Return a StringAttr with the buffer contents.
  return builder.getStringAttr(input->getBuffer());
}

/// Determine the output file for some operation.
hw::OutputFileAttr BlackBoxReaderPass::getOutputFile(Operation *origOp,
                                                     StringAttr fileNameAttr,
                                                     bool isCover) {
  // If the original operation has a specified output file that is not a
  // directory, then just use that.
  auto outputFile = origOp->getAttrOfType<hw::OutputFileAttr>("output_file");
  if (outputFile && !outputFile.isDirectory()) {
    return {outputFile};
  }

  // Exclude Verilog header files since we expect them to be included
  // explicitly by compiler directives in other source files.
  auto *context = &getContext();
  auto fileName = fileNameAttr.getValue();
  auto ext = llvm::sys::path::extension(fileName);
  bool exclude = (ext == ".h" || ext == ".vh" || ext == ".svh");
  auto outDir = targetDir;

  /// If the original operation has an output directory, use that.
  if (outputFile)
    outDir = outputFile.getFilename();
  // In order to output into the testbench directory, we need to have a
  // testbench dir annotation, not have a blackbox target directory annotation
  // (or one set to the current directory), have a DUT annotation, and the
  // module must not be instantiated under the DUT.
  else if (!testBenchDir.empty() && targetDir == "." &&
           !instanceInfo->anyInstanceInEffectiveDesign(
               cast<igraph::ModuleOpInterface>(origOp)))
    outDir = testBenchDir;
  else if (isCover)
    outDir = coverDir;

  // If targetDir is not set explicitly and this is a testbench module, then
  // update the targetDir to be the "../testbench".
  SmallString<128> outputFilePath(outDir);
  llvm::sys::path::append(outputFilePath, fileName);
  return hw::OutputFileAttr::getFromFilename(context, outputFilePath, exclude);
}
