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
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Path.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "firrtl-blackbox-reader"

using namespace circt;
using namespace firrtl;

using hw::OutputFileAttr;
using sv::VerbatimOp;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

/// This is used to indicate the directory priority.  Multiple external modules
/// with the same "defname" may have different output filenames.  This is used
/// to choose the best filename.
enum class Priority { TargetDir = 0, Verification, Explicit, TestBench, Unset };

/// Data extracted from BlackBoxInlineAnno or BlackBoxPathAnno.
struct AnnotationInfo {
  /// The name of the file that should be created for this BlackBox.
  StringAttr name;
  /// The output directory where this annotation should be written.
  OutputFileAttr outputFile;
  /// The body of the BlackBox.  (This should be Verilog text.)
  StringAttr inlineText;
  /// The priority of this annotation.  In the even that multiple annotations
  /// are provided for the same BlackBox, then use this as a tie-breaker if an
  /// external module is instantiated multiple times and those multiple
  /// instantiations disagree on where the module should go.
  Priority priority = Priority::Unset;

#if !defined(NDEBUG)
  /// Pretty print the AnnotationInfo in a YAML-esque format.
  void print(raw_ostream &os, unsigned indent = 0) const {
    if (priority == Priority::Unset) {
      os << "<null>\n";
      return;
    }
    os << llvm::formatv("name: {1}\n"
                        "{0}outputFile: {2}\n"
                        "{0}priority: {3}\n",
                        llvm::fmt_pad("", indent, 0), name,
                        outputFile.getFilename(), (unsigned)priority);
  };
#endif
};

struct BlackBoxReaderPass : public BlackBoxReaderBase<BlackBoxReaderPass> {
  void runOnOperation() override;
  bool runOnAnnotation(Operation *op, Annotation anno, OpBuilder &builder,
                       bool isCover, AnnotationInfo &annotationInfo);
  StringAttr loadFile(Operation *op, StringRef inputPath, OpBuilder &builder);
  std::pair<OutputFileAttr, Priority> getOutputFile(Operation *origOp,
                                                    StringAttr fileNameAttr,
                                                    bool isCover = false);
  void setOutputFile(VerbatimOp op, OutputFileAttr outputFile,
                     StringAttr fileNameAttr);
  // Check if module or any of its parents in the InstanceGraph is a DUT.
  bool isDut(Operation *module);

  using BlackBoxReaderBase::inputPrefix;

private:
  /// A list of all files which will be included in the file list.  This is
  /// subset of all emitted files.
  SmallVector<StringRef> fileListFiles;

  /// The target directory to output black boxes into. Can be changed
  /// through `firrtl.transforms.BlackBoxTargetDirAnno` annotations.
  StringRef targetDir;

  /// The target directory for cover statements.
  StringRef coverDir;

  /// The target directory for testbench files.
  StringRef testBenchDir;

  /// The design-under-test (DUT) as indicated by the presence of a
  /// "sifive.enterprise.firrtl.MarkDUTAnnotation".  This will be null if no
  /// annotation is present.
  FModuleOp dut;

  /// The file list file name (sic) for black boxes. If set, generates a file
  /// that lists all non-header source files for black boxes. Can be changed
  /// through `firrtl.transforms.BlackBoxResourceFileNameAnno` annotations.
  StringRef resourceFileName;

  /// InstanceGraph to determine modules which are under the DUT.
  InstanceGraph *instanceGraph;

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
  llvm::MapVector<Attribute, AnnotationInfo> emittedFileMap;
};
} // end anonymous namespace

/// Emit the annotated source code for black boxes in a circuit.
void BlackBoxReaderPass::runOnOperation() {
  CircuitOp circuitOp = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();
  auto context = &getContext();

  // If this pass has changed anything.
  bool anythingChanged = false;

  // Internalize some string attributes for easy reference later.

  // Determine the target directory and resource file name from the
  // annotations present on the circuit operation.
  targetDir = ".";
  resourceFileName = "firrtl_black_box_resource_files.f";

  // Process black box annotations on the circuit.  Some of these annotations
  // will affect how the rest of the annotations are resolved.
  SmallVector<Attribute, 4> filteredAnnos;
  for (auto annot : AnnotationSet(circuitOp)) {
    // Handle resource file name annotation.
    if (annot.isClass(blackBoxResourceFileNameAnnoClass)) {
      if (auto resourceFN = annot.getMember<StringAttr>("resourceFileName")) {
        resourceFileName = resourceFN.getValue();
        continue;
      }

      circuitOp->emitError(blackBoxResourceFileNameAnnoClass)
          << " annotation missing \"resourceFileName\" attribute";
      signalPassFailure();
      continue;
    }
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

  LLVM_DEBUG(llvm::dbgs() << "Black box target directory: " << targetDir << "\n"
                          << "Black box resource file name: "
                          << resourceFileName << "\n");

  // Newly generated IR will be placed at the end of the circuit.
  auto builder = circuitOp.getBodyBuilder();

  // Do a shallow walk of the circuit to collect information necessary before we
  // do real work.
  for (auto &op : *circuitOp.getBodyBlock()) {
    FModuleOp module = dyn_cast<FModuleOp>(op);
    // Find the DUT if it exists or error if there are multiple DUTs.
    if (module)
      if (failed(extractDUT(module, dut)))
        return signalPassFailure();
  }

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

      auto &bestAnnotationInfo = emittedFileMap[annotationInfo.name];
      if (annotationInfo.priority < bestAnnotationInfo.priority) {
        bestAnnotationInfo = annotationInfo;

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
  for (auto &[verilogName, annotationInfo] : emittedFileMap) {
    LLVM_DEBUG({
      llvm::dbgs().indent(2) << "verilogName: " << verilogName << "\n";
      llvm::dbgs().indent(2) << "annotationInfo:\n";
      annotationInfo.print(llvm::dbgs().indent(4) << "- ", 6);
    });

    auto verbatim = builder.create<VerbatimOp>(builder.getUnknownLoc(),
                                               annotationInfo.inlineText);
    setOutputFile(verbatim, annotationInfo.outputFile, annotationInfo.name);
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
          llvm::sys::path::remove_dots(filePath);
          os << filePath;
        },
        "\n");

    // Put the file list in to a verbatim op.  Use "unknown location" so that no
    // file info will unnecessarily print.
    auto op =
        builder.create<VerbatimOp>(builder.getUnknownLoc(), std::move(output));

    // Attach the output file information to the verbatim op.
    op->setAttr("output_file", hw::OutputFileAttr::getFromFilename(
                                   context, resourceFileName,
                                   /*excludeFromFileList=*/true));
  }

  // If nothing has changed we can preserve the analysis.
  if (!anythingChanged)
    markAllAnalysesPreserved();
  markAnalysesPreserved<InstanceGraph>();

  // Clean up.
  emittedFileMap.clear();
  fileListFiles.clear();
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

    auto outputFile = getOutputFile(op, name, isCover);
    annotationInfo.outputFile = outputFile.first;
    annotationInfo.name = name;
    annotationInfo.inlineText = text;
    annotationInfo.priority = outputFile.second;
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
    auto outputFile = getOutputFile(op, name, isCover);
    annotationInfo.outputFile = outputFile.first;
    annotationInfo.name = name;
    annotationInfo.inlineText = text;
    annotationInfo.priority = outputFile.second;
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
std::pair<OutputFileAttr, Priority>
BlackBoxReaderPass::getOutputFile(Operation *origOp, StringAttr fileNameAttr,
                                  bool isCover) {
  // If the original operation has a specified output file that is not a
  // directory, then just use that.
  auto outputFile = origOp->getAttrOfType<OutputFileAttr>("output_file");
  if (outputFile && !outputFile.isDirectory())
    return {outputFile, Priority::TargetDir};

  // Exclude Verilog header files since we expect them to be included
  // explicitly by compiler directives in other source files.
  auto *context = &getContext();
  auto fileName = fileNameAttr.getValue();
  auto ext = llvm::sys::path::extension(fileName);
  bool exclude = (ext == ".h" || ext == ".vh" || ext == ".svh");
  auto outDir = std::make_pair(targetDir, Priority::TargetDir);
  // In order to output into the testbench directory, we need to have a
  // testbench dir annotation, not have a blackbox target directory annotation
  // (or one set to the current directory), have a DUT annotation, and the
  // module needs to be in or under the DUT.
  if (!testBenchDir.empty() && targetDir.equals(".") && dut && !isDut(origOp))
    outDir = {testBenchDir, Priority::TestBench};
  else if (isCover)
    outDir = {coverDir, Priority::Verification};
  else if (outputFile)
    outDir = {outputFile.getFilename(), Priority::Explicit};

  // If targetDir is not set explicitly and this is a testbench module, then
  // update the targetDir to be the "../testbench".
  auto outFileAttr = OutputFileAttr::getFromDirectoryAndFilename(
      context, outDir.first, fileName,
      /*excludeFromFileList=*/exclude);
  return {outFileAttr, outDir.second};
}

/// This function is called for every file generated.  It does the following
/// things:
///  1. Attaches the output file attribute to the VerbatimOp.
///  2. Record that the file has been generated to avoid duplicates.
///  3. Add each file name to the generated "file list" file.
void BlackBoxReaderPass::setOutputFile(VerbatimOp op, OutputFileAttr outputFile,
                                       StringAttr fileNameAttr) {
  op->setAttr("output_file", outputFile);

  // Append this file to the file list if its not excluded.
  if (!outputFile.getExcludeFromFilelist().getValue())
    fileListFiles.push_back(outputFile.getFilename());
}

/// Return true if module is in the DUT hierarchy.
/// NOLINTNEXTLINE(misc-no-recursion)
bool BlackBoxReaderPass::isDut(Operation *module) {
  // Check if result already cached.
  auto iter = dutModuleMap.find(module);
  if (iter != dutModuleMap.end())
    return iter->getSecond();
  AnnotationSet annos(module);
  // Any module with the dutAnno, is the DUT.
  if (annos.hasAnnotation(dutAnnoClass)) {
    dutModuleMap[module] = true;
    return true;
  }
  auto *node = instanceGraph->lookup(cast<igraph::ModuleOpInterface>(module));
  bool anyParentIsDut = false;
  if (node)
    for (auto *u : node->uses()) {
      // Recursively check the parents.
      auto dut = isDut(u->getInstance()->getParentOfType<FModuleOp>());
      // Cache the result.
      dutModuleMap[module] = dut;
      anyParentIsDut |= dut;
    }
  dutModuleMap[module] = anyParentIsDut;
  return anyParentIsDut;
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
circt::firrtl::createBlackBoxReaderPass(std::optional<StringRef> inputPrefix) {
  auto pass = std::make_unique<BlackBoxReaderPass>();
  if (inputPrefix)
    pass->inputPrefix = inputPrefix->str();
  return pass;
}
