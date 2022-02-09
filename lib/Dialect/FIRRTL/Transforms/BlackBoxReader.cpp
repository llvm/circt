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
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Path.h"
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

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct BlackBoxReaderPass : public BlackBoxReaderBase<BlackBoxReaderPass> {
  void runOnOperation() override;
  bool runOnAnnotation(Operation *op, Annotation anno, OpBuilder &builder,
                       bool isCover);
  bool loadFile(Operation *op, StringRef inputPath, OpBuilder &builder);
  void setOutputFile(VerbatimOp op, StringAttr fileNameAttr, bool isDut = false,
                     bool isCover = false);
  // Check if module or any of its parents in the InstanceGraph is a DUT.
  bool isDut(Operation *module);

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

  /// The target directory for cover statements.
  StringRef coverDir;

  /// The target directory for testbench files.
  StringRef testBenchDir;

  /// The file list file name (sic) for black boxes. If set, generates a file
  /// that lists all non-header source files for black boxes. Can be changed
  /// through `firrtl.transforms.BlackBoxResourceFileNameAnno` annotations.
  StringRef resourceFileName;

  /// InstanceGraph to determine modules which are under the DUT.
  InstanceGraph *instanceGraph;

  /// A cache of the the modules which have been marked as DUT or a testbench.
  /// This is used to determine the output directory.
  DenseMap<Operation *, bool> dutModuleMap;
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
  StringRef targetDirAnnoClass = "firrtl.transforms.BlackBoxTargetDirAnno";
  StringRef resourceFileNameAnnoClass =
      "firrtl.transforms.BlackBoxResourceFileNameAnno";
  StringRef coverAnnoClass =
      "sifive.enterprise.firrtl.ExtractCoverageAnnotation";
  StringRef testbenchAnno = "sifive.enterprise.firrtl.TestBenchDirAnnotation";

  // Determine the target directory and resource file name from the
  // annotations present on the circuit operation.
  targetDir = ".";
  resourceFileName = "firrtl_black_box_resource_files.f";

  // Process black box annotations on the circuit.  Some of these annotations
  // will affect how the rest of the annotations are resolved.
  SmallVector<Attribute, 4> filteredAnnos;
  for (auto annot : AnnotationSet(circuitOp)) {
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

    // Get the testbench and cover directories.
    if (annot.isClass(coverAnnoClass))
      if (auto dir = annot.getMember<StringAttr>("directory")) {
        coverDir = dir.getValue();
        continue;
      }

    if (annot.isClass(testbenchAnno))
      if (auto dir = annot.getMember<StringAttr>("dirname")) {
        testBenchDir = dir.getValue();
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
  auto builder = OpBuilder::atBlockEnd(circuitOp->getBlock());

  // Gather the relevant annotations on all modules in the circuit.
  for (auto &op : *circuitOp.getBody()) {
    if (!isa<FModuleOp>(op) && !isa<FExtModuleOp>(op))
      continue;

    StringRef verifBBClass =
        "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation";
    SmallVector<Attribute, 4> filteredAnnos;
    auto annos = AnnotationSet(&op);
    // If the cover directory is set and it has the verifBBClass annotation,
    // then output directory should be cover dir.
    auto isCover = !coverDir.empty() && annos.hasAnnotation(verifBBClass);
    for (auto anno : annos) {
      if (runOnAnnotation(&op, anno, builder, isCover))
        // Since the annotation was consumed, add a `BlackBox` annotation to
        // indicate that this extmodule was provided by one of the black box
        // annotations. This is useful for metadata generation.
        filteredAnnos.push_back(builder.getDictionaryAttr(
            {{builder.getStringAttr("class"),
              builder.getStringAttr("firrtl.transforms.BlackBox")}}));
      else
        filteredAnnos.push_back(anno.getDict());
    }

    // Update the operation annotations to exclude the ones we have consumed.
    anythingChanged |=
        AnnotationSet(filteredAnnos, context).applyToOperation(&op);
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
    op->setAttr("output_file", hw::OutputFileAttr::getFromFilename(
                                   context, resourceFileName,
                                   /*excludeFromFileList=*/true));
  }

  // If nothing has changed we can preseve the analysis.
  if (!anythingChanged)
    markAllAnalysesPreserved();
  markAnalysesPreserved<InstanceGraph>();

  // Clean up.
  emittedFiles.clear();
  fileListFiles.clear();
}

/// Run on an operation-annotation pair. The annotation need not be a black box
/// annotation. Returns `true` if the annotation was indeed a black box
/// annotation (even if it was incomplete) and should be removed from the op.
bool BlackBoxReaderPass::runOnAnnotation(Operation *op, Annotation anno,
                                         OpBuilder &builder, bool isCover) {
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
    setOutputFile(verbatim, name, isDut(op), isCover);
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
void BlackBoxReaderPass::setOutputFile(VerbatimOp op, StringAttr fileNameAttr,
                                       bool isDut, bool isCover) {
  auto *context = &getContext();
  auto fileName = fileNameAttr.getValue();
  // Exclude Verilog header files since we expect them to be included
  // explicitly by compiler directives in other source files.
  auto ext = llvm::sys::path::extension(fileName);
  bool exclude = (ext == ".h" || ext == ".vh" || ext == ".svh");
  auto outDir = targetDir;
  if (!testBenchDir.empty() && targetDir.equals(".") && !isDut)
    outDir = testBenchDir;
  else if (isCover)
    outDir = coverDir;

  // If targetDir is not set explicitly and this is a testbench module, then
  // update the targetDir to be the "../testbench".
  op->setAttr("output_file", OutputFileAttr::getFromDirectoryAndFilename(
                                 context, outDir, fileName,
                                 /*excludeFromFileList=*/exclude));

  // Record that this file has been generated.
  assert(!emittedFiles.count(fileNameAttr) &&
         "Can't generate the same file twice.");
  emittedFiles.insert(fileNameAttr);

  // Append this file to the file list if its not excluded.
  if (!exclude)
    fileListFiles.push_back(fileName);
}

/// Return true if module is in the DUT hierarchy.
bool BlackBoxReaderPass::isDut(Operation *module) {
  // Check if result already cached.
  auto iter = dutModuleMap.find(module);
  if (iter != dutModuleMap.end())
    return iter->getSecond();
  const StringRef dutAnno = "sifive.enterprise.firrtl.MarkDUTAnnotation";
  AnnotationSet annos(module);
  // Any module with the dutAnno, is the DUT.
  if (annos.hasAnnotation(dutAnno)) {
    dutModuleMap[module] = true;
    return true;
  }
  auto *node = instanceGraph->lookup(module);
  bool anyParentIsDut = false;
  if (node)
    for (auto *u : node->uses()) {
      InstanceOp inst = u->getInstance();
      // Recursively check the parents.
      auto dut = isDut(inst->getParentOfType<FModuleOp>());
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
