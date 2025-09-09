//===- Firtool.h - Definitions for the firtool pipeline setup ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This library parses options for firtool and sets up its pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_FIRTOOL_FIRTOOL_H
#define CIRCT_FIRTOOL_FIRTOOL_H

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace circt {
namespace firtool {
//===----------------------------------------------------------------------===//
// FirtoolOptions
//===----------------------------------------------------------------------===//

/// Set of options used to control the behavior of the firtool pipeline.
class FirtoolOptions {
public:
  FirtoolOptions();

  // Helper Types
  enum BuildMode { BuildModeDefault, BuildModeDebug, BuildModeRelease };
  enum class RandomKind { None, Mem, Reg, All };

  bool isRandomEnabled(RandomKind kind) const {
    return disableRandom != RandomKind::All && disableRandom != kind;
  }

  firrtl::PreserveValues::PreserveMode getPreserveMode() const {
    switch (buildMode) {
    case BuildModeDebug:
      return firrtl::PreserveValues::Named;
    case BuildModeRelease:
      return firrtl::PreserveValues::None;
    case BuildModeDefault:
      return preserveMode;
    }
    llvm_unreachable("unknown build mode");
  }

  StringRef getOutputFilename() const { return outputFilename; }
  StringRef getBlackBoxRootPath() const { return blackBoxRootPath; }
  StringRef getReplaceSequentialMemoriesFile() const { return replSeqMemFile; }
  StringRef getOutputAnnotationFilename() const {
    return outputAnnotationFilename;
  }

  firrtl::PreserveAggregate::PreserveMode getPreserveAggregate() const {
    return preserveAggregate;
  }
  firrtl::CompanionMode getCompanionMode() const { return companionMode; }

  seq::ExternalizeClockGateOptions getClockGateOptions() const {
    return {ckgModuleName, ckgInputName,      ckgOutputName,
            ckgEnableName, ckgTestEnableName, ckgInstName};
  }

  FirtoolOptions &setClockGateOptions(seq::ExternalizeClockGateOptions &opts) {
    ckgModuleName = opts.moduleName;
    ckgInputName = opts.inputName;
    ckgOutputName = opts.outputName;
    ckgEnableName = opts.enableName;
    ckgTestEnableName = opts.testEnableName;
    ckgInstName = opts.instName;
    return *this;
  }

  bool isDefaultOutputFilename() const { return outputFilename == "-"; }
  bool shouldDisableUnknownAnnotations() const {
    return disableAnnotationsUnknown;
  }
  bool shouldDisableClasslessAnnotations() const {
    return disableAnnotationsClassless;
  }
  bool shouldLowerNoRefTypePortAnnotations() const {
    return lowerAnnotationsNoRefTypePorts;
  }
  bool shouldAllowAddingPortsOnPublic() const {
    return allowAddingPortsOnPublic;
  }
  bool shouldConvertProbesToSignals() const { return probesToSignals; }
  bool shouldReplaceSequentialMemories() const { return replSeqMem; }
  bool shouldDisableLayerSink() const { return disableLayerSink; }
  bool shouldDisableOptimization() const { return disableOptimization; }
  bool shouldLowerMemories() const { return lowerMemories; }
  bool shouldDedup() const { return !noDedup; }
  bool shouldEnableDebugInfo() const { return enableDebugInfo; }
  bool shouldIgnoreReadEnableMemories() const { return ignoreReadEnableMem; }
  bool shouldConvertVecOfBundle() const { return vbToBV; }
  bool shouldEtcDisableInstanceExtraction() const {
    return etcDisableInstanceExtraction;
  }
  bool shouldEtcDisableRegisterExtraction() const {
    return etcDisableRegisterExtraction;
  }
  bool shouldEtcDisableModuleInlining() const {
    return etcDisableModuleInlining;
  }
  bool shouldStripDebugInfo() const { return stripDebugInfo; }
  bool shouldStripFirDebugInfo() const { return stripFirDebugInfo; }
  bool shouldExportModuleHierarchy() const { return exportModuleHierarchy; }
  bool shouldDisableAggressiveMergeConnections() const {
    return disableAggressiveMergeConnections;
  }
  bool shouldEnableAnnotationWarning() const { return enableAnnotationWarning; }
  auto getVerificationFlavor() const { return verificationFlavor; }
  bool shouldEmitSeparateAlwaysBlocks() const {
    return emitSeparateAlwaysBlocks;
  }
  bool shouldAddMuxPragmas() const { return addMuxPragmas; }
  bool shouldAddVivadoRAMAddressConflictSynthesisBugWorkaround() const {
    return addVivadoRAMAddressConflictSynthesisBugWorkaround;
  }
  bool shouldExtractTestCode() const { return extractTestCode; }
  bool shouldFixupEICGWrapper() const { return fixupEICGWrapper; }
  bool shouldDisableCSEinClasses() const { return disableCSEinClasses; }
  bool shouldSelectDefaultInstanceChoice() const {
    return selectDefaultInstanceChoice;
  }

  verif::SymbolicValueLowering getSymbolicValueLowering() const {
    return symbolicValueLowering;
  }
  bool shouldDisableWireElimination() const { return disableWireElimination; }

  bool getLintStaticAsserts() const { return lintStaticAsserts; }

  bool getLintXmrsInDesign() const { return lintXmrsInDesign; }

  bool getEmitAllBindFiles() const { return emitAllBindFiles; }

  // Setters, used by the CAPI
  FirtoolOptions &setOutputFilename(StringRef name) {
    outputFilename = name;
    return *this;
  }

  FirtoolOptions &setDisableUnknownAnnotations(bool disable) {
    disableAnnotationsUnknown = disable;
    return *this;
  }

  FirtoolOptions &setDisableAnnotationsClassless(bool value) {
    disableAnnotationsClassless = value;
    return *this;
  }

  FirtoolOptions &setLowerAnnotationsNoRefTypePorts(bool value) {
    lowerAnnotationsNoRefTypePorts = value;
    return *this;
  }

  FirtoolOptions &setAllowAddingPortsOnPublic(bool value) {
    allowAddingPortsOnPublic = value;
    return *this;
  }

  FirtoolOptions &setConvertProbesToSignals(bool value) {
    probesToSignals = value;
    return *this;
  }

  FirtoolOptions &
  setPreserveAggregate(firrtl::PreserveAggregate::PreserveMode value) {
    preserveAggregate = value;
    return *this;
  }

  FirtoolOptions &
  setPreserveValues(firrtl::PreserveValues::PreserveMode value) {
    preserveMode = value;
    return *this;
  }

  FirtoolOptions &setEnableDebugInfo(bool value) {
    enableDebugInfo = value;
    return *this;
  }

  FirtoolOptions &setBuildMode(BuildMode value) {
    buildMode = value;
    return *this;
  }

  FirtoolOptions &setDisableLayerSink(bool value) {
    disableLayerSink = value;
    return *this;
  }

  FirtoolOptions &setDisableOptimization(bool value) {
    disableOptimization = value;
    return *this;
  }

  FirtoolOptions &setVbToBV(bool value) {
    vbToBV = value;
    return *this;
  }

  FirtoolOptions &setNoDedup(bool value) {
    noDedup = value;
    return *this;
  }

  FirtoolOptions &setCompanionMode(firrtl::CompanionMode value) {
    companionMode = value;
    return *this;
  }

  FirtoolOptions &setDisableAggressiveMergeConnections(bool value) {
    disableAggressiveMergeConnections = value;
    return *this;
  }

  FirtoolOptions &setLowerMemories(bool value) {
    lowerMemories = value;
    return *this;
  }

  FirtoolOptions &setBlackBoxRootPath(StringRef value) {
    blackBoxRootPath = value;
    return *this;
  }

  FirtoolOptions &setReplSeqMem(bool value) {
    replSeqMem = value;
    return *this;
  }

  FirtoolOptions &setReplSeqMemFile(StringRef value) {
    replSeqMemFile = value;
    return *this;
  }

  FirtoolOptions &setExtractTestCode(bool value) {
    extractTestCode = value;
    return *this;
  }

  FirtoolOptions &setIgnoreReadEnableMem(bool value) {
    ignoreReadEnableMem = value;
    return *this;
  }

  FirtoolOptions &setDisableRandom(RandomKind value) {
    disableRandom = value;
    return *this;
  }

  FirtoolOptions &setOutputAnnotationFilename(StringRef value) {
    outputAnnotationFilename = value;
    return *this;
  }

  FirtoolOptions &setEnableAnnotationWarning(bool value) {
    enableAnnotationWarning = value;
    return *this;
  }

  FirtoolOptions &setAddMuxPragmas(bool value) {
    addMuxPragmas = value;
    return *this;
  }

  FirtoolOptions &setVerificationFlavor(firrtl::VerificationFlavor value) {
    verificationFlavor = value;
    return *this;
  }

  FirtoolOptions &setEmitSeparateAlwaysBlocks(bool value) {
    emitSeparateAlwaysBlocks = value;
    return *this;
  }

  FirtoolOptions &setEtcDisableInstanceExtraction(bool value) {
    etcDisableInstanceExtraction = value;
    return *this;
  }

  FirtoolOptions &setEtcDisableRegisterExtraction(bool value) {
    etcDisableRegisterExtraction = value;
    return *this;
  }

  FirtoolOptions &setEtcDisableModuleInlining(bool value) {
    etcDisableModuleInlining = value;
    return *this;
  }

  FirtoolOptions &
  setAddVivadoRAMAddressConflictSynthesisBugWorkaround(bool value) {
    addVivadoRAMAddressConflictSynthesisBugWorkaround = value;
    return *this;
  }

  FirtoolOptions &setCkgModuleName(StringRef value) {
    ckgModuleName = value;
    return *this;
  }

  FirtoolOptions &setCkgInputName(StringRef value) {
    ckgInputName = value;
    return *this;
  }

  FirtoolOptions &setCkgOutputName(StringRef value) {
    ckgOutputName = value;
    return *this;
  }

  FirtoolOptions &setCkgEnableName(StringRef value) {
    ckgEnableName = value;
    return *this;
  }

  FirtoolOptions &setCkgTestEnableName(StringRef value) {
    ckgTestEnableName = value;
    return *this;
  }

  FirtoolOptions &setExportModuleHierarchy(bool value) {
    exportModuleHierarchy = value;
    return *this;
  }

  FirtoolOptions &setStripFirDebugInfo(bool value) {
    stripFirDebugInfo = value;
    return *this;
  }

  FirtoolOptions &setStripDebugInfo(bool value) {
    stripDebugInfo = value;
    return *this;
  }

  FirtoolOptions &setFixupEICGWrapper(bool value) {
    fixupEICGWrapper = value;
    return *this;
  }

  FirtoolOptions &setDisableCSEinClasses(bool value) {
    disableCSEinClasses = value;
    return *this;
  }

  FirtoolOptions &setSelectDefaultInstanceChoice(bool value) {
    selectDefaultInstanceChoice = value;
    return *this;
  }

  FirtoolOptions &setSymbolicValueLowering(verif::SymbolicValueLowering mode) {
    symbolicValueLowering = mode;
    return *this;
  }

  FirtoolOptions &setDisableWireElimination(bool value) {
    disableWireElimination = value;
    return *this;
  }

  FirtoolOptions &setLintStaticAsserts(bool value) {
    lintStaticAsserts = value;
    return *this;
  }

  FirtoolOptions &setLintXmrsInDesign(bool value) {
    lintXmrsInDesign = value;
    return *this;
  }

  FirtoolOptions &setEmitAllBindFiles(bool value) {
    emitAllBindFiles = value;
    return *this;
  }

private:
  std::string outputFilename;

  // LowerFIRRTLAnnotations
  bool disableAnnotationsUnknown;
  bool disableAnnotationsClassless;
  bool lowerAnnotationsNoRefTypePorts;
  bool allowAddingPortsOnPublic;

  bool probesToSignals;
  firrtl::PreserveAggregate::PreserveMode preserveAggregate;
  firrtl::PreserveValues::PreserveMode preserveMode;
  bool enableDebugInfo;
  BuildMode buildMode;
  bool disableLayerSink;
  bool disableOptimization;
  bool vbToBV;
  bool noDedup;
  firrtl::CompanionMode companionMode;
  bool disableAggressiveMergeConnections;
  bool lowerMemories;
  std::string blackBoxRootPath;
  bool replSeqMem;
  std::string replSeqMemFile;
  bool extractTestCode;
  bool ignoreReadEnableMem;
  RandomKind disableRandom;
  std::string outputAnnotationFilename;
  bool enableAnnotationWarning;
  bool addMuxPragmas;
  firrtl::VerificationFlavor verificationFlavor;
  bool emitSeparateAlwaysBlocks;
  bool etcDisableInstanceExtraction;
  bool etcDisableRegisterExtraction;
  bool etcDisableModuleInlining;
  bool addVivadoRAMAddressConflictSynthesisBugWorkaround;
  std::string ckgModuleName;
  std::string ckgInputName;
  std::string ckgOutputName;
  std::string ckgEnableName;
  std::string ckgTestEnableName;
  std::string ckgInstName;
  bool exportModuleHierarchy;
  bool stripFirDebugInfo;
  bool stripDebugInfo;
  bool fixupEICGWrapper;
  bool disableCSEinClasses;
  bool selectDefaultInstanceChoice;
  verif::SymbolicValueLowering symbolicValueLowering;
  bool disableWireElimination;
  bool lintStaticAsserts;
  bool lintXmrsInDesign;
  bool emitAllBindFiles;
};

void registerFirtoolCLOptions();

LogicalResult populatePreprocessTransforms(mlir::PassManager &pm,
                                           const FirtoolOptions &opt);

LogicalResult populateCHIRRTLToLowFIRRTL(mlir::PassManager &pm,
                                         const FirtoolOptions &opt);

LogicalResult populateLowFIRRTLToHW(mlir::PassManager &pm,
                                    const FirtoolOptions &opt,
                                    StringRef inputFilename);

LogicalResult populateHWToSV(mlir::PassManager &pm, const FirtoolOptions &opt);

LogicalResult populateExportVerilog(mlir::PassManager &pm,
                                    const FirtoolOptions &opt,
                                    std::unique_ptr<llvm::raw_ostream> os);

LogicalResult populateExportVerilog(mlir::PassManager &pm,
                                    const FirtoolOptions &opt,
                                    llvm::raw_ostream &os);

LogicalResult populateExportSplitVerilog(mlir::PassManager &pm,
                                         const FirtoolOptions &opt,
                                         llvm::StringRef directory);

LogicalResult populateFinalizeIR(mlir::PassManager &pm,
                                 const FirtoolOptions &opt);

LogicalResult populateHWToBTOR2(mlir::PassManager &pm,
                                const FirtoolOptions &opt,
                                llvm::raw_ostream &os);

} // namespace firtool
} // namespace circt

#endif // CIRCT_FIRTOOL_FIRTOOL_H
