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

#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/Seq/SeqPasses.h"
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

  static bool isRandomEnabled(RandomKind disabled, RandomKind kind) {
    return disabled != RandomKind::All && disabled != kind;
  }

  bool isRandomEnabled(RandomKind kind) const {
    return isRandomEnabled(disableRandom, kind);
  }

  RandomKind getDisableRandom() const { return disableRandom; }

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
  StringRef getOmirOutputFile() const { return omirOutFile; }
  StringRef getBlackBoxRootPath() const { return blackBoxRootPath; }
  StringRef getChiselInterfaceOutputDirectory() const {
    return chiselInterfaceOutDirectory;
  }
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
            ckgEnableName, ckgTestEnableName, "ckg"};
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
  bool shouldReplicateSequentialMemories() const { return replSeqMem; }
  bool shouldDisableOptimization() const { return disableOptimization; }
  bool shouldLowerMemories() const { return lowerMemories; }
  bool shouldDedup() const { return !noDedup; }
  bool shouldEnableDebugInfo() const { return enableDebugInfo; }
  bool shouldIgnoreReadEnableMemories() const { return ignoreReadEnableMem; }
  bool shouldEmitOMIR() const { return emitOMIR; }
  bool shouldExportChiselInterface() const { return exportChiselInterface; }
  bool shouldDisableHoistingHWPassthrough() const {
    return disableHoistingHWPassthrough;
  }
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
  bool shouldEmitChiselAssertsAsSVA() const { return emitChiselAssertsAsSVA; }
  bool shouldEmitSeparateAlwaysBlocks() const {
    return emitSeparateAlwaysBlocks;
  }
  bool shouldAddMuxPragmas() const { return addMuxPragmas; }
  bool shouldAddVivadoRAMAddressConflictSynthesisBugWorkaround() const {
    return addVivadoRAMAddressConflictSynthesisBugWorkaround;
  }
  bool shouldExtractTestCode() const { return extractTestCode; }

private:
  std::string outputFilename;
  bool disableAnnotationsUnknown;
  bool disableAnnotationsClassless;
  bool lowerAnnotationsNoRefTypePorts;
  firrtl::PreserveAggregate::PreserveMode preserveAggregate;
  firrtl::PreserveValues::PreserveMode preserveMode;
  bool enableDebugInfo;
  BuildMode buildMode;
  bool disableOptimization;
  bool exportChiselInterface;
  std::string chiselInterfaceOutDirectory;
  bool vbToBV;
  bool noDedup;
  firrtl::CompanionMode companionMode;
  bool disableAggressiveMergeConnections;
  bool disableHoistingHWPassthrough;
  bool emitOMIR;
  std::string omirOutFile;
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
  bool emitChiselAssertsAsSVA;
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
  bool exportModuleHierarchy;
  bool stripFirDebugInfo;
  bool stripDebugInfo;
};

void registerFirtoolCLOptions();

LogicalResult populatePreprocessTransforms(mlir::PassManager &pm,
                                           bool disableAnnotationsUnknown,
                                           bool disableAnnotationsClassless,
                                           bool lowerAnnotationsNoRefTypePorts,
                                           bool enableDebugInfo);

LogicalResult populateCHIRRTLToLowFIRRTL(
    mlir::PassManager &pm, bool disableOptimization,
    firrtl::PreserveValues::PreserveMode preserveValues,
    firrtl::PreserveAggregate::PreserveMode preserveAggregate, bool replSeqMem,
    StringRef replSeqMemFile, bool ignoreReadEnableMem,
    bool exportChiselInterface, StringRef chiselInterfaceOutDirectory,
    bool disableHoistingHWPassthrough, bool noDedup, bool vbToBV,
    bool lowerMemories, FirtoolOptions::RandomKind disableRandom,
    firrtl::CompanionMode companionMode, StringRef blackBoxRoot, bool emitOMIR,
    StringRef omirOutFile, bool disableAggressiveMergeConnections);

LogicalResult populateLowFIRRTLToHW(mlir::PassManager &pm,
                                    bool disableOptimization,
                                    StringRef outputAnnotationFilename,
                                    bool enableAnnotationWarning,
                                    bool emitChiselAssertsAsSVA);

LogicalResult populateHWToSV(
    mlir::PassManager &pm, bool disableOptimization, bool extractTestCode,
    bool etcDisableInstanceExtraction, bool etcDisableRegisterExtraction,
    bool etcDisableModuleInlining, seq::ExternalizeClockGateOptions ckg,
    FirtoolOptions::RandomKind disableRandom, bool emitSeparateAlwaysBlocks,
    bool replSeqMem, bool ignoreReadEnableMem, bool addMuxPragmas,
    bool addVivadoRAMAddressConflictSynthesisBugWorkaround);

LogicalResult populateExportVerilog(mlir::PassManager &pm,
                                    bool disableOptimization,
                                    bool stripFirDebugInfo, bool stripDebugInfo,
                                    bool exportModuleHierarchy,
                                    std::unique_ptr<llvm::raw_ostream> os);

LogicalResult populateExportVerilog(mlir::PassManager &pm,
                                    bool disableOptimization,
                                    bool stripFirDebugInfo, bool stripDebugInfo,
                                    bool exportModuleHierarchy,
                                    llvm::raw_ostream &os);

LogicalResult populateExportSplitVerilog(
    mlir::PassManager &pm, bool disableOptimization, bool stripFirDebugInfo,
    bool stripDebugInfo, bool exportModuleHierarchy, llvm::StringRef directory);

LogicalResult populateFinalizeIR(mlir::PassManager &pm);

} // namespace firtool
} // namespace circt

#endif // CIRCT_FIRTOOL_FIRTOOL_H
