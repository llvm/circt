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
  FirtoolOptions(std::nullopt_t) : FirtoolOptions() {}

  // Helper Types
  enum BuildMode { BuildModeDefault, BuildModeDebug, BuildModeRelease };
  enum class RandomKind { None, Mem, Reg, All };


  bool isRandomEnabled(RandomKind kind) const {
    return disableRandom != RandomKind::All && disableRandom != kind;
  }

  firrtl::PreserveValues::PreserveMode getPreserveMode() const {
    if (buildMode == BuildModeDefault)
      return preserveMode;
    switch (buildMode) {
    case BuildModeDebug:
      return firrtl::PreserveValues::Named;
    case BuildModeRelease:
      return firrtl::PreserveValues::None;
    }
    llvm_unreachable("unknown build mode");
  }


private:
  std::string outputFilename;

  bool disableAnnotationsUnknown;

  bool disableAnnotationsClassless;

  bool lowerAnnotationsNoRefTypePorts;
  circt::firrtl::PreserveAggregate::PreserveMode preserveAggregate;
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
                                           const FirtoolOptions &opt);

LogicalResult populateCHIRRTLToLowFIRRTL(mlir::PassManager &pm,
                                         const FirtoolOptions &opt,
                                         StringRef inputFilename);

LogicalResult populateLowFIRRTLToHW(mlir::PassManager &pm,
                                    const FirtoolOptions &opt);

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

} // namespace firtool
} // namespace circt

#endif // CIRCT_FIRTOOL_FIRTOOL_H
