//===- Firtool.cpp - C Interface to Firtool-lib ---------------------------===//
//
//  Implements a C Interface for Firtool-lib.
//
//===----------------------------------------------------------------------===//

#include "circt-c/Firtool/Firtool.h"

#include "circt/Firtool/Firtool.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MlirLogicalResult circtFirtoolPopulatePreprocessTransforms(
    MlirPassManager pm, bool disableAnnotationsUnknown,
    bool disableAnnotationsClassless, bool lowerAnnotationsNoRefTypePorts,
    bool enableDebugInfo) {
  return wrap(firtool::populatePreprocessTransforms(
      *unwrap(pm), disableAnnotationsUnknown, disableAnnotationsClassless,
      lowerAnnotationsNoRefTypePorts, enableDebugInfo));
}

MlirLogicalResult circtFirtoolPopulateCHIRRTLToLowFIRRTL(
    MlirPassManager pm, bool disableOptimization,
    CirctFirtoolPreserveValuesMode preserveValues,
    CirctFirtoolPreserveAggregateMode preserveAggregate, bool replSeqMem,
    MlirStringRef replSeqMemFile, bool ignoreReadEnableMem,
    bool exportChiselInterface, MlirStringRef chiselInterfaceOutDirectory,
    bool disableHoistingHWPassthrough, bool noDedup, bool vbToBV,
    bool lowerMemories, CirctFirtoolRandomKind disableRandom,
    CirctFirtoolCompanionMode companionMode, MlirStringRef blackBoxRoot,
    bool emitOMIR, MlirStringRef omirOutFile,
    bool disableAggressiveMergeConnections) {

  firrtl::PreserveValues::PreserveMode preserveValuesMode;

  switch (preserveValues) {
  case CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_SRIP:
    preserveValuesMode = firrtl::PreserveValues::PreserveMode::Strip;
    break;
  case CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_NONE:
    preserveValuesMode = firrtl::PreserveValues::PreserveMode::Strip;
    break;
  case CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_NAMED:
    preserveValuesMode = firrtl::PreserveValues::PreserveMode::Named;
    break;
  case CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_ALL:
    preserveValuesMode = firrtl::PreserveValues::PreserveMode::All;
    break;
  default:
    llvm_unreachable("unknown preserve values mode");
  }

  firrtl::PreserveAggregate::PreserveMode preserveAggregateMode;

  switch (preserveAggregate) {
  case CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE:
    preserveAggregateMode = firrtl::PreserveAggregate::PreserveMode::None;
    break;
  case CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC:
    preserveAggregateMode = firrtl::PreserveAggregate::PreserveMode::OneDimVec;
    break;
  case CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC:
    preserveAggregateMode = firrtl::PreserveAggregate::PreserveMode::Vec;
    break;
  case CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL:
    preserveAggregateMode = firrtl::PreserveAggregate::PreserveMode::All;
    break;
  default:
    llvm_unreachable("unknown preserve aggregate mode");
  }

  firtool::FirtoolOptions::RandomKind disableRandomKind;

  switch (disableRandom) {
  case CIRCT_FIRTOOL_RANDOM_KIND_NONE:
    disableRandomKind = firtool::FirtoolOptions::RandomKind::None;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_MEM:
    disableRandomKind = firtool::FirtoolOptions::RandomKind::Mem;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_REG:
    disableRandomKind = firtool::FirtoolOptions::RandomKind::Reg;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_ALL:
    disableRandomKind = firtool::FirtoolOptions::RandomKind::All;
    break;
  default:
    llvm_unreachable("unknown random kind");
  }

  firrtl::CompanionMode companionModeValue;

  switch (companionMode) {
  case CIRCT_FIRTOOL_COMPANION_MODE_BIND:
    companionModeValue = firrtl::CompanionMode::Bind;
    break;
  case CIRCT_FIRTOOL_COMPANION_MODE_INSTANTIATE:
    companionModeValue = firrtl::CompanionMode::Instantiate;
    break;
  case CIRCT_FIRTOOL_COMPANION_MODE_DROP:
    companionModeValue = firrtl::CompanionMode::Drop;
    break;
  default:
    llvm_unreachable("unknown companion mode");
  }

  return wrap(firtool::populateCHIRRTLToLowFIRRTL(
      *unwrap(pm), disableOptimization, preserveValuesMode,
      preserveAggregateMode, replSeqMem, unwrap(replSeqMemFile),
      ignoreReadEnableMem, exportChiselInterface,
      unwrap(chiselInterfaceOutDirectory), disableHoistingHWPassthrough,
      noDedup, vbToBV, lowerMemories, disableRandomKind, companionModeValue,
      unwrap(blackBoxRoot), emitOMIR, unwrap(omirOutFile),
      disableAggressiveMergeConnections));
}

MlirLogicalResult
circtFirtoolPopulateLowFIRRTLToHW(MlirPassManager pm, bool disableOptimization,
                                  MlirStringRef outputAnnotationFilename,
                                  bool enableAnnotationWarning,
                                  bool emitChiselAssertsAsSVA) {
  return wrap(firtool::populateLowFIRRTLToHW(
      *unwrap(pm), disableOptimization, unwrap(outputAnnotationFilename),
      enableAnnotationWarning, emitChiselAssertsAsSVA));
}

MlirLogicalResult circtFirtoolPopulateHWToSV(
    MlirPassManager pm, bool disableOptimization, bool extractTestCode,
    bool etcDisableInstanceExtraction, bool etcDisableRegisterExtraction,
    bool etcDisableModuleInlining, MlirStringRef ckgModuleName,
    MlirStringRef ckgInputName, MlirStringRef ckgOutputName,
    MlirStringRef ckgEnableName, MlirStringRef ckgTestEnableName,
    MlirStringRef ckgInstName, CirctFirtoolRandomKind disableRandom,
    bool emitSeparateAlwaysBlocks, bool replSeqMem, bool ignoreReadEnableMem,
    bool addMuxPragmas,
    bool addVivadoRAMAddressConflictSynthesisBugWorkaround) {

  firtool::FirtoolOptions::RandomKind disableRandomKind;

  switch (disableRandom) {
  case CIRCT_FIRTOOL_RANDOM_KIND_NONE:
    disableRandomKind = firtool::FirtoolOptions::RandomKind::None;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_MEM:
    disableRandomKind = firtool::FirtoolOptions::RandomKind::Mem;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_REG:
    disableRandomKind = firtool::FirtoolOptions::RandomKind::Reg;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_ALL:
    disableRandomKind = firtool::FirtoolOptions::RandomKind::All;
    break;
  default:
    llvm_unreachable("unknown random kind");
  }

  return wrap(firtool::populateHWToSV(
      *unwrap(pm), disableOptimization, extractTestCode,
      etcDisableInstanceExtraction, etcDisableRegisterExtraction,
      etcDisableModuleInlining,
      {unwrap(ckgModuleName).str(), unwrap(ckgInputName).str(),
       unwrap(ckgOutputName).str(), unwrap(ckgEnableName).str(),
       unwrap(ckgTestEnableName).str(), unwrap(ckgInstName).str()},
      disableRandomKind, emitSeparateAlwaysBlocks, replSeqMem,
      ignoreReadEnableMem, addMuxPragmas,
      addVivadoRAMAddressConflictSynthesisBugWorkaround));
}

MlirLogicalResult
circtFirtoolPopulateExportVerilog(MlirPassManager pm, bool disableOptimization,
                                  bool stripFirDebugInfo, bool stripDebugInfo,
                                  bool exportModuleHierarchy,
                                  MlirStringCallback callback, void *userData) {
  auto stream =
      std::make_unique<mlir::detail::CallbackOstream>(callback, userData);
  return wrap(firtool::populateExportVerilog(
      *unwrap(pm), disableOptimization, stripFirDebugInfo, stripDebugInfo,
      exportModuleHierarchy, std::move(stream)));
}

MlirLogicalResult circtFirtoolPopulateExportSplitVerilog(
    MlirPassManager pm, bool disableOptimization, bool stripFirDebugInfo,
    bool stripDebugInfo, bool exportModuleHierarchy, MlirStringRef directory) {
  return wrap(firtool::populateExportSplitVerilog(
      *unwrap(pm), disableOptimization, stripFirDebugInfo, stripDebugInfo,
      exportModuleHierarchy, unwrap(directory)));
}

MlirLogicalResult circtFirtoolPopulateFinalizeIR(MlirPassManager pm) {
  return wrap(firtool::populateFinalizeIR(*unwrap(pm)));
}
