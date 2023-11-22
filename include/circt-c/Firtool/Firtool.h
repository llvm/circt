//===-- circt-c/Firtool/Firtool.h - C API for Firtool-lib ---------*- C -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_FIRTOOL_FIRTOOL_H
#define CIRCT_C_FIRTOOL_FIRTOOL_H

#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CirctFirtoolPreserveAggregateMode {
  CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE,
  CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC,
  CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC,
  CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL,
} CirctFirtoolPreserveAggregateMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CirctFirtoolPreserveValuesMode {
  CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_SRIP,
  CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_NONE,
  CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_NAMED,
  CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_ALL,
} CirctFirtoolPreserveValuesMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CirctFirtoolCompanionMode {
  CIRCT_FIRTOOL_COMPANION_MODE_BIND,
  CIRCT_FIRTOOL_COMPANION_MODE_INSTANTIATE,
  CIRCT_FIRTOOL_COMPANION_MODE_DROP,
} CirctFirtoolCompanionMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CirctFirtoolRandomKind {
  CIRCT_FIRTOOL_RANDOM_KIND_NONE,
  CIRCT_FIRTOOL_RANDOM_KIND_MEM,
  CIRCT_FIRTOOL_RANDOM_KIND_REG,
  CIRCT_FIRTOOL_RANDOM_KIND_ALL,
} CirctFirtoolRandomKind;

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulatePreprocessTransforms(
    MlirPassManager pm, bool disableAnnotationsUnknown,
    bool disableAnnotationsClassless, bool lowerAnnotationsNoRefTypePorts,
    bool enableDebugInfo);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateCHIRRTLToLowFIRRTL(
    MlirPassManager pm, bool disableOptimization,
    CirctFirtoolPreserveValuesMode preserveValues,
    CirctFirtoolPreserveAggregateMode preserveAggregate, bool replSeqMem,
    MlirStringRef replSeqMemFile, bool ignoreReadEnableMem,
    bool exportChiselInterface, MlirStringRef chiselInterfaceOutDirectory,
    bool disableHoistingHWPassthrough, bool noDedup, bool vbToBV,
    bool lowerMemories, CirctFirtoolRandomKind disableRandom,
    CirctFirtoolCompanionMode companionMode, MlirStringRef blackBoxRoot,
    bool emitOMIR, MlirStringRef omirOutFile,
    bool disableAggressiveMergeConnections);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateLowFIRRTLToHW(
    MlirPassManager pm, bool disableOptimization,
    MlirStringRef outputAnnotationFilename, bool enableAnnotationWarning,
    bool emitChiselAssertsAsSVA);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateHWToSV(
    MlirPassManager pm, bool disableOptimization, bool extractTestCode,
    bool etcDisableInstanceExtraction, bool etcDisableRegisterExtraction,
    bool etcDisableModuleInlining, MlirStringRef ckgModuleName,
    MlirStringRef ckgInputName, MlirStringRef ckgOutputName,
    MlirStringRef ckgEnableName, MlirStringRef ckgTestEnableName,
    MlirStringRef ckgInstName, CirctFirtoolRandomKind disableRandom,
    bool emitSeparateAlwaysBlocks, bool replSeqMem, bool ignoreReadEnableMem,
    bool addMuxPragmas, bool addVivadoRAMAddressConflictSynthesisBugWorkaround);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateExportVerilog(
    MlirPassManager pm, bool disableOptimization, bool stripFirDebugInfo,
    bool stripDebugInfo, bool exportModuleHierarchy,
    MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateExportSplitVerilog(
    MlirPassManager pm, bool disableOptimization, bool stripFirDebugInfo,
    bool stripDebugInfo, bool exportModuleHierarchy, MlirStringRef directory);

MLIR_CAPI_EXPORTED MlirLogicalResult
circtFirtoolPopulateFinalizeIR(MlirPassManager pm);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_FIRTOOL_FIRTOOL_H
