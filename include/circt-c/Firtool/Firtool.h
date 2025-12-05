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

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(CirctFirtoolFirtoolOptions, void);

#undef DEFINE_C_API_STRUCT

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CirctFirtoolPreserveAggregateMode {
  CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE,
  CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC,
  CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC,
  CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL,
} CirctFirtoolPreserveAggregateMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CirctFirtoolPreserveValuesMode {
  CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_STRIP,
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
typedef enum CirctFirtoolBuildMode {
  CIRCT_FIRTOOL_BUILD_MODE_DEFAULT,
  CIRCT_FIRTOOL_BUILD_MODE_DEBUG,
  CIRCT_FIRTOOL_BUILD_MODE_RELEASE,
} CirctFirtoolBuildMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CirctFirtoolRandomKind {
  CIRCT_FIRTOOL_RANDOM_KIND_NONE,
  CIRCT_FIRTOOL_RANDOM_KIND_MEM,
  CIRCT_FIRTOOL_RANDOM_KIND_REG,
  CIRCT_FIRTOOL_RANDOM_KIND_ALL,
} CirctFirtoolRandomKind;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CirctFirtoolVerificationFlavor {
  CIRCT_FIRTOOL_VERIFICATION_FLAVOR_NONE,
  CIRCT_FIRTOOL_VERIFICATION_FLAVOR_IF_ELSE_FATAL,
  CIRCT_FIRTOOL_VERIFICATION_FLAVOR_IMMEDIATE,
  CIRCT_FIRTOOL_VERIFICATION_FLAVOR_SVA,
} CirctFirtoolVerificationFlavor;

MLIR_CAPI_EXPORTED CirctFirtoolFirtoolOptions
circtFirtoolOptionsCreateDefault(void);
MLIR_CAPI_EXPORTED void
circtFirtoolOptionsDestroy(CirctFirtoolFirtoolOptions options);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetOutputFilename(CirctFirtoolFirtoolOptions options,
                                     MlirStringRef filename);
MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetDisableUnknownAnnotations(
    CirctFirtoolFirtoolOptions options, bool disable);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetDisableAnnotationsClassless(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetLowerAnnotationsNoRefTypePorts(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetAllowAddingPortsOnPublic(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetPreserveAggregate(
    CirctFirtoolFirtoolOptions options,
    CirctFirtoolPreserveAggregateMode value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetPreserveValues(CirctFirtoolFirtoolOptions options,
                                     CirctFirtoolPreserveValuesMode value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetEnableDebugInfo(CirctFirtoolFirtoolOptions options,
                                      bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetBuildMode(CirctFirtoolFirtoolOptions options,
                                CirctFirtoolBuildMode value);
MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetDisableLayerSink(CirctFirtoolFirtoolOptions options,
                                       bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetDisableOptimization(CirctFirtoolFirtoolOptions options,
                                          bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetVbToBv(CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetNoDedup(CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetCompanionMode(CirctFirtoolFirtoolOptions options,
                                    CirctFirtoolCompanionMode value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetNoViews(CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetDisableAggressiveMergeConnections(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetLowerMemories(CirctFirtoolFirtoolOptions options,
                                    bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetBlackBoxRootPath(CirctFirtoolFirtoolOptions options,
                                       MlirStringRef value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetReplSeqMem(CirctFirtoolFirtoolOptions options,
                                 bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetReplSeqMemFile(CirctFirtoolFirtoolOptions options,
                                     MlirStringRef value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetExtractTestCode(CirctFirtoolFirtoolOptions options,
                                      bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetIgnoreReadEnableMem(CirctFirtoolFirtoolOptions options,
                                          bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetDisableRandom(CirctFirtoolFirtoolOptions options,
                                    CirctFirtoolRandomKind value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetOutputAnnotationFilename(
    CirctFirtoolFirtoolOptions options, MlirStringRef value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetEnableAnnotationWarning(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetAddMuxPragmas(CirctFirtoolFirtoolOptions options,
                                    bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetVerificationFlavor(CirctFirtoolFirtoolOptions options,
                                         CirctFirtoolVerificationFlavor value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetEmitSeparateAlwaysBlocks(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetEtcDisableInstanceExtraction(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetEtcDisableRegisterExtraction(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetEtcDisableModuleInlining(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(
    CirctFirtoolFirtoolOptions options, bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetCkgModuleName(CirctFirtoolFirtoolOptions options,
                                    MlirStringRef value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetCkgInputName(CirctFirtoolFirtoolOptions options,
                                   MlirStringRef value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetCkgOutputName(CirctFirtoolFirtoolOptions options,
                                    MlirStringRef value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetCkgEnableName(CirctFirtoolFirtoolOptions options,
                                    MlirStringRef value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetCkgTestEnableName(CirctFirtoolFirtoolOptions options,
                                        MlirStringRef value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetExportModuleHierarchy(CirctFirtoolFirtoolOptions options,
                                            bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetStripFirDebugInfo(CirctFirtoolFirtoolOptions options,
                                        bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetStripDebugInfo(CirctFirtoolFirtoolOptions options,
                                     bool value);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetDisableCSEinClasses(CirctFirtoolFirtoolOptions options,
                                          bool value);

MLIR_CAPI_EXPORTED void circtFirtoolOptionsSetSelectDefaultInstanceChoice(
    CirctFirtoolFirtoolOptions options, bool value);

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulatePreprocessTransforms(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateCHIRRTLToLowFIRRTL(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateLowFIRRTLToHW(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options,
    MlirStringRef inputFilename);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateHWToSV(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateExportVerilog(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options,
    MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateExportSplitVerilog(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options,
    MlirStringRef directory);

MLIR_CAPI_EXPORTED MlirLogicalResult circtFirtoolPopulateFinalizeIR(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_FIRTOOL_FIRTOOL_H
