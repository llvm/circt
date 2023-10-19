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

//===----------------------------------------------------------------------===//
// Option API.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(FirtoolGeneralOptions, void);
DEFINE_C_API_STRUCT(FirtoolPreprocessTransformsOptions, void);
DEFINE_C_API_STRUCT(FirtoolCHIRRTLToLowFIRRTLOptions, void);
DEFINE_C_API_STRUCT(FirtoolLowFIRRTLToHWOptions, void);
DEFINE_C_API_STRUCT(FirtoolHWToSVOptions, void);
DEFINE_C_API_STRUCT(FirtoolExportVerilogOptions, void);
DEFINE_C_API_STRUCT(FirtoolFinalizeIROptions, void);

#undef DEFINE_C_API_STRUCT

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolPreserveAggregateMode {
  FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE,
  FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC,
  FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC,
  FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL,
} FirtoolPreserveAggregateMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolPreserveValuesMode {
  FIRTOOL_PRESERVE_VALUES_MODE_NONE,
  FIRTOOL_PRESERVE_VALUES_MODE_NAMED,
  FIRTOOL_PRESERVE_VALUES_MODE_ALL,
} FirtoolPreserveValuesMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolCompanionMode {
  FIRTOOL_COMPANION_MODE_BIND,
  FIRTOOL_COMPANION_MODE_INSTANTIATE,
  FIRTOOL_COMPANION_MODE_DROP,
} FirtoolCompanionMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolBuildMode {
  FIRTOOL_BUILD_MODE_DEBUG,
  FIRTOOL_BUILD_MODE_RELEASE,
} FirtoolBuildMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolRandomKind {
  FIRTOOL_RANDOM_KIND_NONE,
  FIRTOOL_RANDOM_KIND_MEM,
  FIRTOOL_RANDOM_KIND_REG,
  FIRTOOL_RANDOM_KIND_ALL,
} FirtoolRandomKind;

#define DECLARE_FIRTOOL_OPTIONS_CREATE_DESTROY(opt)                            \
  MLIR_CAPI_EXPORTED Firtool##opt##Options firtool##opt##OptionsCreateDefault( \
      FirtoolGeneralOptions general);                                          \
  MLIR_CAPI_EXPORTED void firtool##opt##OptionsDestroy(                        \
      Firtool##opt##Options options);                                          \
  MLIR_CAPI_EXPORTED void firtool##opt##OptionsSetGeneral(                     \
      Firtool##opt##Options options, FirtoolGeneralOptions general);           \
  MLIR_CAPI_EXPORTED FirtoolGeneralOptions firtool##opt##OptionsGetGeneral(    \
      Firtool##opt##Options options);

#define DECLARE_FIRTOOL_OPTION(opt, name, type)                                \
  MLIR_CAPI_EXPORTED void firtool##opt##OptionsSet##name(                      \
      Firtool##opt##Options options, type value);                              \
  MLIR_CAPI_EXPORTED type firtool##opt##OptionsGet##name(                      \
      Firtool##opt##Options options)

// ========== General ==========

MLIR_CAPI_EXPORTED FirtoolGeneralOptions firtoolGeneralOptionsCreateDefault();
MLIR_CAPI_EXPORTED void
firtoolGeneralOptionsDestroy(FirtoolGeneralOptions options);
DECLARE_FIRTOOL_OPTION(General, DisableOptimization, bool);
DECLARE_FIRTOOL_OPTION(General, ReplSeqMem, bool);
DECLARE_FIRTOOL_OPTION(General, ReplSeqMemFile, MlirStringRef);
DECLARE_FIRTOOL_OPTION(General, IgnoreReadEnableMem, bool);
DECLARE_FIRTOOL_OPTION(General, DisableRandom, FirtoolRandomKind);

// ========== PreprocessTransforms ==========

DECLARE_FIRTOOL_OPTIONS_CREATE_DESTROY(PreprocessTransforms);
DECLARE_FIRTOOL_OPTION(PreprocessTransforms, DisableAnnotationsUnknown, bool);
DECLARE_FIRTOOL_OPTION(PreprocessTransforms, DisableAnnotationsClassless, bool);
DECLARE_FIRTOOL_OPTION(PreprocessTransforms, LowerAnnotationsNoRefTypePorts,
                       bool);

// ========== CHIRRTLToLowFIRRTL ==========

DECLARE_FIRTOOL_OPTIONS_CREATE_DESTROY(CHIRRTLToLowFIRRTL);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, PreserveValues,
                       FirtoolPreserveValuesMode);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, PreserveAggregate,
                       FirtoolPreserveAggregateMode);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, ExportChiselInterface, bool);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, ChiselInterfaceOutDirectory,
                       MlirStringRef);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, DisableHoistingHWPassthrough, bool);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, Dedup, bool);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, NoDedup, bool);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, VbToBv, bool);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, LowerMemories, bool);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, CompanionMode, FirtoolCompanionMode);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, BlackBoxRootPath, MlirStringRef);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, EmitOMIR, bool);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, OMIROutFile, MlirStringRef);
DECLARE_FIRTOOL_OPTION(CHIRRTLToLowFIRRTL, DisableAggressiveMergeConnections,
                       bool);

// ========== LowFIRRTLToHW ==========

DECLARE_FIRTOOL_OPTIONS_CREATE_DESTROY(LowFIRRTLToHW);
DECLARE_FIRTOOL_OPTION(LowFIRRTLToHW, OutputAnnotationFilename, MlirStringRef);
DECLARE_FIRTOOL_OPTION(LowFIRRTLToHW, EnableAnnotationWarning, bool);
DECLARE_FIRTOOL_OPTION(LowFIRRTLToHW, EmitChiselAssertsAsSVA, bool);

// ========== HWToSV ==========

DECLARE_FIRTOOL_OPTIONS_CREATE_DESTROY(HWToSV);
DECLARE_FIRTOOL_OPTION(HWToSV, ExtractTestCode, bool);
DECLARE_FIRTOOL_OPTION(HWToSV, EtcDisableInstanceExtraction, bool);
DECLARE_FIRTOOL_OPTION(HWToSV, EtcDisableRegisterExtraction, bool);
DECLARE_FIRTOOL_OPTION(HWToSV, EtcDisableModuleInlining, bool);
DECLARE_FIRTOOL_OPTION(HWToSV, CkgModuleName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(HWToSV, CkgInputName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(HWToSV, CkgOutputName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(HWToSV, CkgEnableName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(HWToSV, CkgTestEnableName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(HWToSV, CkgInstName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(HWToSV, EmitSeparateAlwaysBlocks, bool);
DECLARE_FIRTOOL_OPTION(HWToSV, AddMuxPragmas, bool);
DECLARE_FIRTOOL_OPTION(HWToSV,
                       AddVivadoRAMAddressConflictSynthesisBugWorkaround, bool);

// ========== ExportVerilog ==========

DECLARE_FIRTOOL_OPTIONS_CREATE_DESTROY(ExportVerilog);
DECLARE_FIRTOOL_OPTION(ExportVerilog, StripFirDebugInfo, bool);
DECLARE_FIRTOOL_OPTION(ExportVerilog, StripDebugInfo, bool);
DECLARE_FIRTOOL_OPTION(ExportVerilog, ExportModuleHierarchy, bool);
// Filename for ExportVerilog, directory for ExportSplitVerilog
DECLARE_FIRTOOL_OPTION(ExportVerilog, OutputPath, MlirStringRef);

#undef DECLARE_FIRTOOL_OPTION
#undef DECLARE_FIRTOOL_OPTIONS_CREATE_DESTROY

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulatePreprocessTransforms(
    MlirPassManager pm, FirtoolPreprocessTransformsOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateCHIRRTLToLowFIRRTL(
    MlirPassManager pm, FirtoolCHIRRTLToLowFIRRTLOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateLowFIRRTLToHW(
    MlirPassManager pm, FirtoolLowFIRRTLToHWOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateHWToSV(MlirPassManager pm, FirtoolHWToSVOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateExportVerilog(
    MlirPassManager pm, FirtoolExportVerilogOptions options,
    MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateExportSplitVerilog(
    MlirPassManager pm, FirtoolExportVerilogOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateFinalizeIR(MlirPassManager pm, FirtoolFinalizeIROptions options);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_FIRTOOL_FIRTOOL_H
