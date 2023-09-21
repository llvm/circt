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

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulatePreprocessTransforms(MlirPassManager pm);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateCHIRRTLToLowFIRRTL(
    MlirPassManager pm, MlirModule module, MlirStringRef inputFilename);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateLowFIRRTLToHW(MlirPassManager pm);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateHWToSV(MlirPassManager pm);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateExportVerilog(
    MlirPassManager pm, MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateExportSplitVerilog(MlirPassManager pm, MlirStringRef directory);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateFinalizeIR(MlirPassManager pm);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_FIRTOOL_FIRTOOL_H
