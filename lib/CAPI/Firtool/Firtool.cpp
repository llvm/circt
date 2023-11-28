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

DEFINE_C_API_PTR_METHODS(CirctFirtoolFirtoolOptions,
                         circt::firtool::FirtoolOptions)

//===----------------------------------------------------------------------===//
// Option API.
//===----------------------------------------------------------------------===//

CirctFirtoolFirtoolOptions circtFirtoolOptionsCreateDefault() {
  auto *options = new firtool::FirtoolOptions();
  return wrap(options);
}

void circtFirtoolOptionsDestroy(CirctFirtoolFirtoolOptions options) {
  delete unwrap(options);
}

void circtFirtoolOptionsSetOutputFilename(CirctFirtoolFirtoolOptions options,
                                          MlirStringRef filename) {
  unwrap(options)->setOutputFilename(unwrap(filename));
}

void circtFirtoolOptionsDisableUnknownAnnotations(
    CirctFirtoolFirtoolOptions options, bool disable) {
  unwrap(options)->setDisableUnknownAnnotations(disable);
}

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MlirLogicalResult
firtoolPopulatePreprocessTransforms(MlirPassManager pm,
                                    CirctFirtoolFirtoolOptions options) {
  return wrap(
      firtool::populatePreprocessTransforms(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
firtoolPopulateCHIRRTLToLowFIRRTL(MlirPassManager pm,
                                  CirctFirtoolFirtoolOptions options,
                                  MlirStringRef inputFilename) {
  return wrap(firtool::populateCHIRRTLToLowFIRRTL(*unwrap(pm), *unwrap(options),
                                                  unwrap(inputFilename)));
}

MlirLogicalResult
firtoolPopulateLowFIRRTLToHW(MlirPassManager pm,
                             CirctFirtoolFirtoolOptions options) {
  return wrap(firtool::populateLowFIRRTLToHW(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult firtoolPopulateHWToSV(MlirPassManager pm,
                                        CirctFirtoolFirtoolOptions options) {
  return wrap(firtool::populateHWToSV(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
firtoolPopulateExportVerilog(MlirPassManager pm,
                             CirctFirtoolFirtoolOptions options,
                             MlirStringCallback callback, void *userData) {
  auto stream =
      std::make_unique<mlir::detail::CallbackOstream>(callback, userData);
  return wrap(firtool::populateExportVerilog(*unwrap(pm), *unwrap(options),
                                             std::move(stream)));
}

MlirLogicalResult
firtoolPopulateExportSplitVerilog(MlirPassManager pm,
                                  CirctFirtoolFirtoolOptions options,
                                  MlirStringRef directory) {
  return wrap(firtool::populateExportSplitVerilog(*unwrap(pm), *unwrap(options),
                                                  unwrap(directory)));
}

MlirLogicalResult
firtoolPopulateFinalizeIR(MlirPassManager pm,
                          CirctFirtoolFirtoolOptions options) {
  return wrap(firtool::populateFinalizeIR(*unwrap(pm), *unwrap(options)));
}
