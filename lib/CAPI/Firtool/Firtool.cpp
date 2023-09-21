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

// TODO: Custom options with C-API
static auto tempDefaultCategory = llvm::cl::OptionCategory{"temp-firtool-capi"};
static auto tempDefaultOptions = firtool::FirtoolOptions{tempDefaultCategory};

MlirLogicalResult firtoolPopulatePreprocessTransforms(MlirPassManager pm) {
  return wrap(
      firtool::populatePreprocessTransforms(*unwrap(pm), tempDefaultOptions));
}

MlirLogicalResult
firtoolPopulateCHIRRTLToLowFIRRTL(MlirPassManager pm, MlirModule module,
                                  MlirStringRef inputFilename) {
  return wrap(firtool::populateCHIRRTLToLowFIRRTL(
      *unwrap(pm), tempDefaultOptions, unwrap(module), unwrap(inputFilename)));
}

MlirLogicalResult firtoolPopulateLowFIRRTLToHW(MlirPassManager pm) {
  return wrap(firtool::populateLowFIRRTLToHW(*unwrap(pm), tempDefaultOptions));
}

MlirLogicalResult firtoolPopulateHWToSV(MlirPassManager pm) {
  return wrap(firtool::populateHWToSV(*unwrap(pm), tempDefaultOptions));
}

MlirLogicalResult firtoolPopulateExportVerilog(MlirPassManager pm,
                                               MlirStringCallback callback,
                                               void *userData) {
  auto stream =
      std::make_unique<mlir::detail::CallbackOstream>(callback, userData);
  return wrap(firtool::populateExportVerilog(*unwrap(pm), tempDefaultOptions,
                                             std::move(stream)));
}

MlirLogicalResult firtoolPopulateExportSplitVerilog(MlirPassManager pm,
                                                    MlirStringRef directory) {
  return wrap(firtool::populateExportSplitVerilog(
      *unwrap(pm), tempDefaultOptions, unwrap(directory)));
}

MlirLogicalResult firtoolPopulateFinalizeIR(MlirPassManager pm) {
  return wrap(firtool::populateFinalizeIR(*unwrap(pm), tempDefaultOptions));
}
