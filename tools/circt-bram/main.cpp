#include "circt/InitAllDialects.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Dialect/Resource/Interfaces/MemoryInterface.h"
#include "circt/Dialect/Resource/Passes/AffineToLoopScheduleModified.h"
#include "circt/Dialect/Resource/Passes/BRAMAnalysisAffine.h"
#include "circt/Dialect/Resource/Passes/BRAMAnalysisLoopSchedule.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  circt::registerAllDialects(registry);
  // register external interfaces for our dialects  
  hls::registerBRAMInterfaceExternalModels(registry);
  hls::registerBRAMAffineAnalysisPass();
  hls::registerBRAMLoopscheduleAnalysisPass();
  hls::registerAffineToLoopScheduleAnalysisPass();
  

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HLS BRAM analysis driver\n", registry));
}