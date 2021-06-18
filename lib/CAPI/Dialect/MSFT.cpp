//===- MSFT.cpp - C Interface for the MSFT Dialect ------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/MSFT.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/Support/raw_ostream.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MSFT, msft, circt::msft::MSFTDialect)

using namespace circt::msft;

MlirLogicalResult mlirMSFTExportTcl(MlirModule module,
                                    MlirStringCallback callback,
                                    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(exportQuartusTcl(unwrap(module), stream));
}

void mlirMSFTRegisterGenerator(MlirContext cCtxt, const char *opName,
                               const char *generatorName,
                               mlirMSFTGeneratorCallback cb,
                               MlirAttribute parameters) {
  mlir::MLIRContext *ctxt = unwrap(cCtxt);
  MSFTDialect *msft = ctxt->getLoadedDialect<MSFTDialect>();
  msft->registerGenerator(
      llvm::StringRef(opName), llvm::StringRef(generatorName),
      [cb](mlir::Operation *op) {
        return unwrap(cb.callback(wrap(op), cb.userData));
      },
      unwrap(parameters));
}
