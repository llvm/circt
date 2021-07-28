//===- MSFT.cpp - C Interface for the MSFT Dialect ------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/MSFT.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/Support/raw_ostream.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MSFT, msft, circt::msft::MSFTDialect)

using namespace circt::msft;

void mlirMSFTRegisterPasses() { circt::msft::registerMSFTPasses(); }

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

void mlirMSFTAddPhysLocationAttr(MlirOperation cOp, const char *entityName,
                                 DeviceType type, long x, long y, long num) {
  mlir::Operation *op = unwrap(cOp);
  mlir::MLIRContext *ctxt = op->getContext();
  PhysLocationAttr loc =
      PhysLocationAttr::get(ctxt, DeviceTypeAttr::get(ctxt, type), x, y, num);
  llvm::SmallString<64> entity("loc:");
  entity.append(entityName);
  op->setAttr(entity, loc);
}
