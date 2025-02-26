//===- ExportSMTLIB.cpp - C Interface to ExportSMTLIB ---------------------===//
//
//  Implements a C Interface for export SMTLIB.
//
//===----------------------------------------------------------------------===//

#include "circt-c/ExportSMTLIB.h"

#include "circt/Target/ExportSMTLIB.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;

MlirLogicalResult mlirExportSMTLIB(MlirModule module,
                                   MlirStringCallback callback,
                                   void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(ExportSMTLIB::exportSMTLIB(unwrap(module), stream));
}
