//===- RTLDialect.cpp - C Interface EmitVerilog ---------------------------===//
//
//  Implements a C Interface for emitVerilog
//
//===----------------------------------------------------------------------===//

#include "circt-c/EmitVerilog.h"

#include "circt/EmitVerilog.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;

MlirLogicalResult mlirEmitVerilog(MlirModule module,
                                  MlirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  return wrap(emitVerilog(unwrap(module), stream));
}
