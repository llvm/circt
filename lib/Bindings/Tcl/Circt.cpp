#include "Circt.h"

#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/ESI.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/MSFT.h"
#include "circt-c/Dialect/SV.h"
#include "circt-c/Dialect/Seq.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir-c/Registration.h"
#include "mlir/IR/Operation.h"

int operationTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

void operationTypeUpdateStringProc(Tcl_Obj *obj) {
  obj->bytes = Tcl_Alloc(1);
  obj->bytes[0] = '\0';
  obj->length = 0;
}

void operationTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  auto *op = ((mlir::Operation *) src->internalRep.otherValuePtr)->clone();
  dup->internalRep.otherValuePtr = op;
}

void operationTypeFreeIntRepProc(Tcl_Obj *obj) {
  // TODO: make this not leak memory without segfaulting
  //auto *op = (mlir::Operation *) obj->internalRep.otherValuePtr;
  //op->erase();
}

