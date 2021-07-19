#include "Circt.h"

#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/ESI.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/MSFT.h"
#include "circt-c/Dialect/SV.h"
#include "circt-c/Dialect/Seq.h"
#include "mlir-c/Registration.h"

int tclRegisterDialects(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
  auto *contextObj = Tcl_NewObj();
  contextObj->typePtr = Tcl_GetObjType("MlirContext");
  auto context = mlirContextCreate();

  // Collect CIRCT dialects to register.
  MlirDialectHandle comb = mlirGetDialectHandle__comb__();
  mlirDialectHandleRegisterDialect(comb, context);
  mlirDialectHandleLoadDialect(comb, context);

  MlirDialectHandle esi = mlirGetDialectHandle__esi__();
  mlirDialectHandleRegisterDialect(esi, context);
  mlirDialectHandleLoadDialect(esi, context);

  MlirDialectHandle msft = mlirGetDialectHandle__msft__();
  mlirDialectHandleRegisterDialect(msft, context);
  mlirDialectHandleLoadDialect(msft, context);

  MlirDialectHandle hw = mlirGetDialectHandle__hw__();
  mlirDialectHandleRegisterDialect(hw, context);
  mlirDialectHandleLoadDialect(hw, context);

  MlirDialectHandle seq = mlirGetDialectHandle__seq__();
  mlirDialectHandleRegisterDialect(seq, context);
  mlirDialectHandleLoadDialect(seq, context);

  MlirDialectHandle sv = mlirGetDialectHandle__sv__();
  mlirDialectHandleRegisterDialect(sv, context);
  mlirDialectHandleLoadDialect(sv, context);

  contextObj->internalRep.otherValuePtr = new MlirContext(context);
  Tcl_SetObjResult(interp, contextObj);
  return TCL_OK;
}

int contextTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

void contextTypeUpdateStringProc(Tcl_Obj *obj) {
  obj->bytes = Tcl_Alloc(1);
  obj->bytes[0] = '\0';
  obj->length = 0;
}

void contextTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  auto *context = new MlirContext(*(MlirContext *) src->internalRep.otherValuePtr);
  dup->internalRep.otherValuePtr = context;
}

void contextTypeFreeIntRepProc(Tcl_Obj *obj) {
  delete (MlirContext *) obj->internalRep.otherValuePtr;
}

