#include <stdlib.h>
#include <tcl.h>

#include "Circt.h"
#include "Query.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

int loadFirMlirFile(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
  if (objc != 4) {
    return TCL_ERROR;
  }

  auto *contextObj = objv[2];
  if (Tcl_ConvertToType(interp, contextObj, Tcl_GetObjType("MlirContext")) != TCL_OK) {
    return TCL_ERROR;
  }
  auto *context = unwrap((MlirContext) { contextObj->internalRep.otherValuePtr });

  std::string errorMessage;
  auto input = mlir::openInputFile(llvm::StringRef(objv[3]->bytes), &errorMessage);

  if (!input) {
    llvm::errs() << errorMessage << '\n';
    return TCL_ERROR;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, context);

  MlirOperation module;
  if (!strcmp(objv[1]->bytes, "MLIR")) {
    module = wrap(mlir::parseSourceFile(sourceMgr, context).release().getOperation());
  } else if (!strcmp(objv[1]->bytes, "FIR")) {
    // TODO
    return TCL_ERROR;
  } else {
    return TCL_ERROR;
  }

  if (mlirOperationIsNull(module)) {
    return TCL_ERROR;
  }

  auto *m = module.ptr;

  auto *obj = Tcl_NewObj();
  obj->typePtr = Tcl_GetObjType("MlirOperation");
  obj->internalRep.twoPtrValue.ptr1 = (void *) m;
  obj->internalRep.twoPtrValue.ptr2 = (void *) contextObj;
  Tcl_IncrRefCount(contextObj);
  obj->length = 0;
  obj->bytes = nullptr;
  Tcl_SetObjResult(interp, obj);

  return TCL_OK;
}

extern "C" {

int DLLEXPORT Circt_Init(Tcl_Interp *interp) {
  if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
    return TCL_ERROR;
  }

  // Register types
  Tcl_ObjType *contextType = new Tcl_ObjType;
  contextType->name = "MlirContext";
  contextType->setFromAnyProc = contextTypeSetFromAnyProc;
  contextType->updateStringProc = contextTypeUpdateStringProc;
  contextType->dupIntRepProc = contextTypeDupIntRepProc;
  contextType->freeIntRepProc = contextTypeFreeIntRepProc;
  Tcl_RegisterObjType(contextType);

  Tcl_ObjType *operationType = new Tcl_ObjType;
  operationType->name = "MlirOperation";
  operationType->setFromAnyProc = operationTypeSetFromAnyProc;
  operationType->updateStringProc = operationTypeUpdateStringProc;
  operationType->dupIntRepProc = operationTypeDupIntRepProc;
  operationType->freeIntRepProc = operationTypeFreeIntRepProc;
  Tcl_RegisterObjType(operationType);

  // Register package
  if (Tcl_PkgProvide(interp, "Circt", "1.0") == TCL_ERROR) {
    return TCL_ERROR;
  }

  // Register commands
  Tcl_CreateObjCommand(interp, "loadCirctFile", loadFirMlirFile, NULL, NULL);
  return TCL_OK;
}

}
