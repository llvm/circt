#include <stdlib.h>
#include <tcl.h>

#include "Circt.h"
#include "Query.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

int returnErrorStr(Tcl_Interp *interp, const char *error) {
  Tcl_SetObjResult(interp, Tcl_NewStringObj(error, -1));
  return TCL_ERROR;
}

int loadFirMlirFile(mlir::MLIRContext *context, Tcl_Interp *interp, int objc,
                    Tcl_Obj *const objv[]) {
  if (objc != 3) {
    return returnErrorStr(interp, "usage: circt load [MLIR|FIR] [file]");
  }

  std::string errorMessage;
  auto input =
      mlir::openInputFile(llvm::StringRef(objv[2]->bytes), &errorMessage);

  if (!input) {
    return returnErrorStr(interp, errorMessage.c_str());
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, context);

  MlirOperation module;
  if (!strcmp(objv[1]->bytes, "MLIR")) {
    module = wrap(
        mlir::parseSourceFile(sourceMgr, context).release().getOperation());
  } else if (!strcmp(objv[1]->bytes, "FIR")) {
    // TODO
    return returnErrorStr(interp, "loading FIR files is unimplemented :(");
  } else {
    return TCL_ERROR;
  }

  if (mlirOperationIsNull(module)) {
    return returnErrorStr(interp, "error loading module");
  }

  auto *m = module.ptr;

  auto *obj = Tcl_NewObj();
  obj->typePtr = Tcl_GetObjType("MlirOperation");
  obj->internalRep.otherValuePtr = (void *)m;
  obj->length = 0;
  obj->bytes = nullptr;
  Tcl_SetObjResult(interp, obj);

  return TCL_OK;
}

int circtTclFunction(ClientData cdata, Tcl_Interp *interp, int objc,
                     Tcl_Obj *const objv[]) {
  if (objc < 2) {
    return returnErrorStr(interp, "usage: circt load");
  }

  auto *context = (mlir::MLIRContext *)cdata;

  if (!strcmp("load", objv[1]->bytes)) {
    return loadFirMlirFile(context, interp, objc - 1, objv + 1);
  }

  return returnErrorStr(interp, "usage: circt load");
}

void deleteContext(ClientData data) { delete (mlir::MLIRContext *)data; }

extern "C" {

int DLLEXPORT Circt_Init(Tcl_Interp *interp) {
  if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
    return TCL_ERROR;
  }

  // Register types
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
  auto *context = new mlir::MLIRContext;
  context->loadDialect<circt::hw::HWDialect, circt::comb::CombDialect,
                       circt::sv::SVDialect>();
  Tcl_CreateObjCommand(interp, "circt", circtTclFunction, context,
                       deleteContext);
  return TCL_OK;
}

}
