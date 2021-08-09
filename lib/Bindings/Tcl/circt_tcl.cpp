#include <stdlib.h>
#include <tcl.h>

#include "Query.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

static int returnErrorStr(Tcl_Interp *interp, const char *error) {
  Tcl_SetObjResult(interp, Tcl_NewStringObj(error, -1));
  return TCL_ERROR;
}

static int filterTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  auto *str = Tcl_GetString(obj);
  size_t length = obj->length;
  CirctQueryFilter filter;
  if (!strcmp(str, "*")) {
    filter = circtQueryNewNameFilter(circtQueryNewGlobFilterType());
  } else if (!strcmp(str, "**")) {
    filter = circtQueryNewNameFilter(circtQueryNewRecursiveGlobFilterType());
  } else if (str[0] == '/' && str[length - 1] == '/') {
    char buffer[length - 1];
    memcpy(buffer, str + 1, length - 1);
    buffer[length - 2] = '\0';
    filter = circtQueryNewNameFilter(circtQueryNewRegexFilterType(buffer));
  } else {
    for (size_t i = 0; i < length; i++) {
      char c = str[i];
      if (!(('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || c == '_')) {
        return TCL_ERROR;
      }
    }

    filter = circtQueryNewNameFilter(circtQueryNewLiteralFilterType(str));
  }

  if (obj->typePtr != nullptr) {
    obj->typePtr->freeIntRepProc(obj);
  }

  obj->typePtr = Tcl_GetObjType("Filter");
  obj->internalRep.otherValuePtr = filter.ptr;
  return TCL_OK;
}

static void filterTypeUpdateStringProc(Tcl_Obj *obj) {
  // TODO
  obj->bytes = Tcl_Alloc(1);
  obj->bytes[0] = '\0';
  obj->length = 0;
}

static void filterTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  dup->internalRep.otherValuePtr = circtQueryCloneFilter((CirctQueryFilter){src->internalRep.otherValuePtr}).ptr;
}

static void filterTypeFreeIntRepProc(Tcl_Obj *obj) {
  circtQueryDeleteFilter((CirctQueryFilter){obj->internalRep.otherValuePtr});
}

static int createAndOrFilter(Tcl_Interp *interp, int objc, Tcl_Obj *const objv[],
                             const char* usage, CirctQueryFilter (*createFunc)(size_t, CirctQueryFilter *)) {
  if (objc <= 1) {
    Tcl_WrongNumArgs(interp, objc, objv, usage);
    return TCL_ERROR;
  }

  auto *type = Tcl_GetObjType("Filter");
  if (objc == 2) {
    if (Tcl_ConvertToType(interp, objv[1], type) == TCL_OK) {
      Tcl_SetObjResult(interp, objv[1]);
      return TCL_OK;
    }

    return returnErrorStr(interp, "expected filter");
  }

  CirctQueryFilter filters[objc - 1];
  for (int i = 1; i < objc; ++i) {
    auto *obj = objv[i];
    if (Tcl_IsShared(objv[i])) {
      obj = Tcl_DuplicateObj(objv[i]);
    }

    if (Tcl_ConvertToType(interp, objv[i], type) == TCL_OK) {
      void *ptr = objv[i]->internalRep.otherValuePtr;
      objv[i]->internalRep.otherValuePtr = nullptr;
      objv[i]->typePtr = nullptr;
      filters[i - 1] = { ptr };
    } else {
      for (int j = 0; j < i - 1; ++j) {
        circtQueryDeleteFilter(filters[j]);
      }

      return returnErrorStr(interp, "expected filter");
    }
  }

  auto filter = createFunc(objc - 1, filters);
  auto *result = Tcl_NewObj();
  result->typePtr = type;
  result->internalRep.otherValuePtr = filter.ptr;
  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}

static int createOrFilter(ClientData cdata, Tcl_Interp *interp,
                           int objc, Tcl_Obj *const objv[]) {
  return createAndOrFilter(interp, objc, objv, "usage: or [filter]+", circtQueryNewOrFilter);
}

static int createAndFilter(ClientData cdata, Tcl_Interp *interp,
                           int objc, Tcl_Obj *const objv[]) {
  return createAndOrFilter(interp, objc, objv, "usage: and [filter]+", circtQueryNewAndFilter);
}

static int createInstanceFilter(ClientData cdata, Tcl_Interp *interp,
                           int objc, Tcl_Obj *const objv[]) {
  if (objc <= 1) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: inst [filter]+");
    return TCL_ERROR;
  }

  auto *type = Tcl_GetObjType("Filter");
  if (objc == 2) {
    if (Tcl_ConvertToType(interp, objv[1], type) == TCL_OK) {
      Tcl_SetObjResult(interp, objv[1]);
      return TCL_OK;
    }

    return returnErrorStr(interp, "expected filter");
  }

  auto filter = (CirctQueryFilter){nullptr};
  for (int i = objc - 1; i >= 1; --i) {
    auto *obj = objv[i];
    if (Tcl_IsShared(objv[i])) {
      obj = Tcl_DuplicateObj(objv[i]);
    }

    if (Tcl_ConvertToType(interp, obj, type) == TCL_OK) {
      void *ptr = obj->internalRep.otherValuePtr;
      obj->internalRep.otherValuePtr = nullptr;
      obj->typePtr = nullptr;

      if (filter.ptr == nullptr) {
        filter.ptr = ptr;
      } else {
        filter = circtQueryNewInstanceFilter((CirctQueryFilter){ptr}, filter);
      }
    } else {
      if (filter.ptr != nullptr) {
        circtQueryDeleteFilter(filter);
      }

      return returnErrorStr(interp, "expected filter");
    }
  }

  auto *result = Tcl_NewObj();
  result->typePtr = type;
  result->internalRep.otherValuePtr = filter.ptr;
  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}

static int createAttributeFilter(ClientData cdata, Tcl_Interp *interp,
                                 int objc, Tcl_Obj *const objv[]) {
  if (objc != 3) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: attr [key] [filter type]");
    return TCL_ERROR;
  }

  return TCL_OK;
}

static int createOpFilter(ClientData cdata, Tcl_Interp *interp,
                                 int objc, Tcl_Obj *const objv[]) {
  if (objc != 2) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: op [key]");
    return TCL_ERROR;
  }

  auto *op = Tcl_GetString(objv[1]);
  auto filter = circtQueryNewOperatorFilter(circtQueryNewLiteralFilterType(op));
  auto *result = Tcl_NewObj();
  result->typePtr = Tcl_GetObjType("Filter");
  result->internalRep.otherValuePtr = filter.ptr;
  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}

static int operationTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

static void operationTypeUpdateStringProc(Tcl_Obj *obj) {
  std::string str;
  auto *op = unwrap((MlirOperation){obj->internalRep.otherValuePtr});
  llvm::raw_string_ostream stream(str);
  op->print(stream);
  obj->length = str.length();
  obj->bytes = Tcl_Alloc(obj->length);
  memcpy(obj->bytes, str.c_str(), obj->length);
  obj->bytes[obj->length] = '\0';
}

static void operationTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  auto *op = unwrap((MlirOperation){src->internalRep.otherValuePtr})->clone();
  dup->internalRep.otherValuePtr = wrap(op).ptr;
}

static void operationTypeFreeIntRepProc(Tcl_Obj *obj) {
  auto *op = unwrap((MlirOperation){obj->internalRep.otherValuePtr});
  op->erase();
}

static int loadFirMlirFile(mlir::MLIRContext *context, Tcl_Interp *interp,
                           int objc, Tcl_Obj *const objv[]) {
  if (objc != 3) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: circt load [MLIR|FIR] [file]");
    return TCL_ERROR;
  }

  std::string errorMessage;
  auto input = mlir::openInputFile(llvm::StringRef(Tcl_GetString(objv[2])),
                                   &errorMessage);

  if (!input)
    return returnErrorStr(interp, errorMessage.c_str());

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, context);

  MlirOperation module;
  if (!strcmp(Tcl_GetString(objv[1]), "MLIR"))
    module = wrap(
        mlir::parseSourceFile(sourceMgr, context).release().getOperation());
  else if (!strcmp(Tcl_GetString(objv[1]), "FIR"))
    // TODO
    return returnErrorStr(interp, "loading FIR files is unimplemented :(");
  else
    return TCL_ERROR;

  if (mlirOperationIsNull(module))
    return returnErrorStr(interp, "error loading module");

  auto *m = module.ptr;

  auto *obj = Tcl_NewObj();
  obj->typePtr = Tcl_GetObjType("MlirOperation");
  obj->internalRep.otherValuePtr = (void *)m;
  obj->length = 0;
  obj->bytes = nullptr;
  Tcl_SetObjResult(interp, obj);

  return TCL_OK;
}

static int filter(ClientData cdata, Tcl_Interp *interp,
                           int objc, Tcl_Obj *const objv[]) {
  if (objc != 3) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: circt query [filter] [operation|filter result]");
    return TCL_ERROR;
  }

  if (Tcl_ConvertToType(interp, objv[1], Tcl_GetObjType("Filter")) == TCL_ERROR) {
    return returnErrorStr(interp, "expected filter");
  }

  auto filter = (CirctQueryFilter){objv[1]->internalRep.otherValuePtr};
  if (Tcl_ConvertToType(interp, objv[2], Tcl_GetObjType("MlirOperation")) == TCL_OK) {
    auto root = (MlirOperation){objv[2]->internalRep.otherValuePtr};
    auto result = circtQueryFilterFromRoot(filter, root);
    MlirOperation op;
    auto *type = Tcl_GetObjType("MlirOperation");
    auto *list = Tcl_NewListObj(0, nullptr);
    for (size_t i = 0; !mlirOperationIsNull(op = circtQueryGetFromFilterResult(result, i)); ++i) {
      auto *obj = Tcl_NewObj();
      obj->typePtr = type;
      obj->internalRep.otherValuePtr = wrap(unwrap(op)->clone()).ptr;
      obj->bytes = nullptr;
      obj->length = 0;
      if (Tcl_ListObjAppendElement(interp, list, obj) != TCL_OK) {
        circtQueryDeleteFilterResult(result);
        return returnErrorStr(interp, "something went wrong when appending to list");
      }

      Tcl_SetObjResult(interp, list);
    }
  } else {
    return returnErrorStr(interp, "expected operation or list of operations");
  }

  return TCL_OK;
}

static int circtTclFunction(ClientData cdata, Tcl_Interp *interp, int objc,
                            Tcl_Obj *const objv[]) {
  if (objc < 2) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: circt [load|query]");
    return TCL_ERROR;
  }

  auto *context = (mlir::MLIRContext *) cdata;
  auto *str = Tcl_GetString(objv[1]);

  if (!strcmp("load", str))
    return loadFirMlirFile(context, interp, objc - 1, objv + 1);

  if (!strcmp("query", str))
    return filter(cdata, interp, objc - 1, objv + 1);

  return returnErrorStr(interp, "usage: circt load");
}

static void deleteContext(ClientData data) { delete (mlir::MLIRContext *)data; }

extern "C" {

int DLLEXPORT Circt_Init(Tcl_Interp *interp) {
  if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL)
    return TCL_ERROR;

  // Register types
  auto *operationType = new Tcl_ObjType;
  operationType->name = "MlirOperation";
  operationType->setFromAnyProc = operationTypeSetFromAnyProc;
  operationType->updateStringProc = operationTypeUpdateStringProc;
  operationType->dupIntRepProc = operationTypeDupIntRepProc;
  operationType->freeIntRepProc = operationTypeFreeIntRepProc;
  Tcl_RegisterObjType(operationType);

  auto *filterType = new Tcl_ObjType;
  filterType->name = "Filter";
  filterType->setFromAnyProc = filterTypeSetFromAnyProc;
  filterType->updateStringProc = filterTypeUpdateStringProc;
  filterType->dupIntRepProc = filterTypeDupIntRepProc;
  filterType->freeIntRepProc = filterTypeFreeIntRepProc;
  Tcl_RegisterObjType(filterType);

  // Register package
  if (Tcl_PkgProvide(interp, "Circt", "1.0") == TCL_ERROR)
    return TCL_ERROR;

  // Register commands
  auto *context = new mlir::MLIRContext;
  context->loadDialect<circt::hw::HWDialect, circt::comb::CombDialect,
                       circt::sv::SVDialect>();
  Tcl_CreateObjCommand(interp, "circt", circtTclFunction, context,
                       deleteContext);
  Tcl_CreateObjCommand(interp, "inst", createInstanceFilter, NULL, NULL);
  Tcl_CreateObjCommand(interp, "and", createAndFilter, NULL, NULL);
  Tcl_CreateObjCommand(interp, "or", createOrFilter, NULL, NULL);
  Tcl_CreateObjCommand(interp, "op", createOpFilter, NULL, NULL);
  return TCL_OK;
}
}
