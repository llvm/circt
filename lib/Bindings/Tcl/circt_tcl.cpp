#include <stdlib.h>
#include <tcl.h>

#include "circt-c/Query.h"
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

static CirctQueryFilterType createFilterType(Tcl_Obj *obj) {
  CirctQueryFilterType type;
  auto *str = Tcl_GetString(obj);
  size_t length = obj->length;
  if (!strcmp(str, "*")) {
    type = circtQueryNewGlobFilterType();
  } else if (!strcmp(str, "**")) {
    type = circtQueryNewRecursiveGlobFilterType();
  } else if (str[0] == '/' && str[length - 1] == '/' && length > 2) {
    char buffer[length - 1];
    memcpy(buffer, str + 1, length - 1);
    buffer[length - 2] = '\0';
    type = circtQueryNewRegexFilterType(buffer);
  } else {
    for (size_t i = 0; i < length; i++) {
      char c = str[i];
      if (!(('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || c == '_')) {
        return {nullptr};
      }
    }

    type = circtQueryNewLiteralFilterType(str);
  }

  return type;
}

static int filterTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  auto type = createFilterType(obj);
  if (type.ptr == nullptr) {
    return TCL_ERROR;
  }

  CirctQueryFilter filter = circtQueryNewNameFilter(type);

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

  auto *key = Tcl_GetString(objv[1]);
  auto type = createFilterType(objv[2]);
  if (type.ptr == nullptr) {
    return returnErrorStr(interp, "invalid filter type");
  }

  auto filter = circtQueryNewAttributeFilter(key, type);
  auto *result = Tcl_NewObj();
  result->typePtr = Tcl_GetObjType("Filter");
  result->internalRep.otherValuePtr = filter.ptr;
  Tcl_SetObjResult(interp, result);
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

static int createUsageFilter(ClientData cdata, Tcl_Interp *interp,
                                 int objc, Tcl_Obj *const objv[]) {
  if (objc != 2) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: usage [filter]");
    return TCL_ERROR;
  }

  auto *obj = objv[1];
  if (Tcl_IsShared(obj)) {
    obj = Tcl_DuplicateObj(obj);
  }

  auto *type = Tcl_GetObjType("Filter");
  if (Tcl_ConvertToType(interp, obj, type) != TCL_OK) {
    return returnErrorStr(interp, "expected filter");
  }

  auto *result = Tcl_NewObj();
  result->typePtr = type;
  result->internalRep.otherValuePtr = circtQueryNewUsageFilter({obj->internalRep.otherValuePtr}).ptr;
  obj->internalRep.otherValuePtr = nullptr;
  obj->typePtr = nullptr;
  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}

static int operationTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

static void operationTypeUpdateStringProc(Tcl_Obj *obj) {
  std::string str;
  auto *op = unwrap((MlirOperation){obj->internalRep.twoPtrValue.ptr1});
  llvm::raw_string_ostream stream(str);
  op->print(stream);
  obj->length = str.length();
  obj->bytes = Tcl_Alloc(obj->length);
  memcpy(obj->bytes, str.c_str(), obj->length);
  obj->bytes[obj->length] = '\0';
}

static void operationTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  auto *op = unwrap((MlirOperation){src->internalRep.twoPtrValue.ptr1});
  MlirOperation result;
  if (auto mod = llvm::dyn_cast_or_null<mlir::ModuleOp>(op)) {
    result = wrap(mod.clone().getOperation());
    dup->internalRep.twoPtrValue.ptr2 = circtQueryNewFilterData(result).ptr;
  } else {
    result = wrap(op);
    auto *module = (Tcl_Obj *) src->internalRep.twoPtrValue.ptr2;
    Tcl_IncrRefCount(module);
    dup->internalRep.twoPtrValue.ptr2 = module;
  }

  dup->internalRep.twoPtrValue.ptr1 = result.ptr;
}

static void operationTypeFreeIntRepProc(Tcl_Obj *obj) {
  auto *op = unwrap((MlirOperation){obj->internalRep.twoPtrValue.ptr1});
  if (auto mod = llvm::dyn_cast_or_null<mlir::ModuleOp>(op)) {
    mod.erase();
    circtQueryDeleteFilterData((CirctQueryFilterData){obj->internalRep.twoPtrValue.ptr2});
  } else {
    Tcl_DecrRefCount((Tcl_Obj *) obj->internalRep.twoPtrValue.ptr2);
  }
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
    return returnErrorStr(interp, "unsupported file type");

  if (mlirOperationIsNull(module))
    return returnErrorStr(interp, "error loading module");

  auto *m = module.ptr;

  auto *obj = Tcl_NewObj();
  obj->typePtr = Tcl_GetObjType("MlirOperation");
  obj->internalRep.twoPtrValue.ptr1 = m;
  obj->internalRep.twoPtrValue.ptr2 = circtQueryNewFilterData(module).ptr;
  Tcl_InvalidateStringRep(obj);
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
  Tcl_Obj **objs = nullptr;
  int length = 0;
  auto *type = Tcl_GetObjType("MlirOperation");
  if (Tcl_ConvertToType(interp, objv[2], type) == TCL_OK) {
    objs = (Tcl_Obj **) &objv[2];
    length = 1;
  } else if (Tcl_ListObjGetElements(interp, objv[2], &length, &objs) != TCL_OK) {
    return returnErrorStr(interp, "expected operation or list of operations");
  }

  auto *list = Tcl_NewListObj(0, nullptr);
  for (int i = 0; i < length; ++i) {
    if (Tcl_ConvertToType(interp, objs[i], type) != TCL_OK) {
      return returnErrorStr(interp, "expected operation or list of operations");
    }

    auto root = (MlirOperation){objs[i]->internalRep.twoPtrValue.ptr1};
    Tcl_Obj *module;
    if (auto _ = llvm::dyn_cast_or_null<mlir::ModuleOp>(unwrap(root))) {
      module = objs[i];
    } else {
      module = (Tcl_Obj *) objs[i]->internalRep.twoPtrValue.ptr2;
    }

    auto data = (CirctQueryFilterData){module->internalRep.twoPtrValue.ptr2};
    auto result = circtQueryFilterFromRoot(filter, root, data);
    MlirOperation op;
    for (size_t j = 0; !mlirOperationIsNull(op = circtQueryGetFromFilterResult(result, j)); ++j) {
      auto *obj = Tcl_NewObj();
      obj->typePtr = type;
      obj->internalRep.twoPtrValue.ptr1 = op.ptr;
      obj->internalRep.twoPtrValue.ptr2 = module;
      Tcl_IncrRefCount(module);
      Tcl_InvalidateStringRep(obj);

      int length = 0;
      Tcl_Obj **listRaw = nullptr;
      if (Tcl_ListObjGetElements(interp, list, &length, &listRaw) != TCL_OK) {
        return TCL_ERROR;
      }

      auto *unwrappedOp = unwrap(op);
      bool contained = false;
      for (int k = 0; k < length; ++k) {
        if (unwrap((MlirOperation){list[k].internalRep.twoPtrValue.ptr1}) == unwrappedOp) {
          contained = true;
          break;
        }
      }

      if (!contained && Tcl_ListObjAppendElement(interp, list, obj) != TCL_OK) {
        circtQueryDeleteFilterResult(result);
        return TCL_ERROR;
      }
    }
  }
  Tcl_SetObjResult(interp, list);

  return TCL_OK;
}

static int dumpModuleName(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
  if (objc != 2) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: modname [module]");
    return TCL_ERROR;
  }

  if (Tcl_ConvertToType(interp, objv[1], Tcl_GetObjType("MlirOperation")) == TCL_ERROR) {
    return returnErrorStr(interp, "expected operation");
  }

  auto *op = unwrap((MlirOperation){objv[1]->internalRep.twoPtrValue.ptr1});
  auto *result = mlir::TypeSwitch<mlir::Operation *, Tcl_Obj *>(op)
    .Case<circt::hw::HWModuleOp, circt::hw::HWModuleExternOp>([&](auto &mod) {
      auto name = mod.getNameAttr().getValue();
      return Tcl_NewStringObj(name.data(), name.size());
    })
    .Default([&](auto &op) {
      return Tcl_NewStringObj("", 0);
    });

  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}

static int circtTclFunction(ClientData cdata, Tcl_Interp *interp, int objc,
                            Tcl_Obj *const objv[]) {
  if (objc < 2) {
    Tcl_WrongNumArgs(interp, objc, objv, "usage: circt [load|query|get]");
    return TCL_ERROR;
  }

  auto *context = (mlir::MLIRContext *) cdata;
  auto *str = Tcl_GetString(objv[1]);

  if (!strcmp("load", str))
    return loadFirMlirFile(context, interp, objc - 1, objv + 1);

  if (!strcmp("query", str))
    return filter(cdata, interp, objc - 1, objv + 1);

  if (!strcmp("get", str)) {
    if (objc < 3) {
      Tcl_WrongNumArgs(interp, objc, objv, "usage: circt get [modname|attrs]");
      return TCL_ERROR;
    }

    auto *str = Tcl_GetString(objv[2]);
    if (!strcmp("modname", str))
      return dumpModuleName(cdata, interp, objc - 2, objv + 2);

    return returnErrorStr(interp, "usage: circt get [modname|attrs]");
  }

  return returnErrorStr(interp, "usage: circt [load|query|get]");
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
  Tcl_CreateObjCommand(interp, "attr", createAttributeFilter, NULL, NULL);
  Tcl_CreateObjCommand(interp, "op", createOpFilter, NULL, NULL);
  Tcl_CreateObjCommand(interp, "usage", createUsageFilter, NULL, NULL);
  return TCL_OK;
}
}
