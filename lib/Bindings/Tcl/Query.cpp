#include "Query.h"
#include "mlir/CAPI/IR.h"

int filterNodeTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  std::string bytes(obj->bytes);

  if (bytes == "*") {
    if (obj->typePtr) {
      obj->typePtr->freeIntRepProc(obj);
    }

    obj->typePtr = Tcl_GetObjType("FilterNode");
    obj->internalRep.otherValuePtr = CirctQueryNewGlobFilter();
    return TCL_OK;
  }

  if (bytes == "**") {
    if (obj->typePtr) {
      obj->typePtr->freeIntRepProc(obj);
    }

    obj->typePtr = Tcl_GetObjType("FilterNode");
    obj->internalRep.otherValuePtr = CirctQueryNewRecursiveGlobFilter();
    return TCL_OK;
  }

  if (bytes[0] == '/' && bytes[bytes.size() - 1] == '/') {
    if (obj->typePtr) {
      obj->typePtr->freeIntRepProc(obj);
    }

    obj->typePtr = Tcl_GetObjType("FilterNode");
    char buffer[obj->length - 1];
    buffer[obj->length - 2] = '\0';
    memcpy(buffer, obj->bytes, obj->length - 2);
    obj->internalRep.otherValuePtr = CirctQueryNewRegexFilter(buffer);
    return TCL_OK;
  }

  if (bytes.find("::") == std::string::npos) {
    if (obj->typePtr) {
      obj->typePtr->freeIntRepProc(obj);
    }

    obj->typePtr = Tcl_GetObjType("FilterNode");
    obj->internalRep.otherValuePtr = CirctQueryNewLiteralFilter(obj->bytes);
    return TCL_OK;
  }

  return TCL_ERROR;
}

void filterNodeUpdateStringProc(Tcl_Obj *obj) {
  obj->bytes = Tcl_Alloc(1);
  obj->bytes[0] = '\0';
  obj->length = 0;
}

void filterNodeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
}

void filterNodeFreeIntRepProc(Tcl_Obj *obj) {
}

int filterTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

void filterUpdateStringProc(Tcl_Obj *obj) {
}

void filterDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
}

void filterFreeIntRepProc(Tcl_Obj *obj) {
}

int tclFilter(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
  return TCL_OK;
}
