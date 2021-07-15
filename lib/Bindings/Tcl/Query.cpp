#include <stdlib.h>
#include <string.h>

#include "Query.h"

int filterNodeTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  std::string bytes(obj->bytes, obj->length);
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
    obj->internalRep.otherValuePtr = CirctQueryNewRegexFilter(obj->bytes);
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
}

void filterNodeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  circt::query::FilterNode node = *((CirctQueryFilterNode) src->internalRep.otherValuePtr);
  dup->internalRep.otherValuePtr = std::malloc(sizeof(circt::query::FilterNode));
  *(CirctQueryFilterNode) dup->internalRep.otherValuePtr = node;
}

void filterNodeFreeIntRepProc(Tcl_Obj *obj) {
  std::free(obj->internalRep.otherValuePtr);
}
