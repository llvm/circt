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
  circt::query::FilterNode node = *((CirctQueryFilterNode) src->internalRep.otherValuePtr);
  dup->internalRep.otherValuePtr = std::malloc(sizeof(circt::query::FilterNode));
  *(CirctQueryFilterNode) dup->internalRep.otherValuePtr = node;
}

void filterNodeFreeIntRepProc(Tcl_Obj *obj) {
  if (obj->internalRep.otherValuePtr) {
    delete (CirctQueryFilterNode) obj->internalRep.otherValuePtr;
  }
}

int filterTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

void filterUpdateStringProc(Tcl_Obj *obj) {
  std::string str;
  size_t count = ((CirctQueryFilter) obj->internalRep.twoPtrValue.ptr1)->size();

  for (size_t i = 0; i < count; i++) {
    if (i != 0) {
      str.append("::");
    }
    str.append(((Tcl_Obj**) obj->internalRep.twoPtrValue.ptr2)[i]->bytes);
  }

  auto *bytes = Tcl_Alloc(str.size() + 1);
  bytes[str.size()] = '\0';
  memcpy(bytes, str.data(), str.size());
  obj->bytes = bytes;
  obj->length = str.size();
}

void filterDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  size_t count = ((CirctQueryFilter) src->internalRep.twoPtrValue.ptr1)->size();
  dup->internalRep.twoPtrValue.ptr1 = (void*) new circt::query::Filter;
  *(CirctQueryFilter) dup->internalRep.twoPtrValue.ptr1 = *(CirctQueryFilter) src->internalRep.twoPtrValue.ptr1;

  for (size_t i = 0; i < count; i++) {
    ((Tcl_Obj**) dup->internalRep.twoPtrValue.ptr2)[i] = Tcl_DuplicateObj(((Tcl_Obj**) src->internalRep.twoPtrValue.ptr2)[i]);
  }
}

void filterFreeIntRepProc(Tcl_Obj *obj) {
  size_t count = ((CirctQueryFilter) obj->internalRep.twoPtrValue.ptr1)->size();

  for (size_t i = 0; i < count; i++) {
    Tcl_DecrRefCount(((Tcl_Obj**) obj->internalRep.twoPtrValue.ptr2)[i]);
  }

  delete (CirctQueryFilter) obj->internalRep.twoPtrValue.ptr1;
  delete[] (Tcl_Obj*) obj->internalRep.twoPtrValue.ptr2;
}

int tclFilter(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
  if (objc != 3) {
    return TCL_ERROR;
  }

  auto *filter = (circt::query::Filter *) objv[1]->internalRep.otherValuePtr;
  auto op = (MlirOperation) { objv[2]->internalRep.otherValuePtr };

  auto *result = CirctQueryFilterFromRoot(filter, op);

  return TCL_OK;
}
