#include <stdlib.h>
#include <tcl.h>

#include "Query.h"

static int Hello_Cmd(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[])
{
  Tcl_SetObjResult(interp, Tcl_NewStringObj("Hello, World!", -1));
  return TCL_OK;
}

static int createTclFilter(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
  Tcl_Obj *nodes[objc - 1];

  if (objc == 1) {
    return TCL_ERROR;
  }

  for (int i = 1; i < objc; i++) {
    auto *obj = objv[i];
    if (Tcl_IsShared(obj)) {
      obj = Tcl_DuplicateObj(obj);
    }

    if (Tcl_ConvertToType(interp, obj, Tcl_GetObjType("FilterNode")) != TCL_OK) {
      return TCL_ERROR;
    }

    nodes[i - 1] = obj;
  }
  Tcl_SetObjResult(interp, nodes[0]);
  return TCL_OK;
}

extern "C" {

int DLLEXPORT Circt_Init(Tcl_Interp *interp) {
 if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
    return TCL_ERROR;
 }

  Tcl_ObjType *filterNodeType = (Tcl_ObjType *) malloc(sizeof(Tcl_ObjType));
  filterNodeType->name = "FilterNode";
  filterNodeType->setFromAnyProc = filterNodeTypeSetFromAnyProc;
  filterNodeType->updateStringProc = filterNodeUpdateStringProc;
  filterNodeType->dupIntRepProc = filterNodeDupIntRepProc;
  filterNodeType->freeIntRepProc = filterNodeFreeIntRepProc;
  Tcl_RegisterObjType(filterNodeType);

 if (Tcl_PkgProvide(interp, "Hello", "1.0") == TCL_ERROR) {
    return TCL_ERROR;
 }
 Tcl_CreateObjCommand(interp, "hello", Hello_Cmd, NULL, NULL);
 Tcl_CreateObjCommand(interp, "filter", createTclFilter, NULL, NULL);
 return TCL_OK;
}

}
