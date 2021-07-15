#include <stdlib.h>
#include <tcl.h>

#include "Query.h"

static int Hello_Cmd(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[])
{
  Tcl_SetObjResult(interp, Tcl_NewStringObj("Hello, World!", -1));
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
 return TCL_OK;
}

}
