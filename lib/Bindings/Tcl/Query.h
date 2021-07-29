#include <tcl.h>

#include "circt-c/Query.h"

int filterNodeTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj);
void filterNodeUpdateStringProc(Tcl_Obj *obj);
void filterNodeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup);
void filterNodeFreeIntRepProc(Tcl_Obj *obj);

int filterTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj);
void filterUpdateStringProc(Tcl_Obj *obj);
void filterDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup);
void filterFreeIntRepProc(Tcl_Obj *obj);

int tclFilter(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
