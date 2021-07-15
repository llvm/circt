#include <tcl.h>

#include "circt-c/Query.h"

int filterNodeTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj);
void filterNodeUpdateStringProc(Tcl_Obj *obj);
void filterNodeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup);
void filterNodeFreeIntRepProc(Tcl_Obj *obj);

