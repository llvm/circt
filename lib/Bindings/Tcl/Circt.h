#include <tcl.h>

int operationTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj);
void operationTypeUpdateStringProc(Tcl_Obj *obj);
void operationTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup);
void operationTypeFreeIntRepProc(Tcl_Obj *obj);
