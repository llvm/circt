#include <tcl.h>

int contextTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj);
void contextTypeUpdateStringProc(Tcl_Obj *obj);
void contextTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup);
void contextTypeFreeIntRepProc(Tcl_Obj *obj);

int operationTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj);
void operationTypeUpdateStringProc(Tcl_Obj *obj);
void operationTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup);
void operationTypeFreeIntRepProc(Tcl_Obj *obj);

