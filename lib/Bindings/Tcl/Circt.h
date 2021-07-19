#include <tcl.h>

int tclRegisterDialects(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);

int contextTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj);
void contextTypeUpdateStringProc(Tcl_Obj *obj);
void contextTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup);
void contextTypeFreeIntRepProc(Tcl_Obj *obj);
