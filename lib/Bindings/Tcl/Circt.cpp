#include "Circt.h"

#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/ESI.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/MSFT.h"
#include "circt-c/Dialect/SV.h"
#include "circt-c/Dialect/Seq.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir-c/Registration.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Operation.h"

int contextTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  auto *type = Tcl_GetObjType("MlirContext");
  if (obj->typePtr == type) {
    return TCL_OK;
  }

  if (obj->typePtr) {
    return TCL_ERROR;
  }

  obj->typePtr = type;
  MlirContext context = mlirContextCreate();
  unwrap(context)->loadDialect<circt::firrtl::FIRRTLDialect, circt::hw::HWDialect, circt::comb::CombDialect, circt::sv::SVDialect>();
  obj->internalRep.otherValuePtr = context.ptr;
  Tcl_InvalidateStringRep(obj);
  return TCL_OK;
}

void contextTypeUpdateStringProc(Tcl_Obj *obj) {
  const char* value = "<context>";
  size_t size = strlen(value) + 1;
  obj->bytes = Tcl_Alloc(size);
  memcpy(obj->bytes, value, size);
}

void contextTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  MlirContext context = mlirContextCreate();
  unwrap(context)->loadDialect<circt::firrtl::FIRRTLDialect, circt::hw::HWDialect, circt::comb::CombDialect, circt::sv::SVDialect>();
  dup->internalRep.otherValuePtr = context.ptr;
  Tcl_InvalidateStringRep(dup);
}

void contextTypeFreeIntRepProc(Tcl_Obj *obj) { }

int operationTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

void operationTypeUpdateStringProc(Tcl_Obj *obj) {
  std::string str;
  auto *op = unwrap((MlirOperation) { obj->internalRep.twoPtrValue.ptr1 });
  llvm::raw_string_ostream stream(str);
  op->print(stream);
  obj->length = str.length();
  obj->bytes = Tcl_Alloc(obj->length);
  memcpy(obj->bytes, str.c_str(), obj->length);
  obj->bytes[obj->length] = '\0';
}

void operationTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  auto *op = unwrap((MlirOperation) { src->internalRep.twoPtrValue.ptr1 })->clone();
  dup->internalRep.twoPtrValue.ptr1 = wrap(op).ptr;
  dup->internalRep.twoPtrValue.ptr2 = src->internalRep.twoPtrValue.ptr2;
  Tcl_IncrRefCount((Tcl_Obj *) src->internalRep.twoPtrValue.ptr2);
}

void operationTypeFreeIntRepProc(Tcl_Obj *obj) {
  auto *op = unwrap((MlirOperation) { obj->internalRep.twoPtrValue.ptr1 });
  op->erase();
  Tcl_DecrRefCount((Tcl_Obj *) obj->internalRep.twoPtrValue.ptr2);
}

