// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

moore.class.classdecl @C {
}

// CHECK-LABEL: func.func @ClassToChandle(
// CHECK-SAME: %[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: return %[[ARG]] : !llvm.ptr
func.func @ClassToChandle(%obj: !moore.class<@C>) -> !moore.chandle {
  %handle = moore.conversion %obj : !moore.class<@C> -> !moore.chandle
  return %handle : !moore.chandle
}

// CHECK-LABEL: func.func @ChandleToClass(
// CHECK-SAME: %[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: return %[[ARG]] : !llvm.ptr
func.func @ChandleToClass(%handle: !moore.chandle) -> !moore.class<@C> {
  %obj = moore.conversion %handle : !moore.chandle -> !moore.class<@C>
  return %obj : !moore.class<@C>
}

// CHECK-LABEL: func.func @ClassRefToChandleRef(
// CHECK-SAME: %[[ARG:.*]]: !llhd.ref<!llvm.ptr>) -> !llhd.ref<!llvm.ptr>
// CHECK-NEXT: return %[[ARG]] : !llhd.ref<!llvm.ptr>
func.func @ClassRefToChandleRef(%obj: !moore.ref<class<@C>>) -> !moore.ref<chandle> {
  %handle = moore.conversion %obj : !moore.ref<class<@C>> -> !moore.ref<chandle>
  return %handle : !moore.ref<chandle>
}

// CHECK-LABEL: func.func @ChandleRefToClassRef(
// CHECK-SAME: %[[ARG:.*]]: !llhd.ref<!llvm.ptr>) -> !llhd.ref<!llvm.ptr>
// CHECK-NEXT: return %[[ARG]] : !llhd.ref<!llvm.ptr>
func.func @ChandleRefToClassRef(%handle: !moore.ref<chandle>) -> !moore.ref<class<@C>> {
  %obj = moore.conversion %handle : !moore.ref<chandle> -> !moore.ref<class<@C>>
  return %obj : !moore.ref<class<@C>>
}

// CHECK-NOT: moore.conversion
