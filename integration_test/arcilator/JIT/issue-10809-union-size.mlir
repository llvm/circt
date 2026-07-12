// RUN: arcilator %s --run | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// Issue #10809

// CHECK: 12345678

!union_ty = !hw.union<r: !hw.struct<f0: i32, f1: i64>>

func.func @tick(%u: !union_ty) -> (i32, i64) {
  %s = hw.union_extract %u["r"] : !union_ty
  %counter, %delta = hw.struct_explode %s : !hw.struct<f0: i32, f1: i64>
  return %counter, %delta : i32, i64
}

func.func @entry() {
  %endl = sim.fmt.literal "\n"
  %c = hw.constant 0x12345678 : i32
  %d = hw.constant 1000000 : i64
  %s = hw.struct_create (%c, %d) : !hw.struct<f0: i32, f1: i64>
  %u = hw.union_create "r", %s : !union_ty
  %r:2 = func.call @tick(%u) : (!union_ty) -> (i32, i64)
  %h = sim.fmt.hex %r#0, isUpper false : i32
  %m = sim.fmt.concat (%h, %endl)
  sim.proc.print %m
  return
}
