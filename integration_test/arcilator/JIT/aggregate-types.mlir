// RUN: arcilator %s --run --jit-entry=main | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// End-to-end JIT execution test for aggregate element types in arrays,
// validating support for HW structs and HW arrays in arc.arrayref.alloc
// (including zero-initialization of nested aggregate attributes).

// CHECK:      struct_v0 = 0
// CHECK-NEXT: struct_val0 = 0000
// CHECK-NEXT: struct_v1 = 1
// CHECK-NEXT: struct_val1 = 1234
// CHECK-NEXT: arr_a0_e0 = 00
// CHECK-NEXT: arr_a0_e1 = 00
// CHECK-NEXT: arr_a1_e0 = 22
// CHECK-NEXT: arr_a1_e1 = 11
// CHECK-NEXT: nested_v0 = 0
// CHECK-NEXT: nested_d0_e0 = 00
// CHECK-NEXT: nested_v1 = 1
// CHECK-NEXT: nested_d1_e0 = bb

hw.module @ArrayOfStruct(out valid0: i1, out val0: i16, out valid1: i1, out val1: i16) {
  // Array of structs with zero initialization.
  %arr_zero = hw.aggregate_constant [[false, 0 : i16], [false, 0 : i16]] : !hw.array<2x!hw.struct<valid: i1, val: i16>>
  %c_true = hw.constant true
  %c_val = hw.constant 0x1234 : i16
  %new_struct = hw.struct_create (%c_true, %c_val) : !hw.struct<valid: i1, val: i16>
  %c_idx0 = hw.constant 0 : i1
  %c_idx1 = hw.constant 1 : i1
  %arr_updated = hw.array_inject %arr_zero[%c_idx1], %new_struct : !hw.array<2x!hw.struct<valid: i1, val: i16>>, i1

  %elem0 = hw.array_get %arr_updated[%c_idx0] : !hw.array<2x!hw.struct<valid: i1, val: i16>>, i1
  %v0 = hw.struct_extract %elem0["valid"] : !hw.struct<valid: i1, val: i16>
  %val0_res = hw.struct_extract %elem0["val"] : !hw.struct<valid: i1, val: i16>

  %elem1 = hw.array_get %arr_updated[%c_idx1] : !hw.array<2x!hw.struct<valid: i1, val: i16>>, i1
  %v1 = hw.struct_extract %elem1["valid"] : !hw.struct<valid: i1, val: i16>
  %val1_res = hw.struct_extract %elem1["val"] : !hw.struct<valid: i1, val: i16>

  hw.output %v0, %val0_res, %v1, %val1_res : i1, i16, i1, i16
}

hw.module @ArrayOfArray(out a0_e0: i8, out a0_e1: i8, out a1_e0: i8, out a1_e1: i8) {
  // Array of arrays with zero initialization.
  %arr_zero = hw.aggregate_constant [[0 : i8, 0 : i8], [0 : i8, 0 : i8]] : !hw.array<2x!hw.array<2xi8>>
  %new_subarr = hw.aggregate_constant [0x11 : i8, 0x22 : i8] : !hw.array<2xi8>
  %c_idx0 = hw.constant 0 : i1
  %c_idx1 = hw.constant 1 : i1
  %arr_updated = hw.array_inject %arr_zero[%c_idx1], %new_subarr : !hw.array<2x!hw.array<2xi8>>, i1

  %elem0 = hw.array_get %arr_updated[%c_idx0] : !hw.array<2x!hw.array<2xi8>>, i1
  %a0_e0_res = hw.array_get %elem0[%c_idx0] : !hw.array<2xi8>, i1
  %a0_e1_res = hw.array_get %elem0[%c_idx1] : !hw.array<2xi8>, i1

  %elem1 = hw.array_get %arr_updated[%c_idx1] : !hw.array<2x!hw.array<2xi8>>, i1
  %a1_e0_res = hw.array_get %elem1[%c_idx0] : !hw.array<2xi8>, i1
  %a1_e1_res = hw.array_get %elem1[%c_idx1] : !hw.array<2xi8>, i1

  hw.output %a0_e0_res, %a0_e1_res, %a1_e0_res, %a1_e1_res : i8, i8, i8, i8
}

hw.module @ArrayOfStructWithArray(out valid0: i1, out data0_e0: i8, out valid1: i1, out data1_e0: i8) {
  // Array of structs containing nested arrays with zero initialization.
  %arr_zero = hw.aggregate_constant [[false, [0 : i8, 0 : i8]], [false, [0 : i8, 0 : i8]]] : !hw.array<2x!hw.struct<valid: i1, data: !hw.array<2xi8>>>
  %c_true = hw.constant true
  %new_subarr = hw.aggregate_constant [0xAA : i8, 0xBB : i8] : !hw.array<2xi8>
  %new_struct = hw.struct_create (%c_true, %new_subarr) : !hw.struct<valid: i1, data: !hw.array<2xi8>>
  %c_idx0 = hw.constant 0 : i1
  %c_idx1 = hw.constant 1 : i1
  %arr_updated = hw.array_inject %arr_zero[%c_idx1], %new_struct : !hw.array<2x!hw.struct<valid: i1, data: !hw.array<2xi8>>>, i1

  %elem0 = hw.array_get %arr_updated[%c_idx0] : !hw.array<2x!hw.struct<valid: i1, data: !hw.array<2xi8>>>, i1
  %v0 = hw.struct_extract %elem0["valid"] : !hw.struct<valid: i1, data: !hw.array<2xi8>>
  %d0 = hw.struct_extract %elem0["data"] : !hw.struct<valid: i1, data: !hw.array<2xi8>>
  %d0_e0 = hw.array_get %d0[%c_idx0] : !hw.array<2xi8>, i1

  %elem1 = hw.array_get %arr_updated[%c_idx1] : !hw.array<2x!hw.struct<valid: i1, data: !hw.array<2xi8>>>, i1
  %v1 = hw.struct_extract %elem1["valid"] : !hw.struct<valid: i1, data: !hw.array<2xi8>>
  %d1 = hw.struct_extract %elem1["data"] : !hw.struct<valid: i1, data: !hw.array<2xi8>>
  %d1_e0 = hw.array_get %d1[%c_idx0] : !hw.array<2xi8>, i1

  hw.output %v0, %d0_e0, %v1, %d1_e0 : i1, i8, i1, i8
}

func.func @main() {
  arc.sim.instantiate @ArrayOfStruct as %model_struct {
    arc.sim.step %model_struct : !arc.sim.instance<@ArrayOfStruct>
    %v0 = arc.sim.get_port %model_struct, "valid0" : i1, !arc.sim.instance<@ArrayOfStruct>
    %val0 = arc.sim.get_port %model_struct, "val0" : i16, !arc.sim.instance<@ArrayOfStruct>
    %v1 = arc.sim.get_port %model_struct, "valid1" : i1, !arc.sim.instance<@ArrayOfStruct>
    %val1 = arc.sim.get_port %model_struct, "val1" : i16, !arc.sim.instance<@ArrayOfStruct>
    arc.sim.emit "struct_v0", %v0 : i1
    arc.sim.emit "struct_val0", %val0 : i16
    arc.sim.emit "struct_v1", %v1 : i1
    arc.sim.emit "struct_val1", %val1 : i16
  }

  arc.sim.instantiate @ArrayOfArray as %model_array {
    arc.sim.step %model_array : !arc.sim.instance<@ArrayOfArray>
    %a0_e0 = arc.sim.get_port %model_array, "a0_e0" : i8, !arc.sim.instance<@ArrayOfArray>
    %a0_e1 = arc.sim.get_port %model_array, "a0_e1" : i8, !arc.sim.instance<@ArrayOfArray>
    %a1_e0 = arc.sim.get_port %model_array, "a1_e0" : i8, !arc.sim.instance<@ArrayOfArray>
    %a1_e1 = arc.sim.get_port %model_array, "a1_e1" : i8, !arc.sim.instance<@ArrayOfArray>
    arc.sim.emit "arr_a0_e0", %a0_e0 : i8
    arc.sim.emit "arr_a0_e1", %a0_e1 : i8
    arc.sim.emit "arr_a1_e0", %a1_e0 : i8
    arc.sim.emit "arr_a1_e1", %a1_e1 : i8
  }

  arc.sim.instantiate @ArrayOfStructWithArray as %model_nested {
    arc.sim.step %model_nested : !arc.sim.instance<@ArrayOfStructWithArray>
    %v0 = arc.sim.get_port %model_nested, "valid0" : i1, !arc.sim.instance<@ArrayOfStructWithArray>
    %d0_e0 = arc.sim.get_port %model_nested, "data0_e0" : i8, !arc.sim.instance<@ArrayOfStructWithArray>
    %v1 = arc.sim.get_port %model_nested, "valid1" : i1, !arc.sim.instance<@ArrayOfStructWithArray>
    %d1_e0 = arc.sim.get_port %model_nested, "data1_e0" : i8, !arc.sim.instance<@ArrayOfStructWithArray>
    arc.sim.emit "nested_v0", %v0 : i1
    arc.sim.emit "nested_d0_e0", %d0_e0 : i8
    arc.sim.emit "nested_v1", %v1 : i1
    arc.sim.emit "nested_d1_e0", %d1_e0 : i8
  }

  return
}
