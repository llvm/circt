// RUN: circt-opt %s --arc-allocate-state | FileCheck %s

// CHECK-LABEL: arc.model @test
arc.model @test io !hw.modty<input x : i1, output y : i1> {
^bb0(%arg0: !arc.storage):
  // CHECK-NEXT: ([[PTR:%.+]]: !arc.storage<5780>):

  // CHECK-NEXT: arc.alloc_storage [[PTR]][0] : (!arc.storage<5780>) -> !arc.storage<1159>
  // CHECK-NEXT: arc.initial {
  arc.initial {
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][0] : !arc.storage<5780> -> !arc.storage<1159>
    %0 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i8>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i16>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i32>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i64>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1> // make sure the current offset is not already 16-byte aligned
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i9001>
    %1 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 0 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 1 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 2 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 4 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 8 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 16 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 32 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 1158 : i32}
    // CHECK-NEXT: scf.execute_region {
    scf.execute_region {
      arc.state_read %0 : <i1>
      // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][0] : !arc.storage<5780> -> !arc.storage<1159>
      // CHECK-NEXT: [[STATE:%.+]] = arc.storage.get [[SUBPTR]][0] : !arc.storage<1159> -> !arc.state<i1>
      // CHECK-NEXT: arc.state_read [[STATE]] : <i1>
      arc.state_read %1 : <i1>
      // CHECK-NEXT: [[STATE:%.+]] = arc.storage.get [[SUBPTR]][1158] : !arc.storage<1159> -> !arc.state<i1>
      // CHECK-NEXT: arc.state_read [[STATE]] : <i1>
      scf.yield
      // CHECK-NEXT: scf.yield
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: }

  // CHECK-NEXT: arc.alloc_storage [[PTR]][1168] : (!arc.storage<5780>) -> !arc.storage<4609>
  // CHECK-NEXT: arc.initial {
  arc.initial {
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][1168] : !arc.storage<5780> -> !arc.storage<4609>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i1, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i8, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i16, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i32, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i64, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i9001, i1>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 0 : i32, stride = 1 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i1, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 4 : i32, stride = 1 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i8, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 8 : i32, stride = 2 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i16, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 16 : i32, stride = 4 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i32, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 32 : i32, stride = 8 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i64, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 64 : i32, stride = 1136 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i9001, i1>
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 4608 : i32}
  }
  // CHECK-NEXT: }

  // CHECK-NEXT: arc.alloc_storage [[PTR]][5778] : (!arc.storage<5780>) -> !arc.storage<2>
  // CHECK-NEXT: arc.initial {
  arc.initial {
    arc.root_input "x", %arg0 : (!arc.storage) -> !arc.state<i1>
    arc.root_output "y", %arg0 : (!arc.storage) -> !arc.state<i1>
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][5778] : !arc.storage<5780> -> !arc.storage<2>
    // CHECK-NEXT: arc.root_input "x", [[SUBPTR]] {offset = 0 : i32}
    // CHECK-NEXT: arc.root_output "y", [[SUBPTR]] {offset = 1 : i32}
  }
  // CHECK-NEXT: }
}

// CHECK-LABEL: arc.model @StructPadding
// CHECK-NEXT: !arc.storage<4>
arc.model @StructPadding io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // This !hw.struct is only 11 bits wide, but mapped to an !llvm.struct, each
  // field gets byte-aligned.
  arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<!hw.struct<tag: i5, sign_ext: i1, offset: i3, size: i2>>
}

// CHECK-LABEL: arc.model @ArrayPadding
// CHECK-NEXT: !arc.storage<4>
arc.model @ArrayPadding io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // This !hw.array is only 18 bits wide, but mapped to an !llvm.array, each
  // element gets aligned to the next power-of-two.
  arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<!hw.array<2xi9>>
}
