// RUN: circt-opt %s --verify-diagnostics | circt-opt | FileCheck %s

// CHECK-LABEL: arc.define @Foo
arc.define @Foo(%arg0: i42, %arg1: i9) -> (i42, i9) {
  %c-1_i42 = hw.constant -1 : i42

  // CHECK: arc.output %c-1_i42, %arg1 : i42, i9
  arc.output %c-1_i42, %arg1 : i42, i9
}

arc.define @Bar(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

// CHECK-LABEL: hw.module @Module
hw.module @Module(%clock: i1, %enable: i1, %a: i42, %b: i9) {
  // CHECK: arc.state @Foo(%a, %b) clock %clock lat 1 : (i42, i9) -> (i42, i9)
  arc.state @Foo(%a, %b) clock %clock lat 1 : (i42, i9) -> (i42, i9)

  // CHECK: arc.state @Foo(%a, %b) clock %clock enable %enable lat 1 : (i42, i9) -> (i42, i9)
  arc.state @Foo(%a, %b) clock %clock enable %enable lat 1 : (i42, i9) -> (i42, i9)
}

// CHECK-LABEL: arc.define @SupportRecurisveMemoryEffects
arc.define @SupportRecurisveMemoryEffects(%arg0: i42, %arg1: i1) {
  %0 = scf.if %arg1 -> i42 {
    %1 = comb.and %arg0, %arg0 : i42
    scf.yield %1 : i42
  } else {
    scf.yield %arg0 : i42
  }
  arc.output
}

// CHECK-LABEL: @LookupTable(%arg0: i32, %arg1: i8)
arc.define @LookupTable(%arg0: i32, %arg1: i8) -> () {
  // CHECK-NEXT: %{{.+}} = arc.lut() : () -> i32 {
  // CHECK-NEXT:   %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT:   arc.output %c0_i32 : i32
  // CHECK-NEXT: }
  %0 = arc.lut () : () -> i32 {
    ^bb0():
      %0 = hw.constant 0 : i32
      arc.output %0 : i32
  }
  // CHECK-NEXT: %{{.+}} = arc.lut(%arg1, %arg0) : (i8, i32) -> i32 {
  // CHECK-NEXT: ^bb0(%arg2: i8, %arg3: i32):
  // CHECK-NEXT:   arc.output %arg3 : i32
  // CHECK-NEXT: }
  %1 = arc.lut (%arg1, %arg0) : (i8, i32) -> i32 {
    ^bb0(%arg2: i8, %arg3: i32):
      arc.output %arg3 : i32
  }
  arc.output
}

// CHECK-LABEL: func.func @StorageAccess
func.func @StorageAccess(%arg0: !arc.storage<10000>) {
  // CHECK-NEXT: arc.storage.get %arg0[42] : !arc.storage<10000> -> !arc.state<i9>
  // CHECK-NEXT: arc.storage.get %arg0[1337] : !arc.storage<10000> -> !arc.memory<4 x i19, 4>
  // CHECK-NEXT: arc.storage.get %arg0[9001] : !arc.storage<10000> -> !arc.storage<123>
  %0 = arc.storage.get %arg0[42] : !arc.storage<10000> -> !arc.state<i9>
  %1 = arc.storage.get %arg0[1337] : !arc.storage<10000> -> !arc.memory<4 x i19, 4>
  %2 = arc.storage.get %arg0[9001] : !arc.storage<10000> -> !arc.storage<123>
  return
}

// CHECK-LABEL: func.func @zeroCount
func.func @zeroCount(%arg0 : i32) {
  // CHECK-NEXT: {{%.+}} = arc.zero_count leading %arg0  : i32
  %0 = arc.zero_count leading %arg0  : i32
  // CHECK-NEXT: {{%.+}} = arc.zero_count trailing %arg0  : i32
  %1 = arc.zero_count trailing %arg0  : i32
  return
}

// CHECK-LABEL: @testCallOp
arc.define @testCallOp(%arg0: i1, %arg1: i32) {
  // CHECK-NEXT: {{.*}} = arc.call @dummyCallee1(%arg0, %arg1) : (i1, i32) -> i32
  %0 = arc.call @dummyCallee1(%arg0, %arg1) : (i1, i32) -> i32
  // CHECK-NEXT: arc.call @dummyCallee2()
  arc.call @dummyCallee2() : () -> ()
  arc.output
}
arc.define @dummyCallee1(%arg0: i1, %arg1: i32) -> i32 {
  arc.output %arg1 : i32
}
arc.define @dummyCallee2() {
  arc.output
}
