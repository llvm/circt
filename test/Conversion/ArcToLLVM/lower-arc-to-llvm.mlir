// RUN: circt-opt %s --lower-arc-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @Types(
// CHECK-SAME:    %arg0: !llvm.ptr
// CHECK-SAME:    %arg1: !llvm.ptr
// CHECK-SAME:    %arg2: !llvm.ptr
// CHECK-SAME:  ) -> !llvm.struct<(
// CHECK-SAME:    ptr
// CHECK-SAME:    ptr
// CHECK-SAME:  )> {
func.func @Types(
  %arg0: !arc.storage,
  %arg1: !arc.state<i1>,
  %arg2: !arc.memory<4 x i7, i2>
) -> (
  !arc.storage,
  !arc.state<i1>,
  !arc.memory<4 x i7, i2>
) {
  return %arg0, %arg1, %arg2 : !arc.storage, !arc.state<i1>, !arc.memory<4 x i7, i2>
  // CHECK: llvm.return
  // CHECK-SAME: !llvm.struct<(ptr, ptr, ptr)>
}
// CHECK-NEXT: }

// CHECK-LABEL: llvm.func @StorageTypes(
// CHECK-SAME:    %arg0: !llvm.ptr
// CHECK-SAME:  ) -> !llvm.struct<(
// CHECK-SAME:    ptr
// CHECK-SAME:    ptr
// CHECK-SAME:    ptr
// CHECK-SAME:  )> {
func.func @StorageTypes(%arg0: !arc.storage) -> (!arc.state<i1>, !arc.memory<4 x i1, i2>, !arc.storage) {
  %0 = arc.storage.get %arg0[42] : !arc.storage -> !arc.state<i1>
  // CHECK-NEXT: [[OFFSET:%.+]] = llvm.mlir.constant(42 :
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[[[OFFSET]]]
  %1 = arc.storage.get %arg0[43] : !arc.storage -> !arc.memory<4 x i1, i2>
  // CHECK-NEXT: [[OFFSET:%.+]] = llvm.mlir.constant(43 :
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[[[OFFSET]]]
  %2 = arc.storage.get %arg0[44] : !arc.storage -> !arc.storage
  // CHECK-NEXT: [[OFFSET:%.+]] = llvm.mlir.constant(44 :
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[[[OFFSET]]]
  return %0, %1, %2 : !arc.state<i1>, !arc.memory<4 x i1, i2>, !arc.storage
  // CHECK: llvm.return
}
// CHECK-NEXT: }

// CHECK-LABEL: llvm.func @StateAllocation(
// CHECK-SAME:    %arg0: !llvm.ptr) {
func.func @StateAllocation(%arg0: !arc.storage<10>) {
  arc.root_input "a", %arg0 {offset = 0} : (!arc.storage<10>) -> !arc.state<i1>
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[0]
  arc.root_output "b", %arg0 {offset = 1} : (!arc.storage<10>) -> !arc.state<i2>
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[1]
  arc.alloc_state %arg0 {offset = 2} : (!arc.storage<10>) -> !arc.state<i3>
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[2]
  arc.alloc_memory %arg0 {offset = 3, stride = 1} : (!arc.storage<10>) -> !arc.memory<4 x i1, i2>
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[3]
  arc.alloc_storage %arg0[7] : (!arc.storage<10>) -> !arc.storage<3>
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[7]
  return
  // CHECK-NEXT: llvm.return
}
// CHECK-NEXT: }

// CHECK-LABEL: llvm.func @StateUpdates(
// CHECK-SAME:    %arg0: !llvm.ptr) {
func.func @StateUpdates(%arg0: !arc.storage<1>) {
  %0 = arc.alloc_state %arg0 {offset = 0} : (!arc.storage<1>) -> !arc.state<i1>
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[0]
  %1 = arc.state_read %0 : <i1>
  // CHECK-NEXT: [[LOAD:%.+]] = llvm.load [[PTR]] : !llvm.ptr -> i1
  arc.state_write %0 = %1 : <i1>
  // CHECK-NEXT: llvm.store [[LOAD]], [[PTR]] : i1, !llvm.ptr
  %false = hw.constant false
  arc.state_write %0 = %false if %1 : <i1>
  // CHECK-NEXT:   [[FALSE:%.+]] = llvm.mlir.constant(false)
  // CHECK-NEXT:   llvm.cond_br [[LOAD]], [[BB1:\^.+]], [[BB2:\^.+]]
  // CHECK-NEXT: [[BB1]]:
  // CHECK-NEXT:   llvm.store [[FALSE]], [[PTR]]
  // CHECK-NEXT:   llvm.br [[BB2]]
  // CHECK-NEXT: [[BB2]]:
  return
  // CHECK-NEXT: llvm.return
}
// CHECK-NEXT: }

// CHECK-LABEL: llvm.func @MemoryUpdates(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: i1) {
func.func @MemoryUpdates(%arg0: !arc.storage<24>, %enable: i1) {
  %0 = arc.alloc_memory %arg0 {offset = 0, stride = 6} : (!arc.storage<24>) -> !arc.memory<4 x i42, i19>
  // CHECK-NEXT: [[PTR:%.+]] = llvm.getelementptr %arg0[0]

  %clk = hw.constant true
  %c3_i19 = hw.constant 3 : i19
  // CHECK-NEXT: llvm.mlir.constant(true
  // CHECK-NEXT: [[THREE:%.+]] = llvm.mlir.constant(3

  %1 = arc.memory_read %0[%c3_i19] : <4 x i42, i19>
  %2 = arith.addi %1, %1 : i42
  // CHECK-NEXT:   [[ADDR:%.+]] = llvm.zext [[THREE]] : i19 to i20
  // CHECK-NEXT:   [[FOUR:%.+]] = llvm.mlir.constant(4
  // CHECK-NEXT:   [[INBOUNDS:%.+]] = llvm.icmp "ult" [[ADDR]], [[FOUR]]
  // CHECK-NEXT:   [[GEP:%.+]] = llvm.getelementptr [[PTR]][[[ADDR]]] : (!llvm.ptr, i20) -> !llvm.ptr, i64
  // CHECK-NEXT:   llvm.cond_br [[INBOUNDS]], [[BB_LOAD:\^.+]], [[BB_SKIP:\^.+]]
  // CHECK-NEXT: [[BB_LOAD]]:
  // CHECK-NEXT:   [[TMP:%.+]] = llvm.load [[GEP]] : !llvm.ptr -> i42
  // CHECK-NEXT:   llvm.br [[BB_RESUME:\^.+]]([[TMP]] : i42)
  // CHECK-NEXT: [[BB_SKIP]]:
  // CHECK-NEXT:   [[TMP:%.+]] = llvm.mlir.constant
  // CHECK-NEXT:   llvm.br [[BB_RESUME:\^.+]]([[TMP]] : i42)
  // CHECK-NEXT: [[BB_RESUME]]([[LOADED:%.+]]: i42):
  // CHECK:        [[ADDED:%.+]] = llvm.add [[LOADED]], [[LOADED]]

  arc.memory_write %0[%c3_i19], %2 if %enable : <4 x i42, i19>
  // CHECK-NEXT:   [[ADDR:%.+]] = llvm.zext [[THREE]] : i19 to i20
  // CHECK-NEXT:   [[FOUR:%.+]] = llvm.mlir.constant(4
  // CHECK-NEXT:   [[INBOUNDS:%.+]] = llvm.icmp "ult" [[ADDR]], [[FOUR]]
  // CHECK-NEXT:   [[GEP:%.+]] = llvm.getelementptr [[PTR]][[[ADDR]]] : (!llvm.ptr, i20) -> !llvm.ptr, i64
  // CHECK-NEXT:   [[COND:%.+]] = llvm.and %arg1, [[INBOUNDS]]
  // CHECK-NEXT:   llvm.cond_br [[COND]], [[BB_STORE:\^.+]], [[BB_RESUME:\^.+]]
  // CHECK-NEXT: [[BB_STORE]]:
  // CHECK-NEXT:   llvm.store [[ADDED]], [[GEP]] : i42, !llvm.ptr
  // CHECK-NEXT:   llvm.br [[BB_RESUME]]
  // CHECK-NEXT: [[BB_RESUME]]:

  arc.memory_write %0[%c3_i19], %2 : <4 x i42, i19>
  // CHECK-NEXT:   [[ADDR:%.+]] = llvm.zext [[THREE]] : i19 to i20
  // CHECK-NEXT:   [[FOUR:%.+]] = llvm.mlir.constant(4
  // CHECK-NEXT:   [[INBOUNDS:%.+]] = llvm.icmp "ult" [[ADDR]], [[FOUR]]
  // CHECK-NEXT:   [[GEP:%.+]] = llvm.getelementptr [[PTR]][[[ADDR]]] : (!llvm.ptr, i20) -> !llvm.ptr, i64
  // CHECK-NEXT:   llvm.cond_br [[INBOUNDS]], [[BB_STORE:\^.+]], [[BB_RESUME:\^.+]]
  // CHECK-NEXT: [[BB_STORE]]:
  // CHECK-NEXT:   llvm.store [[ADDED]], [[GEP]] : i42, !llvm.ptr
  // CHECK-NEXT:   llvm.br [[BB_RESUME]]
  // CHECK-NEXT: [[BB_RESUME]]:
  return
  // CHECK-NEXT:   llvm.return
}
// CHECK-NEXT: }

// CHECK-LABEL: llvm.func @zeroCount(
func.func @zeroCount(%arg0 : i32) {
  // CHECK-NEXT: "llvm.intr.ctlz"(%arg0) <{is_zero_poison = true}> : (i32) -> i32
  %0 = arc.zero_count leading %arg0  : i32
  // CHECK-NEXT: "llvm.intr.cttz"(%arg0) <{is_zero_poison = true}> : (i32) -> i32
  %1 = arc.zero_count trailing %arg0  : i32
  return
}

// FIXME: this does not really belong here, but there is no better place either.
// CHECK-LABEL: llvm.func @lowerCombParity(
func.func @lowerCombParity(%arg0: i32) -> i1 {
  // CHECK: %[[CNT:.*]] = llvm.intr.ctpop(%arg0) : (i32) -> i32
  // CHECK: llvm.trunc %[[CNT]] : i32 to i1
  %0 = comb.parity %arg0 : i32

  return %0 : i1
}

// CHECK-LABEL: llvm.func @funcCallOp(
func.func @funcCallOp(%arg0: i32) -> (i32, i32) {
  // CHECK-NEXT: [[V0:%.+]] = llvm.call @dummyFuncCallee(%arg0) : (i32) -> !llvm.struct<(i32, i32)>
  // CHECK-NEXT: [[V1:%.+]] = llvm.extractvalue [[V0]][0] : !llvm.struct<(i32, i32)>
  // CHECK-NEXT: [[V2:%.+]] = llvm.extractvalue [[V0]][1] : !llvm.struct<(i32, i32)>
  %0:2 = func.call @dummyFuncCallee(%arg0) : (i32) -> (i32, i32)
  // CHECK-NEXT: [[V3:%.+]] = llvm.mlir.poison : !llvm.struct<(i32, i32)>
  // CHECK-NEXT: [[V4:%.+]] = llvm.insertvalue [[V1]], [[V3]][0] : !llvm.struct<(i32, i32)>
  // CHECK-NEXT: [[V5:%.+]] = llvm.insertvalue [[V2]], [[V4]][1] : !llvm.struct<(i32, i32)>
  // CHECK-NEXT: llvm.return [[V5]] :
  func.return %0#0, %0#1 : i32, i32
}
func.func @dummyFuncCallee(%arg0: i32) -> (i32, i32) {
  func.return %arg0, %arg0 : i32, i32
}

func.func @seqClocks(%clk1: !seq.clock, %clk2: !seq.clock) -> !seq.clock {
  %0 = seq.from_clock %clk1
  %1 = seq.from_clock %clk2
  %2 = arith.xori %0, %1 : i1
  %3 = seq.to_clock %2
  %4 = seq.clock_inv %3
  %5 = seq.clock_gate %4, %0
  return %5 : !seq.clock
}
// CHECK-LABEL: llvm.func @seqClocks
//  CHECK-SAME: ([[CLK1:%.+]]: i1, [[CLK2:%.+]]: i1)
//       CHECK: [[RES:%.+]] = llvm.xor [[CLK1]], [[CLK2]]
//       CHECK: [[TRUE:%.+]] = llvm.mlir.constant(true) : i1
//       CHECK: [[RES1:%.+]] = llvm.xor [[RES]], [[TRUE]] : i1
//       CHECK: [[RES2:%.+]] = llvm.and [[RES1]], [[CLK1]] : i1
//       CHECK: llvm.return [[RES2]] : i1

// CHECK-LABEL: llvm.func @ReadAggregates(
// CHECK-SAME: %arg0: !llvm.ptr
// CHECK-SAME: %arg1: !llvm.ptr
func.func @ReadAggregates(%arg0: !arc.state<!hw.struct<a: i1, b: i1>>, %arg1: !arc.state<!hw.array<4xi1>>) {
  // CHECK: llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i1, i1)>
  // CHECK: llvm.load %arg1 : !llvm.ptr -> !llvm.array<4 x i1>
  arc.state_read %arg0 : <!hw.struct<a: i1, b: i1>>
  arc.state_read %arg1 : <!hw.array<4xi1>>
  return
}

// CHECK-LABEL: llvm.func @WriteStruct(
// CHECK-SAME: %arg0: !llvm.ptr
// CHECK-SAME: %arg1: !llvm.struct<(i1, i1)>
func.func @WriteStruct(%arg0: !arc.state<!hw.struct<a: i1, b: i1>>, %arg1: !hw.struct<a: i1, b: i1>) {
  // CHECK: [[CONST:%.+]] = llvm.load {{%.+}} : !llvm.ptr -> !llvm.struct<(i1, i1)>
  %0 = hw.aggregate_constant [false, false] : !hw.struct<a: i1, b: i1>
  // CHECK: llvm.store [[CONST]], %arg0 : !llvm.struct<(i1, i1)>, !llvm.ptr
  // CHECK: llvm.store %arg1, %arg0 : !llvm.struct<(i1, i1)>, !llvm.ptr
  arc.state_write %arg0 = %0 : <!hw.struct<a: i1, b: i1>>
  arc.state_write %arg0 = %arg1 : <!hw.struct<a: i1, b: i1>>
  return
}

// CHECK-LABEL: llvm.func @WriteArray(
// CHECK-SAME: %arg0: !llvm.ptr
// CHECK-SAME: %arg1: !llvm.array<4 x i1>
func.func @WriteArray(%arg0: !arc.state<!hw.array<4xi1>>, %arg1: !hw.array<4xi1>) {
  // CHECK: [[CONST:%.+]] = llvm.load {{%.+}} : !llvm.ptr -> !llvm.array<4 x i1>
  %0 = hw.aggregate_constant [false, false, false, false] : !hw.array<4xi1>
  // CHECK: llvm.store [[CONST]], %arg0 : !llvm.array<4 x i1>, !llvm.ptr
  // CHECK: llvm.store %arg1, %arg0 : !llvm.array<4 x i1>, !llvm.ptr
  arc.state_write %arg0 = %0 : <!hw.array<4xi1>>
  arc.state_write %arg0 = %arg1 : <!hw.array<4xi1>>
  return
}

// The LLVM IR does not like `i0` types. The lowering replaces all `i0` values
// with constants to allow canonicalizers to elide i0 values as needed.
// See https://github.com/llvm/circt/pull/8871.
// CHECK-LABEL: llvm.func @DontCrashOnI0(
func.func @DontCrashOnI0(%arg0: i1, %arg1: !hw.array<1xi42>) -> i42 {
  // CHECK: [[STACK:%.+]] = llvm.alloca {{%.+}} x !llvm.array<1 x i42>
  // CHECK: [[ZERO:%.+]] = llvm.mlir.constant(0 : i0) : i0
  // CHECK: [[ZEXT:%.+]] = llvm.zext [[ZERO]] : i0 to i1
  // CHECK: [[GEP:%.+]] = llvm.getelementptr [[STACK]][0, [[ZEXT]]] :
  // CHECK: [[RESULT:%.+]] = llvm.load [[GEP]] : !llvm.ptr -> i42
  // CHECK: llvm.return [[RESULT]]
  %0 = comb.extract %arg0 from 0 : (i1) -> i0
  %1 = hw.array_get %arg1[%0] : !hw.array<1xi42>, i0
  return %1 : i42
}

// CHECK-LABEL: llvm.func @ExecuteEmpty
func.func @ExecuteEmpty() {
  // CHECK-NEXT: llvm.br [[BB:\^.+]]
  // CHECK-NEXT: [[BB]]:
  arc.execute {
    // CHECK-NEXT: llvm.br [[BB:\^.+]]
    arc.output
  }
  // CHECK-NEXT: [[BB]]:
  // CHECK-NEXT: llvm.return
  return
}

// CHECK-LABEL: llvm.func @ExecuteWithOperandsAndResults
func.func @ExecuteWithOperandsAndResults(%arg0: i42, %arg1: !hw.array<4xi19>, %arg2: !arc.storage) {
  // CHECK-NEXT: llvm.br [[BB:\^.+]](%arg0, %arg1, %arg2 : i42, !llvm.array<4 x i19>, !llvm.ptr)
  // CHECK-NEXT: [[BB]]([[ARG0:%.+]]: i42, [[ARG1:%.+]]: !llvm.array<4 x i19>, [[ARG2:%.+]]: !llvm.ptr):
  %4:3 = arc.execute (%arg0, %arg1, %arg2 : i42, !hw.array<4xi19>, !arc.storage) -> (i42, !hw.array<4xi19>, !arc.storage) {
  ^bb0(%0: i42, %1: !hw.array<4xi19>, %2: !arc.storage):
    // CHECK-NEXT: llvm.br [[BB:\^.+]]([[ARG2]] : !llvm.ptr)
    cf.br ^bb1(%2 : !arc.storage)
  ^bb1(%3: !arc.storage):
    // CHECK-NEXT: [[BB]]([[ARG2:%.+]]: !llvm.ptr):
    // CHECK-NEXT: llvm.br [[BB:\^.+]]([[ARG0]], [[ARG1]], [[ARG2]] : i42, !llvm.array<4 x i19>, !llvm.ptr)
    arc.output %0, %1, %3 : i42, !hw.array<4xi19>, !arc.storage
  }
  // CHECK-NEXT: [[BB]]([[ARG0:%.+]]: i42, [[ARG1:%.+]]: !llvm.array<4 x i19>, [[ARG2:%.+]]: !llvm.ptr):
  // CHECK-NEXT: llvm.call @Dummy([[ARG0:%.+]], [[ARG1:%.+]], [[ARG2:%.+]]) : (i42, !llvm.array<4 x i19>, !llvm.ptr) -> ()
  call @Dummy(%4#0, %4#1, %4#2) : (i42, !hw.array<4xi19>, !arc.storage) -> ()
  // CHECK-NEXT: llvm.return
  return
}

// CHECK-LABEL: @issue9171
func.func @issue9171(%arg0: !arc.state<!hw.array<4xi1>>, %idx: i2) -> (i1) {

  // Load the array from memory
  // CHECK-NEXT: [[ARRLD:%.+]]  = llvm.load %arg0 : !llvm.ptr -> !llvm.array<4 x i1>
  // CHECK-NEXT: [[CCLTH:%.+]]  = builtin.unrealized_conversion_cast [[ARRLD]]
  // CHECK-NEXT: [[CCHTL:%.+]]  = builtin.unrealized_conversion_cast [[CCLTH]]

  // Spill the array value on the stack
  // CHECK-NEXT: [[CST1:%.+]]   = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[ALLOCA:%.+]] = llvm.alloca [[CST1]] x !llvm.array<4 x i1>
  // CHECK-NEXT: llvm.store [[CCHTL]], [[ALLOCA]] : !llvm.array<4 x i1>, !llvm.ptr

  // Write the new value to memory
  // CHECK:      [[CPTR:%.+]]   = llvm.mlir.addressof
  // CHECK-NEXT: [[CARR:%.+]]   = llvm.load [[CPTR]] : !llvm.ptr -> !llvm.array<4 x i1>
  // CHECK-NEXT: llvm.store [[CARR]], %arg0 : !llvm.array<4 x i1>, !llvm.ptr

  // Load saved value from the stack
  // CHECK-NEXT: [[IDX:%.+]]    = llvm.zext %arg1 : i2 to i3
  // CHECK-NEXT: [[GPTR:%.+]]   = llvm.getelementptr [[ALLOCA]][0, [[IDX]]]
  // CHECK-NEXT: [[LD:%.+]]     = llvm.load [[GPTR]]
  // CHECK-NEXT: llvm.return [[LD]] : i1

  %pre = arc.state_read %arg0 : <!hw.array<4xi1>>
  %cst = hw.aggregate_constant [false, true, true, false] : !hw.array<4xi1>
  arc.state_write %arg0 = %cst : <!hw.array<4xi1>>
  %get = hw.array_get %pre[%idx] : !hw.array<4xi1>, i2
  return %get : i1
}

// CHECK-LABEL: llvm.mlir.global
// CHECK-SAME:  internal constant @[[NAMESYM:.+]]("fooModelName\00")
// CHECK:  llvm.mlir.global external @arcRuntimeModel_fooModelSym() {addr_space = 0 : i32} : !llvm.struct<(i64, i64, ptr, ptr)> {
// CHECK:    %0 = llvm.mlir.constant({{.+}} : i64) : i64
// CHECK:    %1 = llvm.mlir.constant(1234567 : i64) : i64
// CHECK:    %2 = llvm.mlir.addressof @[[NAMESYM]] : !llvm.ptr
// CHECK:    %3 = llvm.mlir.zero : !llvm.ptr
// CHECK:    %4 = llvm.mlir.poison : !llvm.struct<(i64, i64, ptr, ptr)>
// CHECK:    %5 = llvm.insertvalue %0, %4[0] : !llvm.struct<(i64, i64, ptr, ptr)>
// CHECK:    %6 = llvm.insertvalue %1, %5[1] : !llvm.struct<(i64, i64, ptr, ptr)>
// CHECK:    %7 = llvm.insertvalue %2, %6[2] : !llvm.struct<(i64, i64, ptr, ptr)>
// CHECK:    %8 = llvm.insertvalue %3, %7[3] : !llvm.struct<(i64, i64, ptr, ptr)>
// CHECK:  }

arc.runtime.model @arcRuntimeModel_fooModelSym "fooModelName" numStateBytes 1234567

func.func private @Dummy(%arg0: i42, %arg1: !hw.array<4xi19>, %arg2: !arc.storage)

// CHECK-LABEL: llvm.func @dyn_string
// CHECK:    %0 = llvm.mlir.addressof @_arc_str_0 : !llvm.ptr
// CHECK:    %1 = llvm.mlir.constant(16 : i64) : i64
// CHECK:    %2 = llvm.alloca %1 x !llvm.struct<packed (i64, ptr)> : (i64) -> !llvm.ptr
// CHECK:    llvm.call @arcRuntimeIR_stringInit(%2, %0) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:    %3 = llvm.getelementptr %2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, ptr)>
// CHECK:    %4 = llvm.load %3 : !llvm.ptr -> i64
// CHECK:    %5 = llvm.mlir.addressof @_arc_str_1 : !llvm.ptr
// CHECK:    %6 = llvm.mlir.constant(16 : i64) : i64
// CHECK:    %7 = llvm.alloca %6 x !llvm.struct<packed (i64, ptr)> : (i64) -> !llvm.ptr
// CHECK:    llvm.call @arcRuntimeIR_stringInit(%7, %5) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:    %8 = llvm.mlir.constant(16 : i64) : i64
// CHECK:    %9 = llvm.alloca %8 x !llvm.struct<packed (i64, ptr)> : (i64) -> !llvm.ptr
// CHECK:    %10 = llvm.mlir.constant(0 : i64) : i64
// CHECK:    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
// CHECK:    llvm.call @arcRuntimeIR_stringConcat(%9, %2, %7, %11) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:    llvm.return
func.func @dyn_string() {
  %s1 = sim.string.literal "Hello"
  %l1 = sim.string.length %s1
  %s2 = sim.string.literal "World"
  %s3 = sim.string.concat (%s1, %s2)
  return
}

