// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

// CHECK: to !llhd.time
builtin.unrealized_conversion_cast to !llhd.time
// CHECK: to !llhd.ref<i42>
builtin.unrealized_conversion_cast to !llhd.ref<i42>
// CHECK: to !llhd.ref<!llhd.time>
builtin.unrealized_conversion_cast to !llhd.ref<!llhd.time>

// CHECK-LABEL: @basic
// CHECK-SAME: (in [[IN0:%.+]] : i32, out out0 : i32)
hw.module @basic(in %in0 : i32, out out0 : i32) {
  // CHECK: %{{.*}} = llhd.delay [[IN0]] by <0ns, 1d, 0e> : i32
  %0 = llhd.delay %in0 by <0ns, 1d, 0e> : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @sigExtract
hw.module @sigExtract(in %arg0 : !llhd.ref<i32>, in %arg1 : i5) {
  // CHECK-NEXT: %{{.*}} = llhd.sig.extract %arg0 from %arg1 : <i32> -> <i5>
  %1 = llhd.sig.extract %arg0 from %arg1 : <i32> -> <i5>
}

// CHECK-LABEL: @sigArray
hw.module @sigArray(in %arg0 : !llhd.ref<!hw.array<5xi1>>, in %arg1 : i3) {
  // CHECK-NEXT: %{{.*}} = llhd.sig.array_slice %arg0 at %arg1 : <!hw.array<5xi1>> -> <!hw.array<3xi1>>
  %0 = llhd.sig.array_slice %arg0 at %arg1 : <!hw.array<5xi1>> -> <!hw.array<3xi1>>
  // CHECK-NEXT: %{{.*}} = llhd.sig.array_get %arg0[%arg1] : <!hw.array<5xi1>>
  %1 = llhd.sig.array_get %arg0[%arg1] : <!hw.array<5xi1>>
}

// CHECK-LABEL: @sigStructExtract
hw.module @sigStructExtract(in %arg0 : !llhd.ref<!hw.struct<foo: i1, bar: i2, baz: i3>>) {
  // CHECK-NEXT: %{{.*}} = llhd.sig.struct_extract %arg0["foo"] : <!hw.struct<foo: i1, bar: i2, baz: i3>>
  %0 = llhd.sig.struct_extract %arg0["foo"] : <!hw.struct<foo: i1, bar: i2, baz: i3>>
  // CHECK-NEXT: %{{.*}} = llhd.sig.struct_extract %arg0["baz"] : <!hw.struct<foo: i1, bar: i2, baz: i3>>
  %1 = llhd.sig.struct_extract %arg0["baz"] : <!hw.struct<foo: i1, bar: i2, baz: i3>>
}

// CHECK-LABEL: @checkSigInst
hw.module @checkSigInst() {
  // CHECK: %[[CI1:.*]] = hw.constant
  %cI1 = hw.constant 0 : i1
  // CHECK-NEXT: %sigI1 = llhd.sig %[[CI1]] : i1
  %sigI1 = llhd.sig %cI1 : i1
  // CHECK-NEXT: %[[CI64:.*]] = hw.constant
  %cI64 = hw.constant 0 : i64
  // CHECK-NEXT: %sigI64 = llhd.sig %[[CI64]] : i64
  %sigI64 = llhd.sig %cI64 : i64

  // CHECK-NEXT: %[[TUP:.*]] = hw.struct_create
  %tup = hw.struct_create (%cI1, %cI64) : !hw.struct<foo: i1, bar: i64>
  // CHECK-NEXT: %sigTup = llhd.sig %[[TUP]] : !hw.struct<foo: i1, bar: i64>
  %sigTup = llhd.sig %tup : !hw.struct<foo: i1, bar: i64>

  // CHECK-NEXT: %[[ARRAY:.*]] = hw.array_create
  %array = hw.array_create %cI1, %cI1 : i1
  // CHECK-NEXT: %sigArray = llhd.sig %[[ARRAY]] : !hw.array<2xi1>
  %sigArray = llhd.sig %array : !hw.array<2xi1>
}

// CHECK-LABEL: @checkPrb
hw.module @checkPrb(
  in %arg0 : !llhd.ref<i1>,
  in %arg1 : !llhd.ref<i64>,
  in %arg2 : !llhd.ref<!hw.array<3xi8>>,
  in %arg3 : !llhd.ref<!hw.struct<foo: i1, bar: i2, baz: i4>>
) {
  // CHECK: %{{.*}} = llhd.prb %arg0 : i1
  %0 = llhd.prb %arg0 : i1
  // CHECK-NEXT: %{{.*}} = llhd.prb %arg1 : i64
  %1 = llhd.prb %arg1 : i64
  // CHECK-NEXT: %{{.*}} = llhd.prb %arg2 : !hw.array<3xi8>
  %2 = llhd.prb %arg2 : !hw.array<3xi8>
  // CHECK-NEXT: %{{.*}} = llhd.prb %arg3 : !hw.struct<foo: i1, bar: i2, baz: i4>
  %3 = llhd.prb %arg3 : !hw.struct<foo: i1, bar: i2, baz: i4>
}

// CHECK-LABEL: @checkOutput
hw.module @checkOutput(in %arg0 : i32) {
  %t = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: %{{.+}} = llhd.output %arg0 after %{{.*}} : i32
  %0 = llhd.output %arg0 after %t : i32
  // CHECK-NEXT: %{{.+}} = llhd.output "sigName" %arg0 after %{{.*}} : i32
  %1 = llhd.output "sigName" %arg0 after %t : i32
}

// CHECK-LABEL: @checkDrv
hw.module @checkDrv(
  in %arg0 : !llhd.ref<i1>,
  in %arg1 : !llhd.ref<i64>,
  in %arg2 : i1,
  in %arg3 : i64,
  in %arg5 : !llhd.ref<!hw.array<3xi8>>,
  in %arg6 : !llhd.ref<!hw.struct<foo: i1, bar: i2, baz: i4>>,
  in %arg7 : !hw.array<3xi8>,
  in %arg8 : !hw.struct<foo: i1, bar: i2, baz: i4>
) {
  %t = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.drv %arg0, %arg2 after %{{.*}} : i1
  llhd.drv %arg0, %arg2 after %t : i1
  // CHECK-NEXT: llhd.drv %arg1, %arg3 after %{{.*}} : i64
  llhd.drv %arg1, %arg3 after %t : i64
  // CHECK-NEXT: llhd.drv %arg1, %arg3 after %{{.*}} if %arg2 : i64
  llhd.drv %arg1, %arg3 after %t if %arg2 : i64
  // CHECK-NEXT: llhd.drv %arg5, %arg7 after %{{.*}} : !hw.array<3xi8>
  llhd.drv %arg5, %arg7 after %t : !hw.array<3xi8>
  // CHECK-NEXT: llhd.drv %arg6, %arg8 after %{{.*}} : !hw.struct<foo: i1, bar: i2, baz: i4>
  llhd.drv %arg6, %arg8 after %t : !hw.struct<foo: i1, bar: i2, baz: i4>
}

// CHECK-LABEL: @check_wait_0
hw.module @check_wait_0() {
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK: llhd.wait ^[[BB:.*]]
    llhd.wait ^bb1
    // CHECK-NEXT: ^[[BB]]
  ^bb1:
    llhd.halt
  }
}

// CHECK-LABEL: @check_wait_1
hw.module @check_wait_1() {
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK-NEXT: %[[TIME:.*]] = llhd.constant_time
    %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
    // CHECK-NEXT: llhd.wait delay %[[TIME]], ^[[BB:.*]](%[[TIME]] : !llhd.time)
    llhd.wait delay %time, ^bb1(%time: !llhd.time)
    // CHECK-NEXT: ^[[BB]](%[[T:.*]]: !llhd.time):
  ^bb1(%t: !llhd.time):
    llhd.halt
  }
}

// CHECK-LABEL: @check_wait_2
hw.module @check_wait_2(in %arg0 : !llhd.ref<i64>, in %arg1 : !llhd.ref<i1>) {
  // CHECK: [[PRB0:%.+]] = llhd.prb %arg0
  %prb0 = llhd.prb %arg0 : i64
  // CHECK: [[PRB1:%.+]] = llhd.prb %arg1
  %prb1 = llhd.prb %arg1 : i1
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.wait ([[PRB0]], [[PRB1]] : i64, i1), ^[[BB:.*]](%arg1 : !llhd.ref<i1>)
    llhd.wait (%prb0, %prb1 : i64, i1), ^bb1(%arg1 : !llhd.ref<i1>)
    // CHECK: ^[[BB]](%[[A:.*]]: !llhd.ref<i1>):
  ^bb1(%a: !llhd.ref<i1>):
    llhd.halt
  }
}

// CHECK-LABEL: @check_wait_3
hw.module @check_wait_3(in %arg0 : !llhd.ref<i64>, in %arg1 : !llhd.ref<i1>) {
  // CHECK: [[PRB0:%.+]] = llhd.prb %arg0
  %prb0 = llhd.prb %arg0 : i64
  // CHECK: [[PRB1:%.+]] = llhd.prb %arg1
  %prb1 = llhd.prb %arg1 : i1
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK-NEXT: %[[TIME:.*]] = llhd.constant_time
    %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
    // CHECK-NEXT: llhd.wait delay %[[TIME]], ([[PRB0]], [[PRB1]] : i64, i1), ^[[BB:.*]](%arg1, %arg0 : !llhd.ref<i1>, !llhd.ref<i64>)
    llhd.wait delay %time, (%prb0, %prb1 : i64, i1), ^bb1(%arg1, %arg0 : !llhd.ref<i1>, !llhd.ref<i64>)
    // CHECK: ^[[BB]](%[[A:.*]]: !llhd.ref<i1>, %[[B:.*]]: !llhd.ref<i64>):
  ^bb1(%a: !llhd.ref<i1>, %b: !llhd.ref<i64>):
    llhd.halt
  }
}

// CHECK-LABEL: @FinalProcess
hw.module @FinalProcess () {
  // CHECK-NEXT: llhd.final {
  // CHECK-NEXT:   llhd.halt
  // CHECK-NEXT: }
  llhd.final {
    llhd.halt
  }
}

hw.module @ProcessWithResults(in %arg0: i42, in %arg1: i9001) {
  %0:2 = llhd.process -> i42, i9001 {
    llhd.wait yield (%arg0, %arg1 : i42, i9001), ^bb1
  ^bb1:
    llhd.halt %arg0, %arg1 : i42, i9001
  }
}

hw.module @CombinationalProcess(in %arg0: i1, in %arg1: i42, in %arg2: i9001, in %arg3: i42, in %arg4: i9001) {
  llhd.combinational {
    llhd.yield
  }
  %0:2 = llhd.combinational -> i42, i9001 {
    cf.cond_br %arg0, ^bb1(%arg1, %arg2 : i42, i9001), ^bb1(%arg3, %arg4 : i42, i9001)
  ^bb1(%1: i42, %2: i9001):
    llhd.yield %1, %2 : i42, i9001
  }
}
