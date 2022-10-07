// RUN: circt-opt -handshake-lower-extmem-to-hw="wrap-esi=true" %s | FileCheck %s


//CHECK-LABEL: hw.module.extern @_main_hw(%arg0: !esi.channel<i64>, %arg1: !esi.channel<i64>, %v: !esi.channel<i32>, %mem_ld0.data: !esi.channel<i32>, %mem_st0.done: !esi.channel<i0>, %argCtrl: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>, mem_ld0.addr: !esi.channel<i4>, mem_st0: !esi.channel<!hw.struct<address: i4, data: i32>>)

//CHECK-LABEL: esi.mem.ram @mem i32 x 10

//CHECK-LABEL: hw.module @main_esi_wrapper(%arg0: !esi.channel<i64>, %arg1: !esi.channel<i64>, %v: !esi.channel<i32>, %argCtrl: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>) {
//CHECK-NEXT:   %0 = esi.service.req.inout %main.mem_ld0.addr -> <@mem::@read>([]) : !esi.channel<i4> -> !esi.channel<i32>
//CHECK-NEXT:   %1 = esi.service.req.inout %main.mem_st0 -> <@mem::@write>([]) : !esi.channel<!hw.struct<address: i4, data: i32>> -> !esi.channel<i0>
//CHECK-NEXT:   %main.out0, %main.mem_ld0.addr, %main.mem_st0 = hw.instance "main" @_main_hw(arg0: %arg0: !esi.channel<i64>, arg1: %arg1: !esi.channel<i64>, v: %v: !esi.channel<i32>, mem_ld0.data: %0: !esi.channel<i32>, mem_st0.done: %1: !esi.channel<i0>, argCtrl: %argCtrl: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>, mem_ld0.addr: !esi.channel<i4>, mem_st0: !esi.channel<!hw.struct<address: i4, data: i32>>)
//CHECK-NEXT:   hw.output %main.out0 : !esi.channel<i0>
//CHECK-NEXT: }

handshake.func @main(%arg0: index, %arg1: index, %v: i32, %mem : memref<10xi32>, %argCtrl: none) -> none {
  %ldData, %stCtrl, %ldCtrl = handshake.extmemory[ld=1, st=1](%mem : memref<10xi32>)(%storeData, %storeAddr, %loadAddr) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %fCtrl:2 = fork [2] %argCtrl : none
  %loadData, %loadAddr = load [%arg0] %ldData, %fCtrl#0 : index, i32
  %storeData, %storeAddr = store [%arg1] %v, %fCtrl#1 : index, i32
  sink %loadData : i32
  %finCtrl = join %stCtrl, %ldCtrl : none, none
  return %finCtrl : none
}
