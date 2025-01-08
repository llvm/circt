// RUN: circt-opt %s --lower-smt-to-z3-llvm --canonicalize | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void --shared-libs=%libz3 | \
// RUN: FileCheck %s

// RUN: circt-opt %s --lower-smt-to-z3-llvm=debug=true --canonicalize | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void --shared-libs=%libz3 | \
// RUN: FileCheck %s

// REQUIRES: libz3
// REQUIRES: mlir-cpu-runner

func.func @entry() {
  %false = llvm.mlir.constant(0 : i1) : i1
  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %c42_bv65 = smt.bv.constant #smt.bv<42> : !smt.bv<65>
    %1 = smt.declare_fun : !smt.bv<65>
    %2 = smt.declare_fun "a" : !smt.bv<65>
    %3 = smt.eq %c42_bv65, %1, %2 : !smt.bv<65>
    func.call @check(%3) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %c0_bv8 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
    %c-1_bv8 = smt.bv.constant #smt.bv<-1> : !smt.bv<8>
    %2 = smt.distinct %c0_bv8, %c-1_bv8 : !smt.bv<8>
    func.call @check(%2) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %0 = smt.declare_fun : !smt.func<(!smt.bv<4>) !smt.array<[!smt.int -> !smt.sort<"uninterpreted_sort"[!smt.bool]>]>>
    %1 = smt.declare_fun : !smt.func<(!smt.bv<4>) !smt.array<[!smt.int -> !smt.sort<"uninterpreted_sort"[!smt.bool]>]>>
    %c0_bv4 = smt.bv.constant #smt.bv<0> : !smt.bv<4>
    %2 = smt.apply_func %0(%c0_bv4) : !smt.func<(!smt.bv<4>) !smt.array<[!smt.int -> !smt.sort<"uninterpreted_sort"[!smt.bool]>]>>
    %3 = smt.apply_func %1(%c0_bv4) : !smt.func<(!smt.bv<4>) !smt.array<[!smt.int -> !smt.sort<"uninterpreted_sort"[!smt.bool]>]>>
    %4 = smt.eq %2, %3 : !smt.array<[!smt.int -> !smt.sort<"uninterpreted_sort"[!smt.bool]>]>
    func.call @check(%4) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: unsat
  // CHECK: Res: -1
  smt.solver (%false) : (i1) -> () {
  ^bb0(%arg0: i1):
    %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
    %0 = scf.if %arg0 -> !smt.bv<32> {
      %1 = smt.declare_fun : !smt.bv<32>
      scf.yield %1 : !smt.bv<32>
    } else {
      %c1_bv32 = smt.bv.constant #smt.bv<-1> : !smt.bv<32>
      scf.yield %c1_bv32 : !smt.bv<32>
    }
    %1 = smt.eq %c0_bv32, %0 : !smt.bv<32>
    func.call @check(%1) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: unsat
  // CHECK: Res: -1
  smt.solver () : () -> () {
    %0 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
    %1 = smt.bv.constant #smt.bv<2> : !smt.bv<32>
    %2 = smt.bv.cmp ugt %0, %1 : !smt.bv<32>
    func.call @check(%2) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: unsat
  // CHECK: Res: -1
  smt.solver () : () -> () {
    %t = smt.constant true
    %f = smt.constant false
    %0 = smt.xor %t, %f, %t, %f
    func.call @check(%0) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %0 = smt.bv.constant #smt.bv<0x0f> : !smt.bv<8>
    %1 = smt.bv.constant #smt.bv<0x0> : !smt.bv<4>
    %2 = smt.bv.concat %0, %1 : !smt.bv<8>, !smt.bv<4>
    %3 = smt.bv.extract %0 from 0 : (!smt.bv<8>) -> !smt.bv<4>
    %4 = smt.bv.constant #smt.bv<0x0f0> : !smt.bv<12>
    %5 = smt.bv.constant #smt.bv<0xf> : !smt.bv<4>
    %6 = smt.eq %2, %4 : !smt.bv<12>
    %7 = smt.eq %3, %5 : !smt.bv<4>
    %8 = smt.and %6, %7
    func.call @check(%8) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %0 = smt.int.constant -42
    %1 = smt.int.constant -9223372036854775809
    %2 = smt.int.constant 9223372036854775809
    %3 = smt.int.abs %1
    %4 = smt.int.cmp lt %1, %0
    %5 = smt.eq %3, %2 : !smt.int
    %6 = smt.and %4, %5
    func.call @check(%6) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %0 = smt.exists {
    ^bb0(%arg0: !smt.int):
      %1 = smt.forall {
      ^bb1(%arg1: !smt.int):
        %2 = smt.int.mul %arg0, %arg1
        %3 = smt.int.constant 0
        %4 = smt.eq %2, %3 : !smt.int
        smt.yield %4 : !smt.bool
      }
      smt.yield %1 : !smt.bool
    }
    func.call @check(%0) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %0 = smt.declare_fun : !smt.func<(!smt.sort<"A">) !smt.sort<"B">>
    %1 = smt.forall {
    ^bb0(%arg0: !smt.sort<"A">, %arg1: !smt.sort<"A">):
      %2 = smt.apply_func %0(%arg0) : !smt.func<(!smt.sort<"A">) !smt.sort<"B">>
      %3 = smt.apply_func %0(%arg1) : !smt.func<(!smt.sort<"A">) !smt.sort<"B">>
      %4 = smt.eq %2, %3 : !smt.sort<"B">
      %5 = smt.eq %arg0, %arg1 : !smt.sort<"A">
      %6 = smt.eq %4, %5 : !smt.bool
      smt.yield %6 : !smt.bool
    } patterns {
    ^bb0(%arg0: !smt.sort<"A">, %arg1: !smt.sort<"A">):
      %2 = smt.apply_func %0(%arg0) : !smt.func<(!smt.sort<"A">) !smt.sort<"B">>
      %3 = smt.apply_func %0(%arg1) : !smt.func<(!smt.sort<"A">) !smt.sort<"B">>
      smt.yield %2, %3 : !smt.sort<"B">, !smt.sort<"B">
    }
    func.call @check(%1) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: unsat
  // CHECK: Res: -1
  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %c3 = smt.int.constant 3
    %c4 = smt.int.constant 4
    %unsat_eq = smt.eq %c3, %c4 : !smt.int
    smt.push 1
    func.call @check(%unsat_eq) : (!smt.bool) -> ()
    smt.pop 1
    %sat_eq = smt.eq %c3, %c3 : !smt.int
    func.call @check(%sat_eq) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: unsat
  // CHECK: Res: -1
  // CHECK: sat
  // CHECK: Res: 1
  smt.solver () : () -> () {
    %c3 = smt.int.constant 3
    %c4 = smt.int.constant 4
    %unsat_eq = smt.eq %c3, %c4 : !smt.int
    func.call @check(%unsat_eq) : (!smt.bool) -> ()
    smt.reset
    %sat_eq = smt.eq %c3, %c3 : !smt.int
    func.call @check(%sat_eq) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: unknown
  // CHECK: Res: 0
  smt.solver () : () -> () {
    smt.set_logic "HORN"
    %c = smt.declare_fun : !smt.int
    %c4 = smt.int.constant 4
    %eq = smt.eq %c, %c4 : !smt.int
    func.call @check(%eq) : (!smt.bool) -> ()
    smt.yield
  }

  // CHECK: unsat
  // CHECK: Res: -1
  smt.solver () : () -> () {
    %c4 = smt.int.constant 4
    %c4_bv16 = smt.bv.constant #smt.bv<4> : !smt.bv<16>
    %int2bv = smt.int2bv %c4 : !smt.bv<16>
    %eq = smt.distinct %c4_bv16, %int2bv : !smt.bv<16>
    func.call @check(%eq) : (!smt.bool) -> ()
    smt.yield
  }

  return
}


func.func @check(%expr: !smt.bool) {
  smt.assert %expr
  %0 = smt.check sat {
    %1 = llvm.mlir.addressof @sat : !llvm.ptr
    llvm.call @printf(%1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    smt.yield %c1 : i32
  } unknown {
    %1 = llvm.mlir.addressof @unknown : !llvm.ptr
    llvm.call @printf(%1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    smt.yield %c0 : i32
  } unsat {
    %1 = llvm.mlir.addressof @unsat : !llvm.ptr
    llvm.call @printf(%1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %c-1 = llvm.mlir.constant(-1 : i32) : i32
    smt.yield %c-1 : i32
  } -> i32
  %1 = llvm.mlir.addressof @res : !llvm.ptr
  llvm.call @printf(%1, %0) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
  return
}

llvm.func @printf(!llvm.ptr, ...) -> i32
llvm.mlir.global private constant @res("Res: %d\n\00") {addr_space = 0 : i32}
llvm.mlir.global private constant @sat("sat\n\00") {addr_space = 0 : i32}
llvm.mlir.global private constant @unsat("unsat\n\00") {addr_space = 0 : i32}
llvm.mlir.global private constant @unknown("unknown\n\00") {addr_space = 0 : i32}
