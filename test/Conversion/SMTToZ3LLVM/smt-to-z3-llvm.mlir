// RUN: circt-opt %s --lower-smt-to-z3-llvm | FileCheck %s
// RUN: circt-opt %s --lower-smt-to-z3-llvm=debug=true | FileCheck %s --check-prefix=CHECK-DEBUG

// CHECK-LABEL: llvm.mlir.global internal @ctx_0()
// CHECK-NEXT:   llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:   llvm.return
// CHECK-LABEL: llvm.mlir.global internal @solver_0()
// CHECK-NEXT:   llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:   llvm.return

// CHECK-LABEL: llvm.mlir.global internal @ctx()
llvm.mlir.global internal @ctx() {alignment = 8 : i64} : !llvm.ptr {
  %0 = llvm.mlir.zero : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
// CHECK-LABEL: llvm.mlir.global internal @solver()
llvm.mlir.global internal @solver() {alignment = 8 : i64} : !llvm.ptr {
  %0 = llvm.mlir.zero : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}


// CHECK-LABEL: llvm.func @test
// CHECK:   [[CONFIG:%.+]] = llvm.call @Z3_mk_config() : () -> !llvm.ptr
// CHECK-DEBUG: [[PROOF_STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
// CHECK-DEBUG: [[TRUE_STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
// CHECK-DEBUG: llvm.call @Z3_set_param_value({{.*}}, [[PROOF_STR]], [[TRUE_STR]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:   [[CTX:%.+]] = llvm.call @Z3_mk_context([[CONFIG]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:   [[CTX_ADDR:%.+]] = llvm.mlir.addressof @ctx_0 : !llvm.ptr
// CHECK:   llvm.store [[CTX]], [[CTX_ADDR]] : !llvm.ptr, !llvm.ptr
// CHECK:   llvm.call @Z3_del_config([[CONFIG]]) : (!llvm.ptr) -> ()
// CHECK:   [[SOLVER:%.+]] = llvm.call @Z3_mk_solver([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:   llvm.call @Z3_solver_inc_ref([[CTX]], [[SOLVER]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:   [[SOLVER_ADDR:%.+]] = llvm.mlir.addressof @solver_0 : !llvm.ptr
// CHECK:   llvm.store [[SOLVER]], [[SOLVER_ADDR]] : !llvm.ptr, !llvm.ptr
// CHECK:   llvm.call @solver
// CHECK:   llvm.call @Z3_solver_dec_ref([[CTX]], [[SOLVER]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:   llvm.call @Z3_del_context([[CTX]]) : (!llvm.ptr) -> ()
// CHECK:   llvm.return

// CHECK-LABEL: llvm.func @test_logic
// CHECK:   [[CONFIG1:%.+]] = llvm.call @Z3_mk_config() : () -> !llvm.ptr
// CHECK-DEBUG: [[PROOF_STR1:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
// CHECK-DEBUG: [[TRUE_STR1:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
// CHECK-DEBUG: llvm.call @Z3_set_param_value({{.*}}, [[PROOF_STR1]], [[TRUE_STR1]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:   [[CTX1:%.+]] = llvm.call @Z3_mk_context([[CONFIG1]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:   [[CTX_ADDR1:%.+]] = llvm.mlir.addressof @ctx_0 : !llvm.ptr
// CHECK:   llvm.store [[CTX1]], [[CTX_ADDR1]] : !llvm.ptr, !llvm.ptr
// CHECK:   llvm.call @Z3_del_config([[CONFIG1]]) : (!llvm.ptr) -> ()
// CHECK:   [[LOGICADDR:%.+]] = llvm.mlir.addressof [[LOGICSTR:@.+]] : !llvm.ptr
// CHECK:   [[SOLVER1:%.+]] = llvm.call @Z3_mk_solver_for_logic([[CTX1]], [[LOGICADDR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK:   llvm.call @Z3_solver_inc_ref([[CTX1]], [[SOLVER1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:   [[SOLVER_ADDR1:%.+]] = llvm.mlir.addressof @solver_0 : !llvm.ptr
// CHECK:   llvm.store [[SOLVER1]], [[SOLVER_ADDR1]] : !llvm.ptr, !llvm.ptr
// CHECK:   llvm.call @solver
// CHECK:   llvm.call @Z3_solver_dec_ref([[CTX1]], [[SOLVER1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:   llvm.call @Z3_del_context([[CTX1]]) : (!llvm.ptr) -> ()
// CHECK:   llvm.return

// CHECK-LABEL: llvm.func @solver
func.func @test(%arg0: i32) {
  %0 = smt.solver (%arg0) : (i32) -> (i32) {
  ^bb0(%arg1: i32):
    // CHECK: [[S_ADDR:%.+]] = llvm.mlir.addressof @solver_0 : !llvm.ptr
    // CHECK: [[S:%.+]] = llvm.load [[S_ADDR]] : !llvm.ptr -> !llvm.ptr
    // CHECK: [[CTX_ADDR:%.+]] = llvm.mlir.addressof @ctx_0 : !llvm.ptr
    // CHECK: [[CTX:%.+]] = llvm.load [[CTX_ADDR]] : !llvm.ptr -> !llvm.ptr

    // CHECK: [[STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
    // CHECK: [[INT_SORT:%.+]] = llvm.call @Z3_mk_int_sort([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
    // CHECK: [[BOOL_SORT:%.+]] = llvm.call @Z3_mk_bool_sort([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
    // CHECK: [[ARRAY_SORT:%.+]] = llvm.call @Z3_mk_array_sort([[CTX]], [[INT_SORT]], [[BOOL_SORT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK: llvm.call @Z3_mk_fresh_const([[CTX]], [[STR]], [[ARRAY_SORT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr

    // Test: declare constant, array, int, bool types
    %1 = smt.declare_fun "a" : !smt.array<[!smt.int -> !smt.bool]>

    // CHECK: [[ZERO:%.+]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: [[STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
    // CHECK: [[SYM:%.+]] = llvm.call @Z3_mk_string_symbol([[CTX]], [[STR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK: [[SORT:%.+]] = llvm.call @Z3_mk_uninterpreted_sort([[CTX]], [[SYM]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK: [[ARR0:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
    // CHECK: [[C65:%.+]] = llvm.mlir.constant(65 : i32) : i32
    // CHECK: [[BV_SORT:%.+]] = llvm.call @Z3_mk_bv_sort([[CTX]], [[C65]]) : (!llvm.ptr, i32) -> !llvm.ptr
    // CHECK: [[ARR1:%.+]] = llvm.insertvalue [[BV_SORT]], [[ARR0]][0] : !llvm.array<2 x ptr>
    // CHECK: [[C4:%.+]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: [[BV_SORT:%.+]] = llvm.call @Z3_mk_bv_sort([[CTX]], [[C4]]) : (!llvm.ptr, i32) -> !llvm.ptr
    // CHECK: [[ARR2:%.+]] = llvm.insertvalue [[BV_SORT]], [[ARR1]][1] : !llvm.array<2 x ptr>
    // CHECK: [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[STORAGE:%.+]] = llvm.alloca [[C1]] x !llvm.array<2 x ptr> : (i32) -> !llvm.ptr
    // CHECK: llvm.store [[ARR2]], [[STORAGE]] : !llvm.array<2 x ptr>, !llvm.ptr
    // CHECK: [[C2:%.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: [[V2:%.+]] = llvm.call @Z3_mk_fresh_func_decl([[CTX]], [[ZERO]], [[C2]], [[STORAGE]], [[SORT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr

    // Test: declare function, bit-vector, uninterpreted sort types
    %2 = smt.declare_fun : !smt.func<(!smt.bv<65>, !smt.bv<4>) !smt.sort<"uninterpreted_sort"[!smt.bool]>>
    
    // CHECK: [[C4:%.+]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: [[BV_SORT:%.+]] = llvm.call @Z3_mk_bv_sort([[CTX]], [[C4]]) : (!llvm.ptr, i32) -> !llvm.ptr
    // CHECK: [[C0:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[BV0:%.+]] = llvm.call @Z3_mk_unsigned_int64([[CTX]], [[C0]], [[BV_SORT]]) : (!llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
    %c0_bv4 = smt.bv.constant #smt.bv<0> : !smt.bv<4>

    // CHECK: [[C65:%.+]] = llvm.mlir.constant(65 : i32) : i32
    // CHECK: [[BV_SORT:%.+]] = llvm.call @Z3_mk_bv_sort([[CTX]], [[C65]]) : (!llvm.ptr, i32) -> !llvm.ptr
    // CHECK: [[STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
    // CHECK: [[BV42:%.+]] = llvm.call @Z3_mk_numeral([[CTX]], [[STR]], [[BV_SORT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %c42_bv65 = smt.bv.constant #smt.bv<42> : !smt.bv<65>

    // CHECK: [[ARR0:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
    // CHECK: [[ARR1:%.+]] = llvm.insertvalue [[BV42]], [[ARR0]][0] : !llvm.array<2 x ptr>
    // CHECK: [[ARR2:%.+]] = llvm.insertvalue [[BV0]], [[ARR1]][1] : !llvm.array<2 x ptr>
    // CHECK: [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[STORAGE:%.+]] = llvm.alloca [[C1]] x !llvm.array<2 x ptr> : (i32) -> !llvm.ptr
    // CHECK: llvm.store [[ARR2]], [[STORAGE]] : !llvm.array<2 x ptr>, !llvm.ptr
    // CHECK: [[C2:%.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: llvm.call @Z3_mk_app([[CTX]], [[V2]], [[C2]], [[STORAGE]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    %3 = smt.apply_func %2(%c42_bv65, %c0_bv4) : !smt.func<(!smt.bv<65>, !smt.bv<4>) !smt.sort<"uninterpreted_sort"[!smt.bool]>>

    // CHECK: [[EQ2:%.+]] = llvm.call @Z3_mk_eq([[CTX]], [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK: [[EQ3:%.+]] = llvm.call @Z3_mk_eq([[CTX]], [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK: [[C2:%.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[STORAGE:%.+]] = llvm.alloca [[C1]] x !llvm.array<2 x ptr> : (i32) -> !llvm.ptr
    // CHECK: [[ARR0:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
    // CHECK: [[ARR1:%.+]] = llvm.insertvalue [[EQ2]], [[ARR0]][0] : !llvm.array<2 x ptr>
    // CHECK: [[ARR2:%.+]] = llvm.insertvalue [[EQ3]], [[ARR1]][1] : !llvm.array<2 x ptr>
    // CHECK: llvm.store [[ARR2]], [[STORAGE]] : !llvm.array<2 x ptr>, !llvm.ptr
    // CHECK: [[EQ0:%.+]] = llvm.call @Z3_mk_and([[CTX]], [[C2]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    %4 = smt.eq %c0_bv4, %c0_bv4, %c0_bv4 : !smt.bv<4>

    // CHECK: [[EQ1:%.+]] = llvm.call @Z3_mk_eq([[CTX]], [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %5 = smt.eq %c0_bv4, %c0_bv4 : !smt.bv<4>

    // CHECK-NEXT: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: [[STORAGE:%.+]] = llvm.alloca [[ONE]] x !llvm.array<3 x ptr> : (i32) -> !llvm.ptr
    // CHECK-NEXT: [[A0:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
    // CHECK-NEXT: [[A1:%.+]] = llvm.insertvalue [[BV0]], [[A0]][0] : !llvm.array<3 x ptr>
    // CHECK-NEXT: [[A2:%.+]] = llvm.insertvalue [[BV0]], [[A1]][1] : !llvm.array<3 x ptr>
    // CHECK-NEXT: [[A3:%.+]] = llvm.insertvalue [[BV0]], [[A2]][2] : !llvm.array<3 x ptr>
    // CHECK-NEXT: llvm.store [[A3]], [[STORAGE]] : !llvm.array<3 x ptr>, !llvm.ptr
    // CHECK-NEXT: [[DISTINCT:%.+]] = llvm.call @Z3_mk_distinct([[CTX]], [[THREE]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    %6 = smt.distinct %c0_bv4, %c0_bv4, %c0_bv4 : !smt.bv<4>

    // CHECK: [[C3:%.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[STORAGE:%.+]] = llvm.alloca [[C1]] x !llvm.array<3 x ptr> : (i32) -> !llvm.ptr
    // CHECK: [[ARR0:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
    // CHECK: [[ARR1:%.+]] = llvm.insertvalue [[EQ0]], [[ARR0]][0] : !llvm.array<3 x ptr>
    // CHECK: [[ARR2:%.+]] = llvm.insertvalue [[EQ1]], [[ARR1]][1] : !llvm.array<3 x ptr>
    // CHECK: [[ARR3:%.+]] = llvm.insertvalue [[DISTINCT]], [[ARR2]][2] : !llvm.array<3 x ptr>
    // CHECK: llvm.store [[ARR3]], [[STORAGE]] : !llvm.array<3 x ptr>, !llvm.ptr
    // CHECK: [[AND:%.+]] = llvm.call @Z3_mk_and([[CTX]], [[C3]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    %7 = smt.and %4, %5, %6

    // CHECK: llvm.call @Z3_solver_assert([[CTX]], [[S]], [[AND]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    smt.assert %7

    // CHECK: llvm.call @Z3_solver_reset([[CTX]], [[S]]) : (!llvm.ptr, !llvm.ptr) -> ()
    smt.reset

    // CHECK: llvm.call @Z3_solver_push([[CTX]], [[S]]) : (!llvm.ptr, !llvm.ptr) -> ()
    smt.push 1

    // CHECK: llvm.call @Z3_solver_push([[CTX]], [[S]]) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @Z3_solver_push([[CTX]], [[S]]) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @Z3_solver_push([[CTX]], [[S]]) : (!llvm.ptr, !llvm.ptr) -> ()
    smt.push 3

    // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @Z3_solver_pop([[CTX]], [[S]], [[CONST1]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    smt.pop 1

    // CHECK: [[CONST5:%.+]] = llvm.mlir.constant(5 : i32)
    // CHECK: llvm.call @Z3_solver_pop([[CTX]], [[S]], [[CONST5]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    smt.pop 5

    // CHECK-DEBUG: [[SOLVER_STR:%.+]] = llvm.call @Z3_solver_to_string({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-DEBUG: [[FMT_STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
    // CHECK-DEBUG: llvm.call @printf([[FMT_STR]], [[SOLVER_STR]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK:   [[CHECK:%.+]] = llvm.call @Z3_solver_check([[CTX]], [[S]]) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:   [[IS_SAT:%.+]] = llvm.icmp "eq" [[CHECK]], [[C1]] : i32
    // CHECK:   llvm.cond_br [[IS_SAT]], ^[[BB1:.+]], ^[[BB2:.+]]
    // CHECK: ^[[BB1]]:
    // CHECK-DEBUG: [[CTX_ADDR:%.+]] = llvm.mlir.addressof @ctx_0 : !llvm.ptr
    // CHECK-DEBUG: [[CTX0:%.+]] = llvm.load [[CTX_ADDR]] : !llvm.ptr -> !llvm.ptr
    // CHECK-DEBUG: [[MODEL:%.+]] = llvm.call @Z3_solver_get_model([[CTX0]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-DEBUG: [[MODEL_STR:%.+]] = llvm.call @Z3_model_to_string([[CTX0]], [[MODEL]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-DEBUG: [[FMT_STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
    // CHECK-DEBUG: llvm.call @printf([[FMT_STR]], [[MODEL_STR]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:   llvm.br ^[[BB7:.+]]([[C1]] : i32)
    // CHECK: ^[[BB2]]:
    // CHECK:   [[CNEG1:%.+]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK:   [[IS_UNSAT:%.+]] = llvm.icmp "eq" [[CHECK]], [[CNEG1]] : i32
    // CHECK:   llvm.cond_br [[IS_UNSAT]], ^[[BB3:.+]], ^[[BB4:.+]]
    // CHECK: ^[[BB3]]:
    // CHECK-DEBUG: [[CTX_ADDR:%.+]] = llvm.mlir.addressof @ctx_0 : !llvm.ptr
    // CHECK-DEBUG: [[CTX1:%.+]] = llvm.load [[CTX_ADDR]] : !llvm.ptr -> !llvm.ptr
    // CHECK-DEBUG: [[PROOF:%.+]] = llvm.call @Z3_solver_get_proof([[CTX1]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-DEBUG: [[PROOF_STR:%.+]] = llvm.call @Z3_ast_to_string([[CTX1]], [[PROOF]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-DEBUG: [[FMT_STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
    // CHECK-DEBUG: llvm.call @printf([[FMT_STR]], [[PROOF_STR]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK:   [[CNEG1:%.+]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK:   llvm.br ^[[BB5:.+]]([[CNEG1:%.+]] : i32)
    // CHECK: ^[[BB4]]:
    // CHECK:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:   llvm.br ^[[BB5]]([[C0]] : i32)
    // CHECK: ^[[BB5]]([[ARG0:%.+]]: i32):
    // CHECK:   llvm.br ^[[BB6:.+]]
    // CHECK: ^[[BB6]]:
    // CHECK:   llvm.br ^[[BB7]]([[ARG0]] : i32)
    // CHECK: ^[[BB7]]({{.+}}: i32):
    // CHECK:   llvm.br
    %8 = smt.check sat {
      %c1 = llvm.mlir.constant(1 : i32) : i32
      smt.yield %c1 : i32
    } unknown {
      %c0 = llvm.mlir.constant(0 : i32) : i32
      smt.yield %c0 : i32
    } unsat {
      %c-1 = llvm.mlir.constant(-1 : i32) : i32
      smt.yield %c-1 : i32
    } -> i32

    // CHECK: [[TRUE:%.+]] = llvm.call @Z3_mk_true({{%[0-9a-zA-Z_]+}}) : (!llvm.ptr) -> !llvm.ptr
    %true = smt.constant true
    // CHECK-NEXT: [[FALSE:%.+]] = llvm.call @Z3_mk_false({{%[0-9a-zA-Z_]+}}) : (!llvm.ptr) -> !llvm.ptr
    %false = smt.constant false

    // CHECK-NEXT: llvm.call @Z3_mk_bvneg({{%[0-9a-zA-Z_]+}}, [[BV0]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.neg %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvadd({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.add %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvmul({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.mul %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvurem({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.urem %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvsrem({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.srem %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvsmod({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.smod %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvudiv({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.udiv %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvsdiv({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.sdiv %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvshl({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.shl %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvlshr({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.lshr %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvashr({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.ashr %c0_bv4, %c0_bv4 : !smt.bv<4>

    // CHECK-NEXT: llvm.call @Z3_mk_bvnot({{%[0-9a-zA-Z_]+}}, [[BV0]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.not %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvand({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.and %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvor({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.or %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvxor({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.xor %c0_bv4, %c0_bv4 : !smt.bv<4>

    // CHECK-NEXT: llvm.call @Z3_mk_concat({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.concat %c0_bv4, %c0_bv4 : !smt.bv<4>, !smt.bv<4>
    // CHECK-NEXT: [[TWO:%.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-NEXT: llvm.call @Z3_mk_extract({{%[0-9a-zA-Z_]+}}, [[THREE]], [[TWO]], [[BV0]]) : (!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
    smt.bv.extract %c0_bv4 from 2 : (!smt.bv<4>) -> !smt.bv<2>
    // CHECK-NEXT: [[TWO:%.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: llvm.call @Z3_mk_repeat({{%[0-9a-zA-Z_]+}}, [[TWO]], [[BV0]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    smt.bv.repeat 2 times %c0_bv4 : !smt.bv<4>

    // CHECK-NEXT: llvm.call @Z3_mk_bvslt({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.cmp slt %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvsle({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.cmp sle %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvsgt({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.cmp sgt %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvsge({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.cmp sge %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvult({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.cmp ult %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvule({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.cmp ule %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvugt({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.cmp ugt %c0_bv4, %c0_bv4 : !smt.bv<4>
    // CHECK-NEXT: llvm.call @Z3_mk_bvuge({{%[0-9a-zA-Z_]+}}, [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.bv.cmp uge %c0_bv4, %c0_bv4 : !smt.bv<4>

    // CHECK-NEXT: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: [[STORAGE:%.+]] = llvm.alloca [[ONE]] x !llvm.array<3 x ptr> : (i32) -> !llvm.ptr
    // CHECK-NEXT: [[A0:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
    // CHECK-NEXT: [[A1:%.+]] = llvm.insertvalue [[TRUE]], [[A0]][0] : !llvm.array<3 x ptr>
    // CHECK-NEXT: [[A2:%.+]] = llvm.insertvalue [[FALSE]], [[A1]][1] : !llvm.array<3 x ptr>
    // CHECK-NEXT: [[A3:%.+]] = llvm.insertvalue [[TRUE]], [[A2]][2] : !llvm.array<3 x ptr>
    // CHECK-NEXT: llvm.store [[A3]], [[STORAGE]] : !llvm.array<3 x ptr>, !llvm.ptr
    // CHECK-NEXT: llvm.call @Z3_mk_or({{%[0-9a-zA-Z_]+}}, [[THREE]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    smt.or %true, %false, %true

    // CHECK-NEXT: llvm.call @Z3_mk_ite({{%[0-9a-zA-Z_]+}}, [[TRUE]], [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.ite %true, %c0_bv4, %c0_bv4 : !smt.bv<4>

    // CHECK-NEXT: llvm.call @Z3_mk_not({{%[0-9a-zA-Z_]+}}, [[TRUE]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.not %true
    // CHECK-NEXT: llvm.call @Z3_mk_xor({{%[0-9a-zA-Z_]+}}, [[TRUE]], [[FALSE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.xor %true, %false
    // CHECK-NEXT: [[V0:%.+]] = llvm.call @Z3_mk_xor({{%[0-9a-zA-Z_]+}}, [[TRUE]], [[FALSE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: [[V1:%.+]] = llvm.call @Z3_mk_xor({{%[0-9a-zA-Z_]+}}, [[V0]], [[TRUE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: llvm.call @Z3_mk_xor({{%[0-9a-zA-Z_]+}}, [[V1]], [[FALSE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.xor %true, %false, %true, %false
    // CHECK-NEXT: llvm.call @Z3_mk_implies({{%[0-9a-zA-Z_]+}}, [[TRUE]], [[FALSE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.implies %true, %false

    // CHECK-NEXT: [[FOUR:%.+]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: [[BV_SORT:%.+]] = llvm.call @Z3_mk_bv_sort({{%[0-9a-zA-Z_]+}}, [[FOUR]]) : (!llvm.ptr, i32) -> !llvm.ptr
    // CHECK-NEXT: [[ARR:%.+]] = llvm.call @Z3_mk_const_array({{%[0-9a-zA-Z_]+}}, [[BV_SORT]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %9 = smt.array.broadcast %c0_bv4 : !smt.array<[!smt.bv<4> -> !smt.bv<4>]>

    // CHECK-NEXT: llvm.call @Z3_mk_select({{%[0-9a-zA-Z_]+}}, [[ARR]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.array.select %9[%c0_bv4] : !smt.array<[!smt.bv<4> -> !smt.bv<4>]>

    // CHECK-NEXT: llvm.call @Z3_mk_store({{%[0-9a-zA-Z_]+}}, [[ARR]], [[BV0]], [[BV0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.array.store %9[%c0_bv4], %c0_bv4 : !smt.array<[!smt.bv<4> -> !smt.bv<4>]>

    // CHECK-NEXT: [[SORT:%.+]] = llvm.call @Z3_mk_int_sort({{%[0-9a-zA-Z_]+}}) : (!llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: [[VAL:%.+]] = llvm.mlir.constant(-123 : i64) : i64
    // CHECK-NEXT: [[C123:%.+]] = llvm.call @Z3_mk_int64({{%[0-9a-zA-Z_]+}}, [[VAL]], [[SORT]]) : (!llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
    %10 = smt.int.constant -123

    // CHECK-NEXT: [[INT_SORT:%.+]] = llvm.call @Z3_mk_int_sort({{%[0-9a-zA-Z_]+}}) : (!llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: [[STR:%.+]] = llvm.mlir.addressof @str{{.*}} : !llvm.ptr
    // CHECK-NEXT: [[NUMERAL:%.+]] = llvm.call @Z3_mk_numeral({{%[0-9a-zA-Z_]+}}, [[STR]], [[INT_SORT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: llvm.call @Z3_mk_unary_minus({{%[0-9a-zA-Z_]+}}, [[NUMERAL]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.int.constant -9223372036854775809

    // CHECK-NEXT: [[SORT:%.+]] = llvm.call @Z3_mk_int_sort({{%[0-9a-zA-Z_]+}}) : (!llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: [[ZERO:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: [[ZERO_VAL:%.+]] = llvm.call @Z3_mk_int64({{%[0-9a-zA-Z_]+}}, [[ZERO]], [[SORT]]) : (!llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: [[COND:%.+]] = llvm.call @Z3_mk_lt({{%[0-9a-zA-Z_]+}}, [[C123]], [[ZERO_VAL]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: [[TWO:%.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: [[STORAGE:%.+]] = llvm.alloca [[ONE]] x !llvm.array<2 x ptr> : (i32) -> !llvm.ptr
    // CHECK-NEXT: [[A0:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
    // CHECK-NEXT: [[A1:%.+]] = llvm.insertvalue [[ZERO_VAL]], [[A0]][0] : !llvm.array<2 x ptr>
    // CHECK-NEXT: [[A2:%.+]] = llvm.insertvalue [[C123]], [[A1]][1] : !llvm.array<2 x ptr>
    // CHECK-NEXT: llvm.store [[A2]], [[STORAGE]] : !llvm.array<2 x ptr>, !llvm.ptr
    // CHECK-NEXT: [[NEG:%.+]] = llvm.call @Z3_mk_sub({{%[0-9a-zA-Z_]+}}, [[TWO]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    // CHECK-NEXT: llvm.call @Z3_mk_ite({{%[0-9a-zA-Z_]+}}, [[COND]], [[NEG]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.int.abs %10

    // CHECK: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: [[ALLOCA:%.+]] = llvm.alloca
    // CHECK: llvm.call @Z3_mk_add({{%[0-9a-zA-Z_]+}}, [[THREE]], [[ALLOCA]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    smt.int.add %10, %10, %10

    // CHECK: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: [[ALLOCA:%.+]] = llvm.alloca
    // CHECK: llvm.call @Z3_mk_mul({{%[0-9a-zA-Z_]+}}, [[THREE]], [[ALLOCA]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    smt.int.mul %10, %10, %10

    // CHECK: [[THREE:%.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: [[ALLOCA:%.+]] = llvm.alloca
    // CHECK: llvm.call @Z3_mk_sub({{%[0-9a-zA-Z_]+}}, [[THREE]], [[ALLOCA]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    smt.int.sub %10, %10

    // CHECK-NEXT: llvm.call @Z3_mk_div({{%[0-9a-zA-Z_]+}}, [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.int.div %10, %10
    // CHECK-NEXT: llvm.call @Z3_mk_mod({{%[0-9a-zA-Z_]+}}, [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.int.mod %10, %10

    // CHECK-NEXT: llvm.call @Z3_mk_le({{%[0-9a-zA-Z_]+}}, [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.int.cmp le %10, %10
    // CHECK-NEXT: llvm.call @Z3_mk_lt({{%[0-9a-zA-Z_]+}}, [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.int.cmp lt %10, %10
    // CHECK-NEXT: llvm.call @Z3_mk_ge({{%[0-9a-zA-Z_]+}}, [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.int.cmp ge %10, %10
    // CHECK-NEXT: llvm.call @Z3_mk_gt({{%[0-9a-zA-Z_]+}}, [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    smt.int.cmp gt %10, %10

    // CHECK: [[WIDTHCONST:%.+]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: llvm.call @Z3_mk_int2bv({{%[0-9a-zA-Z_]+}}, [[WIDTHCONST]], [[C123]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    smt.int2bv %10 : !smt.bv<4>

    // CHECK: [[C0:%.+]] = llvm.mlir.constant(0 : i32)
    // CHECK: [[C2:%.+]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[ZERO:%.+]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @Z3_mk_fresh_const({{[^ ]*}}, [[ZERO]], {{[^ ]*}})
    // CHECK: [[ZERO:%.+]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @Z3_mk_fresh_const({{[^ ]*}}, [[ZERO]], {{[^ ]*}})
    // CHECK: [[BOUND_STORAGE:%.+]] = llvm.alloca {{[^ ]*}} x !llvm.array<2 x ptr>
    // CHECK: llvm.call @Z3_mk_eq
    // CHECK: [[NUM_MULTI_PATTERNS:%.+]] = llvm.mlir.constant(2 : i32)
    // CHECK: llvm.call @Z3_mk_add
    // CHECK: [[NUM_PATTERNS:%.+]] = llvm.mlir.constant(1 : i32)
    // CHECK: [[PATTERN_STORAGE:%.+]] = llvm.alloca {{[^ ]*}} x !llvm.array<1 x ptr>
    // CHECK: llvm.call @Z3_mk_pattern({{[^ ]*}}, [[NUM_PATTERNS]], [[PATTERN_STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    // CHECK: llvm.call @Z3_mk_add
    // CHECK: llvm.call @Z3_mk_sub
    // CHECK: [[NUM_PATTERNS:%.+]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[PATTERN_STORAGE:%.+]] = llvm.alloca {{[^ ]*}} x !llvm.array<2 x ptr>
    // CHECK: llvm.call @Z3_mk_pattern({{[^ ]*}}, [[NUM_PATTERNS]], [[PATTERN_STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    // CHECK: [[MULTI_PATTERN_STORAGE:%.+]] = llvm.alloca {{[^ ]*}} x !llvm.array<2 x ptr> : (i32) -> !llvm.ptr
    // CHECK: llvm.call @Z3_mk_forall_const({{[^ ]*}}, [[C0]], [[C2]], [[BOUND_STORAGE]], [[NUM_MULTI_PATTERNS]], [[MULTI_PATTERN_STORAGE]], {{[^ ]*}}) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %58 = smt.forall {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
      %59 = smt.eq %arg2, %arg3 : !smt.int
      smt.yield %59 : !smt.bool
    } patterns {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
      %59 = smt.int.add %arg2, %arg3
      smt.yield %59 : !smt.int
    }, {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
      %59 = smt.int.add %arg2, %arg3
      %60 = smt.int.sub %arg2, %arg3
      smt.yield %59, %60 : !smt.int, !smt.int
    }

    // CHECK: [[C42:%.+]] = llvm.mlir.constant(42 : i32)
    // CHECK: [[C2:%.+]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[STR:%.+]] = llvm.mlir.addressof @{{[^ ]*}} : !llvm.ptr
    // CHECK: llvm.call @Z3_mk_fresh_const({{[^ ]*}}, [[STR]]
    // CHECK: [[STR:%.+]] = llvm.mlir.addressof @{{[^ ]*}} : !llvm.ptr
    // CHECK: llvm.call @Z3_mk_fresh_const({{[^ ]*}}, [[STR]]
    // CHECK: [[BOUND_STORAGE:%.+]] = llvm.alloca {{[^ ]*}} x !llvm.array<2 x ptr>
    // CHECK: llvm.mlir.undef : !llvm.array<2 x ptr>
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    // CHECK: llvm.store {{[^ ]*}}, [[BOUND_STORAGE]]
    // CHECK: llvm.call @Z3_mk_eq
    // CHECK: [[ZERO:%.+]] = llvm.mlir.constant(0 : i32)
    // CHECK: [[NULL_PTR:%.+]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @Z3_mk_exists_const({{[^ ]*}}, [[C42]], [[C2]], [[BOUND_STORAGE]], [[ZERO]], [[NULL_PTR]], {{[^ ]*}}) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %59 = smt.exists ["a", "b"] weight 42 {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
      %60 = smt.eq %arg2, %arg3 : !smt.int
      smt.yield %60 : !smt.bool
    }

    smt.yield %8 : i32
  }

  // CHECK: llvm.return
  return
}

// CHECK-LABEL:  llvm.func @solver
func.func @test_logic() {
  smt.solver () : () -> () {
    %c0_bv4 = smt.bv.constant #smt.bv<0> : !smt.bv<4>
    smt.set_logic "HORN"
    smt.check sat {} unknown {} unsat {}
    smt.yield
  }
  func.return
}
