// RUN: TESTNAME=addi

// === Lower testbench to LLVMIR
// RUN:   mlir-opt %s -convert-scf-to-std                                      \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts > ${TESTNAME}_tb.mlir

// === Build handshake simulator
// RUN: circt-opt -lower-std-to-handshake %s.kernel                            \
// RUN: | tee ${TESTNAME}_handshake.mlir                                       \
// RUN: | circt-opt -canonicalize='top-down=true region-simplify=true' -handshake-insert-buffer='strategies=all' -lower-handshake-to-firrtl \
// RUN: | firtool --format=mlir --lower-to-hw --verilog > ${TESTNAME}.sv
// RUN: hlt-wrapgen --ref %s.kernel --kernel ${TESTNAME}_handshake.mlir --name ${TESTNAME} -o .
// RUN: cp %circt_obj_root/tools/hlt/Simulator/hlt_verilator_CMakeLists.txt CMakeLists.txt
// RUN: cmake -DHLT_TESTNAME=${TESTNAME} -DCMAKE_BUILD_TYPE=RelWithDebInfo . 
// RUN: make all -j$(nproc)

// === JIT execute the testbench
// RUN: mlir-cpu-runner                                                        \
// RUN:     -e test_${TESTNAME} -entry-point-result=i32 -O3                    \
// RUN:     -shared-libs=%llvm_shlibdir/libmlir_c_runner_utils%shlibext        \
// RUN:     -shared-libs=%llvm_shlibdir/libmlir_runner_utils%shlibext          \
// RUN:     -shared-libs=libhlt_${TESTNAME}%shlibext ${TESTNAME}_tb.mlir       \
// RUN: | FileCheck %s

func private @addi_call(i32, i32) -> ()
func private @addi_await() -> (i32)
func private @printI64(i32)   
func private @printComma()
func @test_addi() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32

  // Call
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = arith.index_cast %i : index to i32
    call @addi_call(%c1_i32, %0) : (i32, i32) -> ()
  }

  // Await
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = arith.index_cast %i : index to i32
    %res = call @addi_await() : () -> (i32)
    call @printI64(%res) : (i32) -> ()
    call @printComma() : () -> ()
  }

  // CHECK: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

  return %c0_i32 : i32
}


