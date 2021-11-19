// RUN: TESTNAME=comp

// === Lower testbench to LLVMIR
// RUN:   mlir-opt %s -convert-scf-to-std                                      \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts > ${TESTNAME}_tb.mlir

// === Build handshake simulator
// RUN: mlir-opt --convert-scf-to-std %s.kernel                                \
// RUN: | circt-opt -lower-std-to-handshake -canonicalize='top-down=true       \
// RUN:     region-simplify=true' -handshake-insert-buffer='strategies=all' > ${TESTNAME}_handshake.mlir
// RUN: circt-opt -lower-handshake-to-firrtl ${TESTNAME}_handshake.mlir > ${TESTNAME}_handshake_firrtl.mlir
// RUN: firtool --format=mlir --lower-to-hw --verilog ${TESTNAME}_handshake_firrtl.mlir > ${TESTNAME}.sv
// RUN: hlt-wrapgen --ref %s.kernel --kernel ${TESTNAME}_handshake_firrtl.mlir --name ${TESTNAME} --type=handshakeFIRRTL -o .
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


func private @comp_call(i32, i32) -> ()
func private @comp_await() -> (i32)
func private @printI64(i32)
func private @printComma()
func @test_comp() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32

  // Call
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = arith.index_cast %i : index to i32
    call @comp_call(%0, %c5_i32) : (i32, i32) -> ()
  }

  // Await
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = arith.index_cast %i : index to i32
    %res = call @comp_await() : () -> (i32)
    call @printI64(%res) : (i32) -> ()
    call @printComma() : () -> ()
  }

  // CHECK: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 

  return %c0_i32 : i32
}