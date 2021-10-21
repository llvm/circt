// An example of how the Std wrapper can be used to run tests with a standard
// dialect implementations of a kernel.

// RUN: TESTNAME=addi

// === Lower testbench to LLVMIR
// RUN:   mlir-opt %s -convert-scf-to-std                                      \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts > ${TESTNAME}_tb.mlir

// Build standard simulator
// RUN: hlt-wrapgen --ref %s.kernel --kernel %s.kernel --name ${TESTNAME} -o .
// RUN: mlir-opt %s.kernel -convert-scf-to-std                                 \
// RUN:                    -convert-std-to-llvm                                \
// RUN: | mlir-translate --mlir-to-llvmir -o=${TESTNAME}.ll
// RUN: cp %circt_obj_root/tools/hlt/Simulator/hlt_std_CMakeLists.txt CMakeLists.txt
// RUN: cmake -DHLT_TESTNAME=${TESTNAME} -DCMAKE_BUILD_TYPE=RelWithDebInfo . 
// RUN: make all -j$(nproc)

// JIT execute the testbench
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
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %c1_i32 = constant 1 : i32
  %c0_i32 = constant 0 : i32

  // Call
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = index_cast %i : index to i32
    call @addi_call(%c1_i32, %0) : (i32, i32) -> ()
  }

  // Await
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = index_cast %i : index to i32
    %res = call @addi_await() : () -> (i32)
    call @printI64(%res) : (i32) -> ()
    call @printComma() : () -> ()
  }

  // CHECK: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

  return %c0_i32 : i32
}