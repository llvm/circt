// RUN: arcilator --emit-mlir %s | FileCheck %s

hw.module @Top(in %clk : !seq.clock) {
  sim.triggered %clk {
    %stdout = sim.stdout_stream
    %stdout_msg = sim.fmt.literal "stdout\n"
    sim.proc.print %stdout_msg to %stdout

    %stderr = sim.stderr_stream
    %stderr_msg = sim.fmt.literal "stderr\n"
    sim.proc.print %stderr_msg to %stderr
  }
}

// CHECK-DAG: llvm.func @arcRuntimeIR_formatToStream(!llvm.ptr, !llvm.ptr, ...)
// CHECK-DAG: llvm.func @arcRuntimeIR_getStdoutStream() -> !llvm.ptr
// CHECK-DAG: llvm.func @arcRuntimeIR_getStderrStream() -> !llvm.ptr
// CHECK: llvm.call @arcRuntimeIR_getStdoutStream() : () -> !llvm.ptr
// CHECK: llvm.call @arcRuntimeIR_formatToStream
// CHECK-SAME: vararg(!llvm.func<void (ptr, ptr, ...)>)
// CHECK: llvm.call @arcRuntimeIR_getStderrStream() : () -> !llvm.ptr
// CHECK: llvm.call @arcRuntimeIR_formatToStream
// CHECK-SAME: vararg(!llvm.func<void (ptr, ptr, ...)>)
