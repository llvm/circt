# REQUIRES: bindings_tcl
# RUN: tclsh %s -- %TCL_PATH% %CIRCT_SOURCE% | FileCheck %s

load [lindex $argv 1]libcirct-tcl[info sharedlibextension]
set circuit [circt load MLIR [lindex $argv 2]/integration_test/Bindings/Tcl/Inputs/simple.mlir]

foreach op [circt query [and [inst ** [op wire]] *] $circuit] {
  puts $op

# CHECK: hw.module @test() -> (%test_result: i32) {
# CHECK:   %myArray1 = sv.wire  : !hw.inout<array<42xi8>>
# CHECK:   %owo1.owo_result = hw.instance "owo1" @owo() : () -> i32
# CHECK:   hw.instance "nya1" @nya(%owo1.owo_result) : (i32) -> ()
# CHECK:   hw.output %owo1.owo_result : i32
# CHECK: }
# CHECK: hw.module @always() {
# CHECK:   %true = hw.constant true
# CHECK:   %0 = sv.wire  : !hw.inout<i1>
# CHECK:   %false = hw.constant false
# CHECK:   sv.alwaysff(posedge %true)  {
# CHECK:     sv.passign %0, %false : i1
# CHECK:   }
# CHECK:   hw.output
# CHECK: }
}
