# REQUIRES: bindings_tcl
# RUN: tclsh %s -- %TCL_PATH% %CIRCT_SOURCE% | FileCheck %s

load [lindex $argv 1]libcirct-tcl[info sharedlibextension]
set circuit [circt load MLIR [lindex $argv 2]/integration_test/Bindings/Tcl/Inputs/simple.mlir]

foreach op [circt query [usage *] [circt query [inst * [op wire]] $circuit]] {
  puts $op
# CHECK: sv.passign %0, %false : i1
}

foreach op [circt query [usage *] $circuit] {
  puts $op
# CHECK: %owo1.owo_result = hw.instance "owo1" @owo() : () -> i32
# CHECK: hw.instance "uwu1" @uwu() : () -> ()
# CHECK: hw.instance "nya1" @nya(%owo1.owo_result) : (i32) -> ()
}

