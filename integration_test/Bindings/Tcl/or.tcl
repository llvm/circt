# REQUIRES: bindings_tcl
# RUN: tclsh %s -- %TCL_PATH% %CIRCT_SOURCE% | FileCheck %s

load [lindex $argv 1]libcirct-tcl[info sharedlibextension]
set circuit [circt load MLIR [lindex $argv 2]/integration_test/Bindings/Tcl/Inputs/simple.mlir]

foreach op [circt query [or [op wire] [attr resultNames c]] [circt query ** $circuit]] {
  puts $op
# CHECK: %myArray1 = sv.wire  : !hw.inout<array<42xi8>>
# CHECK: %0 = sv.wire  : !hw.inout<i1>
}
