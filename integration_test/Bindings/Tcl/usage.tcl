# REQUIRES: bindings_tcl
# RUN: tclsh %s -- %TCL_PATH% %CIRCT_SOURCE% | FileCheck %s

load [lindex $argv 1]libcirct-tcl[info sharedlibextension]
set circuit [circt load MLIR [lindex $argv 2]/integration_test/Bindings/Tcl/Inputs/simple.mlir]

foreach op [circt query [usage *] [circt query [inst * [op wire]] $circuit]] {
  puts $op
}

# CHECK: sv.passign %0, %false : i1
