# REQUIRES: bindings_tcl
# RUN: tclsh %s -- %TCL_PATH%
load [lindex $argv 1]libcirct-tcl[info sharedlibextension]

set circuit [circt load MLIR [lindex $argv 1]../../tcl/simple.mlir]
puts $circuit
