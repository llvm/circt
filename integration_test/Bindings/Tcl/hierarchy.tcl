# REQUIRES: bindings_tcl
# RUN: tclsh %s -- %TCL_PATH% %CIRCT_SOURCE% | FileCheck %s

proc print_hierarchy { ops {indent 0} } {
  set query [circt query * $ops]
  foreach {op} $query {
    for {set i 0} {$i < $indent} {incr i} {
      puts -nonewline "    "
    }

    puts -nonewline [circt get opname $op]
    puts -nonewline " "
    puts [circt get modname $op]

    print_hierarchy $op [expr $indent + 1]
  }
}

load [lindex $argv 1]libcirct-tcl[info sharedlibextension]
set circuit [circt load MLIR [lindex $argv 2]/integration_test/Bindings/Tcl/Inputs/simple.mlir]
print_hierarchy $circuit

# CHECK: module.extern ichi
# CHECK: module owo
# CHECK:     constant
# CHECK:     output
# CHECK: module uwu
# CHECK:     output
# CHECK: module nya
# CHECK:     module uwu
# CHECK:         output
# CHECK:     output
# CHECK: module test
# CHECK:     wire
# CHECK:     module owo
# CHECK:         constant
# CHECK:         output
# CHECK:     module nya
# CHECK:         module uwu
# CHECK:             output
# CHECK:         output
# CHECK:     output
# CHECK: module always
# CHECK:     constant
# CHECK:     wire
# CHECK:     constant
# CHECK:     alwaysff
# CHECK:         passign
# CHECK:     output
