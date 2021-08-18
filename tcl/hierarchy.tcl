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
set circuit [circt load MLIR [lindex $argv 2]]
print_hierarchy $circuit
