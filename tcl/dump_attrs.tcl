proc dump_attrs { attrs { indent 0 } } {
  # From https://stackoverflow.com/a/7428623
  # Guess the type of the value; deep *UNSUPPORTED* magic!
  regexp {^value is a (.*?) with a refcount} [::tcl::unsupported::representation $attrs] -> type

  switch $type {
    dict {
      set result "{"
      set pfx ""
      dict for {k v} $attrs {
        append result $pfx "\n"
        for {set i 0} {$i < [expr $indent + 1]} {incr i} {
          append result "    "
        }

        append result [dump_attrs $k] ": " [dump_attrs $v [expr $indent + 1]]
        set pfx ", "
      }
      append result "\n"
      for {set i 0} {$i < $indent} {incr i} {
        append result "    "
      }
      return [append result "}"]
    }
    list {
      set result "\["
      set pfx ""
      foreach v $attrs {
        append result $pfx "\n"
        for {set i 0} {$i < [expr $indent + 1]} {incr i} {
          append result "    "
        }

        append result [dump_attrs $v [expr $indent + 1]]
        set pfx ", "
      }
      append result "\n"
      for {set i 0} {$i < $indent} {incr i} {
        append result "    "
      }
      return [append result "\]"]
    }
    int - double {
      return [expr {$attrs}]
    }
    booleanString {
      return [expr {$attrs ? "true" : "false"}]
    }
    default {
      # Some other type; do some guessing...
      if {[string is integer -strict $attrs]} {
        return [expr {$attrs}]
      } elseif {[string is double -strict $attrs]} {
        return [expr {$attrs}]
      } elseif {[string is boolean -strict $attrs]} {
        return [expr {$attrs ? "true" : "false"}]
      }
    }
  }

  set value [regsub -all "\"" [regsub -all "\n" $attrs "\\n"] "\\\""]
  return "\"$value\""
}

load [lindex $argv 1]libcirct-tcl[info sharedlibextension]
set circuit [circt load MLIR [lindex $argv 2]]
set attrs [circt get attrs [circt query ** $circuit]]
puts [dump_attrs $attrs]
