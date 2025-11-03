firrtl.circuit "Property" { 
  // CHECK: firrtl.module @Property(in %a: !firrtl.string)
  firrtl.module @Property(in %a: !firrtl.string) { }
}