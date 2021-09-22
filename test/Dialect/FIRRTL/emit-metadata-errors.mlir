// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-emit-metadata)' --verify-diagnostics --split-input-file %s

//===----------------------------------------------------------------------===//
// RetimeModules
//===----------------------------------------------------------------------===//

// expected-error @+1 {{sifive.enterprise.firrtl.RetimeModulesAnnotation requires a filename}}
firrtl.circuit "NoFilename" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation"
  }]} {
  firrtl.module @NoFilename() { }
}

// -----

// expected-error @+1 {{sifive.enterprise.firrtl.RetimeModulesAnnotation requires a non-empty filename}}
firrtl.circuit "EmptyFilename" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = ""
  }]} {
  firrtl.module @EmptyFilename() { }
}

// -----

// expected-error @+1 {{more than one sifive.enterprise.firrtl.RetimeModulesAnnotation annotation attached}}
firrtl.circuit "MultipleAnnotations" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = "test0.json"
  }, {
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = "test1.json"
  }]} {
  firrtl.module @MultipleAnnotations() { }
}

