// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg '@Foo' --include firrtl-layer-enable-remover --include firrtl-layer-disable --keep-best=0 | FileCheck %s --check-prefix=ALL-LAYERS --implicit-check-not='firrtl.layer' --implicit-check-not='layers ='
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg '@A::@B' --include firrtl-layer-enable-remover --keep-best=0 | FileCheck %s --check-prefix=KEEP-LAYER-B

// ALL-LAYERS-LABEL: firrtl.circuit "Foo"
// ALL-LAYERS: firrtl.module @Foo()
// ALL-LAYERS-NEXT: }

// KEEP-LAYER-B-LABEL: firrtl.circuit "Foo"
// KEEP-LAYER-B: firrtl.module @Foo() attributes {layers = [@A::@B]}
// KEEP-LAYER-B-NEXT: }
firrtl.circuit "Foo" {
  firrtl.layer @A bind attributes {output_file = #hw.output_file<"a/">, sym_visibility = "private"} {
    firrtl.layer @B bind attributes {output_file = #hw.output_file<"a/b/">} {
    }
    firrtl.layer @C bind attributes {output_file = #hw.output_file<"a/c/">} {
    }
  }
  firrtl.module @Foo() attributes {layers = [@A, @A::@B, @A::@C]} {
  }
}
