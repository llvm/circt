// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-recursive-instantiation))' %s --verify-diagnostics --split-input-file

firrtl.circuit "SelfLoop0" {
  // expected-error @below {{recursive instantiation}}
  firrtl.module @SelfLoop0() {
    // expected-note @below {{SelfLoop0 instantiates SelfLoop0 here}}
    firrtl.instance inst @SelfLoop0()
  }
}

// -----

firrtl.circuit "SelfLoop1" {
  // expected-error @below {{recursive instantiation}}
  firrtl.module @SelfLoop1() {
    // expected-note @below {{SelfLoop1 instantiates SelfLoop1 here}}
    firrtl.instance inst @SelfLoop1()
    // expected-note @below {{SelfLoop1 instantiates SelfLoop1 here}}
    firrtl.instance inst @SelfLoop1()
  }
}

// -----

firrtl.circuit "TwoLoops" {
  // expected-error @below {{recursive instantiation}}
  firrtl.module @TwoLoops() {
    // expected-note @below {{TwoLoops instantiates TwoLoops here}}
    firrtl.instance inst @TwoLoops()
    firrtl.instance inst @OtherModule()
  }
  // expected-error @below {{recursive instantiation}}
  firrtl.module @OtherModule() {
    // expected-note @below {{OtherModule instantiates OtherModule here}}
    firrtl.instance inst @OtherModule()
  }
}

// -----

firrtl.circuit "MutualLoop" {
  firrtl.module @MutualLoop() {
    firrtl.instance a @A()
  }
  firrtl.module @A() {
    // expected-note @below {{A instantiates B here}}
    firrtl.instance b @B()
  }
  // expected-error @below {{recursive instantiation}}
  firrtl.module @B() {
    // expected-note @below {{B instantiates A here}}
    firrtl.instance a @A()
  }
}

// -----

// Should disallow recursive class instantiation.
firrtl.circuit "Classes" {
  firrtl.module @Classes() {
    firrtl.object @RecursiveClass()
  }

  // expected-error @below {{recursive instantiation}}
  firrtl.class @RecursiveClass() {
    // expected-note @below {{RecursiveClass instantiates RecursiveClass here}}
    %0 = firrtl.object @RecursiveClass()
  }
}

// -----

firrtl.circuit "A" {
  firrtl.module @A() {
    // expected-note @below {{A instantiates B here}}
    firrtl.instance b @B()
  }
  firrtl.module @B() {
    // expected-note @below {{B instantiates C here}}
    firrtl.instance c @C()
    // expected-note @below {{B instantiates A here}}
    firrtl.instance a @A()
  }
  // expected-error @below {{recursive instantiation}}
  firrtl.module @C() {
    // expected-note @below {{C instantiates B here}}
    firrtl.instance b @B()
  }
}

// -----

firrtl.circuit "SelfLoop" {
  firrtl.option @Opt {
      firrtl.option_case @A
  }
  firrtl.module @Mod() {}

  // expected-error @below {{recursive instantiation}}
  firrtl.module @SelfLoop() {
    // expected-note @below {{SelfLoop instantiates SelfLoop here}}
    firrtl.instance_choice inst interesting_name @SelfLoop alternatives @Opt { @A -> @Mod }()
    // expected-note @below {{SelfLoop instantiates SelfLoop here}}
    firrtl.instance_choice inst interesting_name @Mod alternatives @Opt { @A -> @SelfLoop }()
  }
}
