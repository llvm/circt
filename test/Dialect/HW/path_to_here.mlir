// RUN: circt-opt --hw-lower-path-to-here --split-input-file --verify-diagnostics %s | FileCheck %s

hw.module @Root() {
  // CHECK: hw.hierpath @sym [@ParentOne::@root]
  hw.path.to_here @sym
}

hw.module @ParentOne() {
  hw.instance "root" @Root() -> ()
}

// -----

hw.module @Root() {
  // CHECK: hw.hierpath @sym [@ParentTwo::@parentOne, @ParentOne::@root]
  hw.path.to_here @sym
}

hw.module @ParentOne() {
  hw.instance "root" @Root() -> ()
}

hw.module @ParentTwo() {
  hw.instance "parentOne" @ParentOne() -> ()
}

// -----

hw.module @NoParent() {
  // expected-error @below {{cannot lower path.to_here ops in modules with no instantiations.}}
  // expected-error @below {{failed to legalize operation 'hw.path.to_here' that was explicitly marked illegal}}
  hw.path.to_here @sym
}

// -----

// Test non-unique instance hierarchy at the first level

hw.module @Root() {
  // expected-error @below {{cannot lower path.to_here ops in module hierarchies with multiple instantiations.}}
  // expected-error @below {{failed to legalize operation 'hw.path.to_here' that was explicitly marked illegal}}
  hw.path.to_here @sym
}

hw.module @FirstUser() {
  // expected-note@below {{instantiated here: hw.instance "root" @Root() -> ()}}
  hw.instance "root" @Root() -> ()
}

hw.module @SecondUser() {
  // expected-note@below {{instantiated here: hw.instance "root" @Root() -> ()}}
  hw.instance "root" @Root() -> ()
}

hw.module @Top() {
  hw.instance "firstUser" @FirstUser() -> ()
  hw.instance "secondUser" @SecondUser() -> ()
}

// -----

// Test non-unique instance hierarchy at the second level.

hw.module @Root() {
  // expected-error @below {{cannot lower path.to_here ops in module hierarchies with multiple instantiations.}}
  // expected-error @below {{failed to legalize operation 'hw.path.to_here' that was explicitly marked illegal}}
  hw.path.to_here @sym
}

hw.module @RootParent() {
  hw.instance "root" @Root() -> ()
}


hw.module @FirstUser() {
  // expected-note@below {{instantiated here: hw.instance "rp" @RootParent() -> ()}}
  hw.instance "rp" @RootParent() -> ()
}

hw.module @SecondUser() {
  // expected-note@below {{instantiated here: hw.instance "rp" @RootParent() -> ()}}
  hw.instance "rp" @RootParent() -> ()
}

hw.module @Top() {
  hw.instance "firstUser" @FirstUser() -> ()
  hw.instance "secondUser" @SecondUser() -> ()
}
