// RUN: circt-opt --hw-lower-hierpathto-ops --split-input-file --verify-diagnostics %s | FileCheck %s

hw.module @Root() {
  // CHECK: hw.hierpath @sym [@ParentOne::@root, @Root::@w]
  hw.hierpath.to @sym(@w)
  %c0 = hw.constant false
  hw.wire %c0 sym @w : i1
}

hw.module @ParentOne() {
  hw.instance "root" @Root() -> ()
}

// -----

hw.module @Root() {
  // CHECK: hw.hierpath @sym [@ParentTwo::@parentOne, @ParentOne::@root, @Root::@w]
  hw.hierpath.to @sym(@w)
  %c0 = hw.constant false
  hw.wire %c0 sym @w : i1
}

hw.module @ParentOne() {
  hw.instance "root" @Root() -> ()
}

hw.module @ParentTwo() {
  hw.instance "parentOne" @ParentOne() -> ()
}

// -----

// CHECK-LABEL:   hw.module @NoParent() {
// CHECK:           hw.hierpath @sym [@NoParent::@w]
hw.module @NoParent() {
  hw.hierpath.to @sym(@w)
  %c0 = hw.constant false
  hw.wire %c0 sym @w : i1
}

// -----

// Test non-unique instance hierarchy at the first level

hw.module @Root() {
  // expected-error @below {{cannot lower hierpath.to ops in module hierarchies with multiple instantiations.}}
  // expected-error @below {{failed to legalize operation 'hw.hierpath.to' that was explicitly marked illegal}}
  hw.hierpath.to @sym(@w)
  %c0 = hw.constant false
  hw.wire %c0 sym @w : i1
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
  // expected-error @below {{cannot lower hierpath.to ops in module hierarchies with multiple instantiations.}}
  // expected-error @below {{failed to legalize operation 'hw.hierpath.to' that was explicitly marked illegal}}
  hw.hierpath.to @sym(@w)
  %c0 = hw.constant false
  hw.wire %c0 sym @w : i1
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
