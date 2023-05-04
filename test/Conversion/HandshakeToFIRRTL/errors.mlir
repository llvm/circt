// RUN: circt-opt -lower-handshake-to-firrtl -verify-diagnostics -split-input-file %s

// Test cycle through a component
// expected-error @+1 {{'builtin.module' op cannot deduce top level function - cycle detected in instance graph (bar->baz->foo->bar).}}
module {
  handshake.func @bar(%ctrl : none) -> (none) {
    %0 = handshake.instance @baz(%ctrl) : (none) -> (none)
    return %0: none
  }

  handshake.func @foo(%ctrl : none) -> (none) {
    %0 = handshake.instance @bar(%ctrl) : (none) -> (none)
    return %0: none  
  }
  
  handshake.func @baz(%ctrl : none) -> (none) {
    %0 = handshake.instance @foo(%ctrl) : (none) -> (none)
    return %0: none  
  }
}

// -----

// test multiple candidate top components
// expected-error @+1 {{'builtin.module' op multiple candidate top-level modules detected (bar, foo). Please remove one of these from the source program.}}
module {
  handshake.func @bar(%ctrl : none) -> (none) {
    %0 = handshake.instance @baz(%ctrl) : (none) -> (none)
    return %0: none
  }

  handshake.func @foo(%ctrl : none) -> (none) {
    %0 = handshake.instance @baz(%ctrl) : (none) -> (none)
    return %0: none  
  }
  
  handshake.func @baz(%ctrl : none) -> (none) {
    return %ctrl: none  
  }
}

// -----

// Test multiple use values
handshake.func @test_mu(%arg0: i32, %arg1 : i1, %arg2: none) -> (i32, i32, none) {
  // expected-error @+1 {{'handshake.cond_br' op result 1 has no uses.}}
  %true, %false = cond_br %arg1, %arg0 : i32
  return %true, %arg0, %arg2 : i32, i32, none
}
