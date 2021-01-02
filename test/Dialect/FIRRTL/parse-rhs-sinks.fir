; RUN: circt-translate -parse-fir -verify-diagnostics -split-input-file %s | FileCheck %s

circuit Foo :
  module Foo :
    output a : UInt<1>
    output b : UInt<1>
    a is invalid
    b <= a

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.connect %a, [[WIRE]]
    ; CHECK: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK: firrtl.connect %b, [[WIRE]]

    ; // -----

circuit Foo :
  module Foo :
    output a : UInt<1>
    output b : UInt<1>
    a <= UInt<1>("h0")
    b <= a

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.connect %a, [[WIRE]]
    ; CHECK: [[ZERO:%.+]] = firrtl.constant
    ; CHECK: firrtl.connect [[WIRE]], [[ZERO]]
    ; CHECK: firrtl.connect %b, [[WIRE]]

    ; // -----

circuit Foo :
  module Foo :
    output a : UInt<1>
    output b : UInt<1>
    input c : UInt<1>
    a <- c
    b <- a

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.connect %a, [[WIRE]]
    ; CHECK: firrtl.partialconnect [[WIRE]], %c
    ; CHECK: firrtl.partialconnect %b, [[WIRE]]

    ; // -----

circuit Foo :
  module Foo :
    output a : { b : UInt<1>, c : UInt<1>}
    output b : UInt<1>
    a is invalid
    b <= a.b

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<bundle<b: uint<1>, c: uint<1>>>
    ; CHECK: firrtl.connect %a, [[WIRE]]
    ; CHECK: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK: [[SUBFIELD:%.+]] = firrtl.subfield [[WIRE]]("b")
    ; CHECK: firrtl.connect %b, [[SUBFIELD]]

    ; // -----

circuit Foo :
  module Foo :
    input a : { b : UInt<1>, flip c : UInt<1>}
    output b : { b : UInt<1>, flip c : UInt<1>}
    output c : UInt<1>
    b <= a
    c <= b.b

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.bundle<b: flip<uint<1>>, c: uint<1>>
    ; CHECK: firrtl.connect %b, [[WIRE]]
    ; CHECK; firrtl.connect [[WIRE]], %a
    ; CHECK: [[SUBFIELD:%.+]] = firrtl.subfield [[WIRE]]("b")
    ; CHECK: firrtl.connect %c, [[SUBFIELD]]

    ; // -----

circuit Foo :
  module Foo :
    input a : { b : UInt<1>, flip c : UInt<1>}
    output b : { b : UInt<1>, flip c : UInt<1>}
    output c : UInt<1>
    b.b <= a.b
    a.c <= b.c
    c <= b.b

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.bundle<b: flip<uint<1>>, c: uint<1>>
    ; CHECK: firrtl.connect %b, [[WIRE]]
    ; CHECK-DAG: [[WIRE_B:%.+]] = firrtl.subfield [[WIRE]]("b")
    ; CHECK-DAG: [[A_B:%.+]] = firrtl.subfield %a("b")
    ; CHECK: firrtl.connect [[WIRE_B]], [[A_B]]
    ; CHECK-DAG: [[WIRE_C:%.+]] = firrtl.subfield [[WIRE]]("c")
    ; CHECK-DAG: [[A_C:%.+]] = firrtl.subfield %a("c")
    ; CHECK: firrtl.connect [[A_C]], [[WIRE_C]]
    ; CHECK-DAG: [[WIRE_B:%.+]] = firrtl.subfield [[WIRE]]("b")
    ; CHECK: firrtl.connect %c, [[WIRE_B]]

    ; // -----

circuit Foo :
  module Foo :
    output a : { a : UInt<1>, flip b : UInt<1>}
    output b : { a : UInt<1>}
    b is invalid
    a <- b

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<bundle<a: uint<1>>>
    ; CHECK: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK: firrtl.partialconnect %a, [[WIRE]]

    ; // -----

circuit Foo :
  module Foo :
    input cond : UInt<1>
    output a : UInt<1>
    output b : UInt<1>
    when eq(cond, UInt<1>("h0")) :
      a <= UInt<1>("h0")
    else :
      when eq(cond, UInt<1>("h1")) :
        a <= UInt<1>("h1")
      else :
        a <= UInt<1>("h0")
    b <= a

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.connect %a, [[WIRE]]
    ; CHECK-COUNT-3: firrtl.connect [[WIRE]]
    ; CHECK: firrtl.connect %b, [[WIRE]]

    ; // -----

circuit Foo :
  module Foo :
    input a : { flip a: UInt<1> }
    node b = a.a

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<bundle<a: uint<1>>>
    ; CHECK: firrtl.connect %a, [[WIRE]]
    ; CHECK: [[WIRE_a:%.+]] = firrtl.subfield [[WIRE]]("a")
    ; CHECK: [[WIRE_a_passive:%.+]] = firrtl.asPassive [[WIRE_a]]
    ; CHECK; firrtl.node [[WIRE_passive]]

    ; // -----

circuit Foo :
  module Foo :
    input a : UInt<2>
    output b : UInt<1>
    output c : UInt<1>
    b is invalid
    c <= not(b)

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.connect %b, [[WIRE]]
    ; CHECK: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK: [[A:%.+]] = firrtl.asPassive [[WIRE]]
    ; CHECK: [[B:%.+]] = firrtl.not [[A]]
    ; CHECK: firrtl.connect %c, [[B]]

    ; // -----

circuit Foo :
  module Foo :
    input a : UInt<1>[2]
    output b : UInt<1>
    output c : UInt<1>
    b is invalid
    c <= a[b]

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.connect %b, [[WIRE]]
    ; CHECK: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK: %1 = firrtl.asPassive [[WIRE]]
    ; CHECK: firrtl.subaccess %a[%1]

    ; // -----

circuit Foo :
  module Foo :
    input decoupled : { flip ready : UInt<1>, valid : UInt<1>}
    decoupled.ready is invalid
    node c = and(decoupled.ready, decoupled.valid)

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.bundle<ready: flip<uint<1>>, valid: uint<1>>
    ; CHECK: firrtl.connect %decoupled, [[WIRE]]
    ; CHECK: [[WIRE_ready:%.+]] = firrtl.subfield [[WIRE]]("ready")
    ; CHECK: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE_ready]], [[INVALID]]
    ; CHECK: [[WIRE_ready:%.+]] = firrtl.subfield [[WIRE]]("ready")
    ; CHECK: [[WIRE_ready_passive:%.+]] = firrtl.asPassive [[WIRE_ready]]
    ; CHECK: [[WIRE_valid:%.+]] = firrtl.subfield [[WIRE]]("valid")
    ; CHECK: [[AND:%.+]] = firrtl.and [[WIRE_ready_passive]], [[WIRE_valid]]
    ; CHECK: %c = firrtl.node [[AND]]

    ; // -----

circuit Foo :
  module Foo :
    output a : { a : UInt<1>}
    output b : { a : UInt<1>}
    wire w : { a : UInt<1>}
    b is invalid
    w <= b
    a <= w

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<bundle<a: uint<1>>>
    ; CHECK: firrtl.connect %b, [[WIRE]]
    ; CHECK: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK: firrtl.connect %w, [[WIRE]]
    ; CHECK: firrtl.connect %a, %w

    ; // -----

circuit Foo :
  module Foo :
    output a : { a : UInt<1>}
    output b : { a : UInt<1>}
    input clk : Clock
    reg r : { a : UInt<1>}, clk with :
      reset => (UInt<1>("h0"), r)
    b is invalid
    r <= b
    a <= r

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<bundle<a: uint<1>>>
    ; CHECK: firrtl.connect %b, [[WIRE]]
    ; CHECK: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK: firrtl.connect %r, [[WIRE]]
    ; CHECK: firrtl.connect %a, %r

    ; // -----

circuit Foo :
  module Foo :
    input clkIn : Clock
    output clkOut : Clock
    reg r : UInt<1>, clkOut with :
      reset => (UInt<1>("h0"), r)
    clkOut <= clkIn

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<clock>
    ; CHECK: firrtl.connect %clkOut, [[WIRE]]
    ; CHECK: [[WIRE_passive:%.+]] = firrtl.asPassive [[WIRE]]
    ; CHECK: %r = firrtl.reg [[WIRE_passive]]

    ; // -----

circuit Foo :
  module Foo :
    output a : AsyncReset
    input clk : Clock
    reg r : UInt<1>, clk with :
      reset => (a, UInt<1>("h0"))
    a is invalid
    r is invalid

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<asyncreset>
    ; CHECK: firrtl.connect %a, [[WIRE]]
    ; CHECK: [[WIRE_passive:%.+]] = firrtl.asPassive [[WIRE]]
    ; CHECK: firrtl.regreset %clk, [[WIRE_passive]]

    ; // -----

circuit Foo :
  module Foo :
    output a : UInt<1>
    input clk : Clock
    input rst : UInt<1>
    reg r : UInt<1>, clk with :
      reset => (rst, a)
    a is invalid
    r is invalid

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.connect %a, [[WIRE]]
    ; CHECK: [[WIRE_passive]] = firrtl.asPassive [[WIRE]]
    ; CHECK: firrtl.regreset %clk, %rst, [[WIRE_passive]]

    ; // -----

circuit Foo :
  module Foo :
    input in : { flip ready : UInt<1>, valid : UInt<1>}
    output out : { flip ready : UInt<1>, valid : UInt<1>}
    out <= in

    ; COM: This test should not be changed. The connect is bi-directional.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    input a : { a : UInt<1>}
    output b : { a : UInt<1>}
    b <= a

    ; COM: This test should not be changed. The connect is legal.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    output a : { flip a : UInt<1>}
    input b : { flip a : UInt<1>}
    a <= b

    ; COM: This test should not be changed. The connect is legally backwards.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    input a : { a : UInt<1>}
    output b : { a : UInt<1>}
    b <- a

    ; COM: This test should not be changed. The connect is legal.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    output a : { flip a : UInt<1>}
    input b : { flip a : UInt<1>}
    a <- b

    ; COM: This test should not be changed. The connect is legally backwards.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    input a : { a : UInt<1>}
    output b : { a : UInt<1>}
    wire w: { a: UInt<1>}
    w <= a
    b <= w

    ; COM: This test should not be changed. The connect is legal.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: %0 = firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    output a : { flip a : UInt<1>}
    input b : { flip a : UInt<1>}
    wire w: { flip a: UInt<1>}
    a <= w
    w <= b

    ; COM: This test should not be changed. The connect is legally backwards with an intermediary wire.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: %0 = firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    input a : { a : UInt<1>}
    output b : { a : UInt<1>}
    wire w: { a: UInt<1>}
    w <- a
    b <- w

    ; COM: This test should not be changed. The connect is legal.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: %0 = firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    output a : { flip a : UInt<1>}
    input b : { flip a : UInt<1>}
    wire w: { flip a: UInt<1>}
    a <- w
    w <- b

    ; COM: This test should not be changed. The connect is legally backwards with an intermediary wire.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: %0 = firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    input a: { flip a: UInt<1> }
    a is invalid
    wire w : { flip a: UInt<1>, b: UInt<1> }
    w is invalid
    w <- a

    ; COM: This test should not result in a wire being created.
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: %0 = firrtl.wire

    ; // -----

circuit Foo :
  module Foo :
    output a: { a: UInt<1> }
    a is invalid
    wire w : { a: UInt<1>, flip b: UInt<1> }
    w is invalid
    w <- a

    ; COM: This test should result in a wire being created
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-COUNT-2: firrtl.wire

    ; // -----

circuit Foo :
  module Bar :
    input a : UInt<1>
  module Foo :
    output a : UInt<1>

    inst bar of Bar
    bar.a is invalid
    a <= bar.a

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK-DAG: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK-DAG: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK-DAG: firrtl.subfield %bar("a")
    ; CHECK-DAG: [[SUBFIELD:%.+]] = firrtl.subfield %bar("a")
    ; CHECK-NOT: -------------------- (DAG region separator)
    ; CHECK-DAG: firrtl.connect %a, [[WIRE]]
    ; CHECK-DAG: firrtl.connect [[SUBFIELD]], [[WIRE]]

    ; // -----

circuit Foo :
  module Bar :
    input a : { flip b : UInt<1>, c : UInt<1>}

    skip

  module Foo :
    input a : { flip b : UInt<1>, c : UInt<1>}

    inst bar of Bar
    a <= bar.a

    ; COM: This circuit should be unmodified
    ; CHECK-LABEL: firrtl.circuit
    ; CHECK-NOT: firrtl.wire

    ; // -----

circuit Foo :
  module Bar :
    input a : UInt<1>

  module Foo :
    output a : UInt<1>

    inst bar of Bar

    bar.a is invalid
    a <= not(bar.a)

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK-DAG: [[INVALID:%.+]] = firrtl.invalidvalue
    ; CHECK-DAG: firrtl.connect [[WIRE]], [[INVALID]]
    ; CHECK-DAG: firrtl.subfield %bar("a")
    ; CHECK-DAG: [[SUBFIELD:%.+]] = firrtl.subfield %bar("a")
    ; CHECK: firrtl.connect [[SUBFIELD]], [[WIRE]]
    ; CHECK: [[WIRE_passive:%.+]] = firrtl.asPassive [[WIRE]]
    ; CHECK: [[NOT:%.+]] = firrtl.not [[WIRE_passive]]
    ; CHECK-DAG: firrtl.connect %a, [[NOT]]

    ; // -----

circuit Foo :
  module Bar :
    input i : UInt<1>
    output o : UInt<1>
    o <= i
  module Foo :
    input i : UInt<1>
    output o : UInt<1>
    inst bar of Bar
    bar.i <= i
    node x = and(bar.i, bar.o)
    o <= x

    ; CHECK-LABEL: firrtl.circuit
    ; CHECK: [[WIRE:%.+_synthetic]] = firrtl.wire : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.connect [[WIRE]], %i
    ; CHECK: [[SUBFIELD_i:%.+]] = firrtl.subfield %bar("i")
    ; CHECK: firrtl.connect [[SUBFIELD_i]], [[WIRE]]
    ; CHECK-DAG: [[WIRE_passive:%.+]] = firrtl.asPassive [[WIRE]]
    ; CHECK-DAG: [[SUBFIELD_o:%.+]] = firrtl.subfield %bar("o")
    ; CHECK: [[AND:%.+]] = firrtl.and [[WIRE_passive]], [[SUBFIELD_o]]
    ; CHECK: %x = firrtl.node [[AND]]
