// RUN: circt-opt %s -export-verilog -o %t.mlir | FileCheck %s
// RUN: circt-opt %t.mlir -export-verilog -o /dev/null | FileCheck %s

// Anonymous enumerations keep their compilation-unit typedef and stay
// unqualified: declaring a package elsewhere does not move them.
// CHECK:      `ifndef _TYPESCOPE_Enums
// CHECK:        typedef enum bit [0:0] {enum0_A, enum0_B} enum0;
// CHECK:      `endif // _TYPESCOPE_Enums

// A package has to be declared before its users, so it is emitted first even
// though it is declared last in this file.
// CHECK-LABEL: package types;
// CHECK-NEXT:    typedef logic [31:0] word;
// CHECK-NEXT:    typedef word wordAlias;
// CHECK-NEXT:    typedef struct packed {word data; logic valid; } packet;
// CHECK-NEXT:    typedef word memory[0:3];
// CHECK-NEXT:    typedef enum bit [0:0] {state_Idle, state_Busy} state;
// CHECK-NEXT:    typedef state stateAlias;
// CHECK-NEXT:  endpackage

// A second package may declare the same type name and refer to the first.
// CHECK-LABEL: package other;
// CHECK-NEXT:    typedef logic [7:0] word;
// CHECK-NEXT:    typedef types::word imported;
// CHECK-NEXT:  endpackage

// CHECK-LABEL: package empty;
// CHECK-NEXT:  endpackage

// CHECK-LABEL: module Consumer(
// CHECK-NEXT:    input  types::word      word,
// CHECK-NEXT:    input  types::wordAlias wordAlias,
// CHECK-NEXT:    input  types::word[3:0] words,
// CHECK-NEXT:    input  types::memory    memory,
// CHECK-NEXT:    input  types::packet    packet,
// CHECK-NEXT:    input  other::word      byte_0,
// CHECK-NEXT:    output types::word      sum,
// CHECK-NEXT:    output types::state     state
hw.module @Consumer(
    in %word: !hw.typealias<@types::@word, i32>,
    in %wordAlias: !hw.typealias<@types::@wordAlias, !hw.typealias<@types::@word, i32>>,
    in %words: !hw.array<4xtypealias<@types::@word, i32>>,
    in %memory: !hw.typealias<@types::@memory, !hw.uarray<4xtypealias<@types::@word, i32>>>,
    in %packet: !hw.typealias<@types::@packet, !hw.struct<data: !hw.typealias<@types::@word, i32>, valid: i1>>,
    in %byte: !hw.typealias<@other::@word, i8>,
    out sum: !hw.typealias<@types::@word, i32>,
    out state: !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>) {
  // CHECK:      wire types::word {{.*}}localWord;
  // CHECK-NEXT: types::packet {{.*}}localPacket;
  %localWord = sv.wire : !hw.inout<!hw.typealias<@types::@word, i32>>
  %localPacket = sv.reg : !hw.inout<!hw.typealias<@types::@packet, !hw.struct<data: !hw.typealias<@types::@word, i32>, valid: i1>>>

  // CHECK: assign sum = word + word;
  %sum = comb.add %word, %word : !hw.typealias<@types::@word, i32>
  // CHECK: assign state = types::state_Idle;
  %idle = hw.enum.constant Idle : !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>
  hw.output %sum, %idle : !hw.typealias<@types::@word, i32>, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>
}

// An interface signal is another user of a package type.
// CHECK-LABEL: interface Signals;
// CHECK-NEXT:    types::word value;
sv.interface @Signals {
  sv.interface.signal @value : !hw.typealias<@types::@word, i32>
}

// Enumeration members are qualified in case items and in constants, and an
// alias of an aliased enumeration resolves to the declaration owning them.
// CHECK-LABEL: module StateMachine(
hw.module @StateMachine(in %clock: i1) {
  // CHECK: types::stateAlias named;
  %named = sv.reg : !hw.inout<!hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>>
  %namedValue = sv.read_inout %named : !hw.inout<!hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>>
  %idle = hw.enum.constant Idle : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>
  %busy = hw.enum.constant Busy : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>
  // CHECK:      case (named)
  // CHECK-NEXT:   types::state_Idle:
  // CHECK-NEXT:     named <= types::state_Busy;
  // CHECK-NEXT:   default:
  // CHECK-NEXT:     named <= types::state_Idle;
  sv.always posedge %clock {
    sv.case case %namedValue : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>
      case Idle: { sv.passign %named, %busy : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>> }
      default: { sv.passign %named, %idle : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>> }
  }
}

// CHECK-LABEL: module Anonymous(
// CHECK-NEXT:    output enum0 out
// CHECK:         assign out = enum0_A;
hw.module @Anonymous(out out: !hw.enum<A, B>) {
  %a = hw.enum.constant A : !hw.enum<A, B>
  hw.output %a : !hw.enum<A, B>
}

sv.package @types {
  hw.typedecl @word : i32
  hw.typedecl @wordAlias : !hw.typealias<@types::@word, i32>
  hw.typedecl @packet : !hw.struct<data: !hw.typealias<@types::@word, i32>, valid: i1>
  hw.typedecl @memory : !hw.uarray<4xtypealias<@types::@word, i32>>
  hw.typedecl @state : !hw.enum<Idle, Busy>
  hw.typedecl @stateAlias : !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>
}

sv.package @other {
  hw.typedecl @word : i8
  hw.typedecl @imported : !hw.typealias<@types::@word, i32>
}

sv.package @empty {}
