; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

circuit test :
  extmodule MyModule :

    ; expected-error @+1 {{unterminated string}}
    parameter FORMAT = "tilelink_timeou
    : " ; End the string above for syntax highlighters...

;// -----

FIRRTL
; expected-error @+1 {{expected version after 'FIRRTL'}}
circuit test :
  module test :

;// -----

FIRRTL version
; expected-error @+1 {{expected version literal}}
circuit test :
  module test :

;// -----

; expected-error @+1 {{expected version literal}}
FIRRTL version 1
circuit test :
  module test :

;// -----

; expected-error @+1 {{expected version literal}}
FIRRTL version 1.
circuit test :
  module test :

;// -----

; expected-error @+1 {{expected version literal}}
FIRRTL version 1.a
circuit test :
  module test :

;// -----

; expected-error @below {{'circuit' must be first token on its line}}
FIRRTL version 1.1.0 circuit test :
  module test :

;// -----

; expected-error @below {{FIRRTL version must be >=0.2.0}}
FIRRTL version 0.1.0
circuit test :
  module test :

;// -----

circuit test test : ; expected-error {{expected ':' in circuit definition}}

;// -----

; expected-error @+1 {{unexpected character}}
@

;// -----

circuit nameConflict :
  module nameConflict :
    input c: Clock   ; expected-note {{previous definition here}}
    input r: Reset
    input c: Analog  ; expected-error {{redefinition of name 'c'}}

;// -----

circuit nameConflict :
  extmodule nameConflict :
    input c: Clock   ; expected-note {{previous definition here}}
    input r: Reset
    input c: Analog  ; expected-error {{redefinition of name 'c'}}

;// -----

circuit invalid_name :
  module invalid_name :
    input c: UInt
    out <= c         ; expected-error {{use of unknown declaration 'out'}}

;// -----

circuit invalid_name :
  module invalid_name :
    output out: UInt
    out <= c         ; expected-error {{use of unknown declaration 'c'}}

;// -----

circuit subfield_of_ground_type :
  module subfield_of_ground_type :
    input a: UInt<1>
    output b: UInt<1>
    ; expected-error @+1 {{subfield requires bundle or object operand}}
    b <= a.a

;// -----

FIRRTL version 1.2.0
circuit invalid_name :
  module invalid_name :
   input out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}
   ; expected-error @+1 {{unknown field 'xx' in type '!firrtl.bundle<member: bundle<"0": bundle<clock: clock, reset: uint<1>>>>'}}
   out_0.xx <- out_0.yy

;// -----

FIRRTL version 1.2.0
circuit invalid_name :
  module invalid_name :
   input out_0 : SInt<8>[5]
   ; expected-error @+1 {{out of range index '5' in vector type '!firrtl.vector<sint<8>, 5>'}}
   out_0[4] <- out_0[5]

;// -----

circuit invalid_add :
  module invalid_add :
   input in : SInt<8>
   input c : Clock
   ; expected-error @+1 {{second operand must be an integer type, not '!firrtl.clock'}}
   node n = add(in, c)

;// -----

circuit invalid_add :
  module invalid_add :
   input in : SInt<8>
   input c : Clock
   ; expected-error @+1 {{operation requires two operands}}
   node n = add(in, in, in)

;// -----

circuit invalid_int_literal :
  module invalid_int_literal :
   node n = add(UInt<8>("hAX"), UInt<10>(42))  ; expected-error {{invalid character in integer literal}}

;// -----
; When scopes are local to the body
FIRRTL version 1.2.0
circuit invalid_name :
  module invalid_name :
    input reset : UInt<1>
    output out : UInt<1>
    when reset :
      node n4 = reset
    out <- n4   ; expected-error {{use of unknown declaration 'n4'}}

;// -----

circuit invalid_inst :
  module invalid_inst :

    ; expected-error @+1 {{use of undefined module name 'some_module' in instance}}
    inst xyz of some_module

;// -----

FIRRTL version 3.3.0
circuit class_inst :
  class some_class :
  module class_inst :
    ; expected-error @below {{cannot create instance of class 'some_class', did you mean object?}}
    inst xyz of some_class

;// -----

circuit MyParameterizedExtModule :
  extmodule MyParameterizedExtModule :
    parameter DEFAULT = 0
    parameter DEFAULT = 32 ; expected-error {{redefinition of parameter 'DEFAULT'}}

;// -----

circuit invalid_name :
  module invalid_name :
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    node n4 = add(bf, bf)  ; expected-error {{operands must be integer types, not '!firrtl.bundle<int_1 flip: uint<1>, int_out: uint<2>>' and '!firrtl.bundle<int_1 flip: uint<1>, int_out: uint<2>>'}}

;// -----

circuit invalid_bits :
  module invalid_bits:
     input a: UInt<8>
     output b: UInt<4>
     ; expected-error @+1 {{high must be equal or greater than low, but got high = 4, low = 7}}
     b <= bits(a, 4, 7)

;// -----

circuit test :
  module invalid_add :
   input in1 : SInt
   input in2 : UInt
   node n = add(in1, in2)  ; expected-error {{operand signedness must match}}

;// -----

circuit invalid_node_not_passive :
  module invalid_node_not_passive :
    input a : { a: UInt, flip b: UInt}
    output b : { a: UInt, flip b: UInt}
    ; expected-error @+1 {{Node cannot be analog and must be passive or passive under a flip}}
    node n = a

;// -----

circuit invalid_node_analog :
  module invalid_node_analog :
    input a : Analog<1>
    output b : Analog<1>
    ; expected-error @+1 {{Node cannot be analog and must be passive or passive under a flip}}
    node n = a

;// -----

circuit Issue418:
  module Issue418:
    input a: UInt<1>
    output b: UInt<1>

    ; expected-error @+1 {{operation requires one operand}}
    b <= not(a, a)

;// -----

circuit Issue3799:
  module Issue3799:
    output a: UInt<0>
    a <= UInt<0>(0) ; ok
    a <= UInt<0>(-0) ; ok
    ; expected-error @below {{zero bit constant must be zero}}
    a <= UInt<0>(1)

;// -----

circuit Issue3799:
  module Issue3799:
    output a: SInt<0>
    a <= SInt<0>(0) ; ok
    a <= SInt<0>(-0) ; ok
    ; expected-error @below {{zero bit constant must be zero}}
    a <= SInt<0>(1)


;// -----

circuit Issue426:
  module Issue426:
    output a: UInt<1>
    ; expected-error @+1 {{initializer too wide for declared width}}
    a <= UInt<1>(2)

;// -----

circuit Issue426:
  module Issue426:
    output a: SInt<1>
    a <= SInt<1>(0) ; ok
    a <= SInt<1>(-1) ; ok
    ; expected-error @+1 {{initializer too wide for declared width}}
    a <= SInt<1>(1)
;// -----

circuit Issue426:
  module Issue426:
    output a: SInt<1>
    ; expected-error @+1 {{initializer too wide for declared width}}
    a <= SInt<1>(-2)

;// -----

circuit circuit:
  module circuit :
    input in: UInt<80>
    inst xyz of circuit
    node n = xyz
    node m = n     ; expected-error {{expected '.' in field reference}}

;// -----

circuit circuit:
  module circuit :
    input in: UInt<80>
    inst xyz of circuit
    node n = xyz.foo   ; expected-error {{use of invalid field name 'foo' on bundle value}}

;// -----

circuit Issue2971: ; expected-error {{no main module}}
  module NotIssue2971:
    input in: UInt<1>
    output out: UInt<1>
    out <= in

;// -----

circuit NoModules: ; expected-error {{no modules found}}
   ; nothing

;// -----

circuit NonAscii:
  module NonAscii:
    input clk: Clock
    input enable: UInt<1>
    printf(clk, enable, "’") ; expected-error {{string characters must be 7-bit ASCII}}

;// -----

circuit UnicodeEscape:
  module UnicodeEscape:
    input clk: Clock
    input enable: UInt<1>
    printf(clk, enable, "\u0065") ; expected-error {{unicode escape not supported in string}}

;// -----

circuit PrintfExpr:
  module PrintfExpr:
    input clk: Clock
    input enable: UInt<1>
    output out : UInt<1>
    out <= printf(clk, enable, "b1") ; expected-error {{unexpected token in statement}}

;// -----

circuit ProbeFlipType:
  module ProbeFlipType:
    output p : Probe<{a: UInt, flip b: UInt}> ; expected-error {{probe inner type must be passive}}

;// -----

circuit NestedProbes:
  module NestedProbes:
    output p : Probe<Probe<UInt>> ; expected-error {{cannot nest reference types}}

;// -----

circuit RefSubaccess:
  module RefSubaccess:
    input in : UInt<1>[3]
    input sel : UInt<2>
    output out : UInt<1>

    out <= read(probe(in)[sel]) ; expected-error {{expected ')' in 'read'}}

;// -----

circuit ProbeBase:
  extmodule ProbeGen:
    output p : Probe<UInt<2>>
    ref p is "x.y"

  module ProbeBase:
    input in : UInt<1>
    output out : UInt

    inst pg of ProbeGen
    out <= probe(pg.p) ; expected-error {{expected base-type expression in 'probe', got}}

;// -----

circuit ProbeProbe:
  module ProbeProbe:
    input in : UInt<1>
    output out : UInt

    out <= probe(probe(in)) ; expected-error {{expected static reference expression in 'probe'}}

;// -----

circuit ReadBase:
  module ReadBase:
    input in : UInt<1>
    output out : UInt

    out <= read(in) ; expected-error {{expected reference-type expression in 'read', got}}

;// -----

circuit DefineBaseLHS:
  module DefineBaseLHS:
    input in : UInt<1>
    output out : UInt

    define out = probe(in) ; expected-error {{expected reference-type expression in 'define' target (LHS), got}}

;// -----

circuit DefineBaseRHS:
  module DefineBaseRHS:
    input in : UInt<1>
    output out : Probe<UInt>

    define out = in ; expected-error {{expected reference-type expression in 'define' source (RHS), got}}

;// -----
circuit RefSubaccess:
  module RefSubaccess:
    input in : UInt<1>[3]
    input sel : UInt<2>
    output out : UInt<1>

    out <= read(probe(in[sel])) ; expected-error {{subaccess not allowed here}}

;// -----

circuit DefineTwice:
  module DefineTwice:
    input in: UInt<1>
    output out : Probe<UInt<1>>
    define out = probe(in) ; expected-error {{destination reference cannot be reused by multiple operations, it can only capture a unique dataflow}}
    define out = probe(in)

;// -----

circuit DefineIntoSubindex:
  module DefineIntoSubindex:
    input in : UInt<1>[3]
    output out : Probe<UInt<1>[3]>

    define out[0] = probe(in) ; expected-error {{cannot define into a sub-element of a reference}}

;// -----

circuit DefineIntoProbe:
  module DefineIntoProbe:
    input in : UInt<1>
    output out : UInt<1>

    define probe(out) = probe(in) ; expected-error {{expected static reference expression in 'define'}}

;// -----

circuit InProbeTop:
  module InProbeTop:
    input p : Probe<UInt<1>> ; expected-error {{input probe not allowed on public module}}

;// -----

circuit InProbeExt:
  extmodule BadExtMod:
    input p : Probe<UInt<1>> ; expected-error {{references in ports must be output on extmodule and intmodule}}

  module InProbeExt:
    inst e of BadExtMod

;// -----

circuit RefDupe:
  extmodule RefDupe:
    output p : Probe<UInt<2>>
    ref p is "x.y"
    ref p is "a.z" ; expected-error {{duplicate ref statement for 'p'}}

;// -----
; Either use ref statements or ABI, can't partially use ref statements.

circuit MissingRef:
  extmodule MissingRef:
    output p : Probe<UInt<2>> ; expected-error {{no ref statement found for ref port "p"}}
    output q : Probe<UInt<2>>
    ref q is "foo.bar"

;// -----

circuit UnusedRef:
  extmodule UnusedRef:
    ref p is "x.y" ; expected-error {{unused ref statement}}

;// -----

circuit RefNotString:
  extmodule RefNotString:
    output p : Probe<UInt<2>>
    ref p is a.b ; expected-error {{expected string in ref statement}}

;// -----

circuit RefForAggElement:
  extmodule RefForAggElement:
    output p : { x: Probe<UInt<2>> }
    ref p.x is a.b ; expected-error {{ref statements for aggregate elements not yet supported}}

;// -----

circuit ConnectWrongType:
  module ConnectWrongType:
    input a: UInt<1>
    output b: SInt<1>

    b <= a ; expected-error {{cannot connect non-equivalent type}}

;// -----

FIRRTL version 3.0.0
circuit ConnectAlternateWrongType:
  module ConnectAlternateWrongType:
    input a: UInt<1>
    output b: SInt<1>

    connect b, a ; expected-error {{cannot connect non-equivalent type}}

;// -----

circuit ProbeConnect:
  module ProbeConnect:
    input in : UInt<1>
    output p : Probe<UInt<1>>

    p <= probe(in) ; expected-error {{cannot connect reference or property types}}

;// -----

FIRRTL version 3.0.0
circuit ProbeConnectAlternate:
  module ProbeConnectAlternate:
    input in : UInt<1>
    output p : Probe<UInt<1>>

    connect p, probe(in) ; expected-error {{cannot connect reference or property types}}

;// -----

FIRRTL version 1.2.0
circuit ProbePartialConnect:
  module ProbePartialConnect:
    input in : UInt<1>
    output p : Probe<UInt<1>>

    p <- probe(in) ; expected-error {{cannot connect reference or property types}}

;// -----

circuit ProbeMem:
  module ProbeMem:
    output out : Probe<UInt<3>> ; (wrong probe type, can't probe memories)

    mem memory:
      data-type => UInt<8>
      depth => 4
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    memory.r is invalid
    memory.w is invalid
    define out = probe(memory.r) ; expected-error {{cannot probe memories or their ports}}


;// -----

circuit ProbeSMem:
  module ProbeSMem:
    input clock : Clock
    input reset : Reset
    output io : { flip addr : UInt<3>, dataOut : UInt<8>}
    output out : Probe<UInt<8>>

    smem mem : UInt<8> [8]
    infer mport read = mem[io.addr], clock
    io.dataOut <= read
    define out = probe(read) ; expected-error {{cannot probe memories or their ports}}


;// -----

circuit LeadingRead:
  module LeadingRead:
    input in : UInt<1>
    read(probe(in)) ; expected-error {{unexpected read() as start of statement}}

;// -----

circuit LeadingProbe:
  module LeadingProbe:
    input in : UInt<1>
    probe(in) ; expected-error {{unexpected probe() as start of statement}}

;// -----

circuit ForceProbe:
  module ForceProbe:
    input in : UInt<1>
    force_initial(probe(in), UInt<1>(1)) ; expected-error {{expected rwprobe-type expression for force_initial destination, got '!firrtl.probe<uint<1>>'}}

;// -----

circuit ForceWithLiteral:
  module ForceWithLiteral:
    input in : UInt<1>
    node n = in
    force_initial(rwprobe(n), 1) ; expected-error {{expected source expression in force_initial}}

;// -----

circuit RWProbeConst:
  extmodule RWProbeConst:
    output p : RWProbe<{a: const UInt}> ; expected-error {{rwprobe cannot contain const}}

;// -----

circuit RWProbeConstPort:
  module RWProbeConstPort:
    input in : const UInt<1>[2]
    output p : RWProbe<UInt<1>>
    define p = rwprobe(in[1]) ; expected-error {{cannot force target of type '!firrtl.const.uint<1>'}}

;// -----

FIRRTL version 2.0.0
circuit PartialConnect_v2p0p0:
  module PartialConnect_v2p0p0:
    input a: UInt<1>
    output b: UInt<1>

    b <- a ; expected-error {{unexpected token '<-' in statement}}

;// -----

FIRRTL version 2.0.0
circuit RegReset_v2p0p0:
  module RegReset_v2p0p0:
    input clock: Clock

    ; expected-error @below {{use of unknown declaration 'regreset'}}
    regreset a: UInt<1>, clock, UInt<1>(0), UInt<1>(0)

;// -----
FIRRTL version 3.0.0
circuit IllegalLiteral:
  module IllegalLiteral:
    input `b: UInt<1> ; expected-error {{unterminated literal identifier}}

;// -----

circuit NodeProbe:
  extmodule Ref:
    output x : Probe<UInt<1>>
  module NodeProbe:
    inst r of Ref
    node n = r.x ; expected-error {{Node cannot be analog and must be passive or passive under a flip}}

;// -----

circuit ConstProbe:
  module ConstProbe:
    output x : const Probe<UInt<1>> ; expected-error {{only hardware types can be 'const'}}

;// -----

circuit MultiConst:
  module MultiConst:
    output x : const const UInt<1> ; expected-error {{'const' can only be specified once on a type}}

;// -----

circuit DefinePromoteToRW:
  extmodule Ref:
    output ro : Probe<UInt<1>>
  module DefinePromoteToRW:
    output rw : RWProbe<UInt<1>>
    inst r of Ref
    ; expected-error @below {{cannot define reference of type '!firrtl.rwprobe<uint<1>>' with incompatible reference of type '!firrtl.probe<uint<1>>'}}
    define rw = r.ro

;// -----
FIRRTL version 3.1.0

circuit PropAssignNonProperty:
  module PropAssignNonProperty:
    input x : UInt<1>
    output y : UInt<1>
    propassign x, y ; expected-error {{can only propassign property types}}

;// -----

FIRRTL version 2.3.9
circuit UnsupportedRadixSpecifiedIntegerLiterals:
  module UnsupportedRadixSpecifiedIntegerLiterals:
    output foo: UInt<8>
    ; expected-error @below {{radix-specified integer literals are a FIRRTL 2.4.0+ feature, but the specified FIRRTL version was 2.3.9}}
    foo <= UInt(0b101010)

;// -----

FIRRTL version 2.4.0
circuit InvalidBinaryRadixSpecifiedIntegerLiteral:
  module InvalidBinaryRadixSpecifiedIntegerLiteral:
    output foo: UInt<8>
    ; expected-error @below {{invalid character in integer literal}}
    foo <= UInt(0b2)

;// -----

FIRRTL version 2.4.0
circuit InvalidOctalRadixSpecifiedIntegerLiteral:
  module InvalidOctalRadixSpecifiedIntegerLiteral:
    output foo: UInt<8>
    ; expected-error @below {{invalid character in integer literal}}
    foo <= UInt(0o8)

;// -----

FIRRTL version 2.4.0
circuit InvalidDecimalRadixSpecifiedIntegerLiteral:
  module InvalidDecimalRadixSpecifiedIntegerLiteral:
    output foo: UInt<8>
    ; expected-error @below {{invalid character in integer literal}}
    foo <= UInt(0da)

;// -----

FIRRTL version 2.4.0
circuit InvalidHexadecimalRadixSpecifiedIntegerLiteral:
  module InvalidHexadecimalRadixSpecifiedIntegerLiteral:
    output foo: UInt<8>
    ; expected-error @below {{invalid character in integer literal}}
    foo <= UInt(0hg)

;// -----

FIRRTL version 3.0.0
circuit UnsupportedStringEncodedIntegerLiterals:
  module UnsupportedStringEncodedIntegerLiterals:
    output foo: UInt<8>
    ; expected-error @below {{String-encoded integer literals are unsupported after FIRRTL 3.0.0}}
    connect foo, UInt("h2a")

;// -----

circuit UnknownTypeAlias:
  module UnknownTypeAlias:
    output x : Foo ; expected-error {{type identifier `Foo` is not declared}}

;// -----

circuit DuplicateAlias:
  type Foo = UInt
  type Foo = UInt; expected-error {{type alias `Foo` is already defined}}
  module DuplicateAlias:

;// -----

circuit KeywordTypeName:
  type Clock = UInt; expected-error {{cannot use keyword 'Clock' for type alias name}}

;// -----

circuit KeywordTypeName:
  type module = UInt; expected-error {{cannot use keyword 'module' for type alias name}}

;// -----

FIRRTL version 3.1.0
circuit UnsupportedVersionDeclGroups:
  ; expected-error @below {{optional groups are a FIRRTL 3.2.0+ feature, but the specified FIRRTL version was 3.1.0}}
  declgroup A, bind:

;// -----

FIRRTL version 3.1.0
circuit UnsupportedVersionGroups:
  module UnsupportedVersionGroups:
    ; expected-error @below {{optional groups are a FIRRTL 3.2.0+ feature, but the specified FIRRTL version was 3.1.0}}
    group A:

;// -----

FIRRTL version 3.2.0
circuit UnsupportedGroupConvention:
  ; expected-error @below {{unknown convention 'foo'}}
  declgroup A, foo :

;// -----

FIRRTL version 3.2.0
circuit GroupAndCircuitShareName:
  ; expected-error @below {{cannot have the same name as the circuit}}
  declgroup GroupAndCircuitShareName, bind :

  module GroupAndCircuitShareName:


;// -----

FIRRTL version 3.3.0
circuit Top:
  ; expected-error @below {{class cannot be the top of a circuit}}
  class Top:

;// -----

FIRRTL version 3.3.0
circuit Top:
  ; expected-error @below {{extclass cannot be the top of a circuit}}
  extclass Top:

;// -----

FIRRTL version 3.3.0
circuit Top:
  module Top:
  class Foo:
    ;; expected-error @below {{ports on classes must be properties}}
    input a: UInt<8>

;// -----

FIRRTL version 3.3.0
circuit Top:
  module Top:
  extclass Foo:
    ;; expected-error @below {{ports on extclasses must be properties}}
    input a: UInt<8>

;// -----
FIRRTL version 3.0.0

circuit Top:
  module Top:
    output s : String
    ; expected-error @above {{Strings are a FIRRTL 3.1.0+ feature, but the specified FIRRTL version was 3.0.0}}

;// -----
FIRRTL version 3.0.0

circuit Top:
  module Top:
    ; expected-error @below {{Integers are a FIRRTL 3.1.0+ feature, but the specified FIRRTL version was 3.0.0}}
    input in : Integer
    output out : Integer
    propassign out, in

;// -----
FIRRTL version 3.3.0

circuit Top:
  module Top:
    ; expected-error @below {{unknown class 'Missing'}}
    input in : Inst<Missing>

;// -----
FIRRTL version 3.3.0

circuit Top:
  module Top:
    ; expected-error @below {{use of undefined class name 'Missing' in object}}
    object x of Missing

;// -----
FIRRTL version 3.3.0

circuit Top:
  class MyClass:
    skip

  module Top:
    output str : String
    object x of MyClass
    ; expected-error @below {{unknown field 'missing' in type '!firrtl.class<@MyClass()>'}}
    propassign str, x.missing

;// -----
FIRRTL version 3.0.0

circuit Top:
  ; expected-error @below {{classes are a FIRRTL 3.3.0+ feature, but the specified FIRRTL version was 3.0.0}}
  class MyClass:
    skip

  module Top:
    object x of MyClass

;// -----
FIRRTL version 3.0.0

circuit Top:
  module Top:
    ; expected-error @below {{object statements are a FIRRTL 3.3.0+ feature, but the specified FIRRTL version was 3.0.0}}
    object x of MyClass

;// -----
FIRRTL version 3.0.0

circuit Top:
  module Top:
    ; expected-error @below {{Inst types are a FIRRTL 3.3.0+ feature, but the specified FIRRTL version was 3.0.0}}
    output o: Inst<MyClass>

;// -----
; Lists not yet supported.
FIRRTL version 3.1.0
circuit ListOfInt:
   module ListOfInt:
     ; expected-error @below {{Lists are a FIRRTL 3.3.0+ feature}}
     output a : List<Integer>

;// -----
FIRRTL version 3.3.0
circuit ListOfUInt:
   module ListOfUInt:
     ; expected-error @below {{expected property type}}
     output a : List<UInt>

;// -----
; Wrong type of elements in List expression.
FIRRTL version 3.3.0
circuit ListOfUInt:
   module ListOfUInt:
     output a : List<String>
     ; expected-error @below {{unexpected expression of type '!firrtl.integer' in List expression of type '!firrtl.string'}}
     propassign a, List<String>(Integer(5))

;// -----
; List should not be leading a statement.
FIRRTL version 3.3.0
circuit LeadingList:
  module LeadingList:
    ; expected-error @below {{unexpected List<>() as start of statement}}
    List<String>()

;// -----
; Path should not be leading a statement.
FIRRTL version 3.3.0
circuit LeadingPath:
  module LeadingPath:
    ; expected-error @below {{unexpected path() as start of statement}}
    path("")

;// -----
FIRRTL version 3.1.0
circuit PathVersion:
  module PathVersion:
    ; expected-error @below {{Paths are a FIRRTL 3.3.0+ feature, but the specified FIRRTL version was 3.1.0}}
    output path : Path
    propassign path, path("")

;// -----
; Path operation should have a single string argument.
FIRRTL version 3.3.0
circuit BadPathExpr:
  module BadPathExp:
    output path : Path
    ; expected-error @below {{expected target string in path expression}}
    propassign path, path()

;// -----
; Path operation should have a single string argument.
FIRRTL version 3.3.0
circuit BadPathExpr:
  module BadPathExp:
    output path : Path
    ; expected-error @below {{expected ')' in path expression}}
    propassign path, path("hello", "goodbye")

;// -----
; Path operation should have a single string argument.
FIRRTL version 3.3.0
circuit BadPathExpr:
  module BadPathExp:
    output path : Path
    ; expected-error @below {{expected target string in path expression}}
    propassign path, path(UInt<1>(1))

;// -----
; CHeck version handling for Bool.
FIRRTL version 3.0.0

circuit Top:
  module Top:
    ; expected-error @below {{Bools are a FIRRTL 3.3.0+ feature, but the specified FIRRTL version was 3.0.0}}
    input in : Bool
    output out : Bool
    propassign out, in

;// -----
; Bool literal must be true or false.
FIRRTL version 3.3.0

circuit Top:
  module Top:
    output out : Bool
    ; expected-error @below {{expected true or false in Bool expression}}
    propassign out, Bool(0)

;// -----
; Properties can't be const.
FIRRTL version 3.3.0

circuit Top:
  module Top:
    ; expected-error @below {{only hardware types can be 'const'}}
    output out : const Bool

;// -----
; AnyRef not yet supported.
FIRRTL version 3.1.0
circuit AnyRef:
   module AnyRef:
     ; expected-error @below {{AnyRef types are a FIRRTL 3.3.0+ feature}}
     output a : AnyRef

;// -----
; Only objects are valid as AnyRef
FIRRTL version 3.3.0
circuit AnyRef:
  module AnyRef:
    output out : AnyRef
    ; expected-error @below {{cannot propassign non-equivalent type '!firrtl.integer' to '!firrtl.anyref'}}
    propassign out, Integer(5)

;// -----
; Only objects are valid as AnyRef in Lists
FIRRTL version 3.3.0
circuit AnyRefList:
  module AnyRefList:
    output list : List<AnyRef>
    ; expected-error @below {{unexpected expression of type '!firrtl.integer' in List expression of type '!firrtl.anyref'}}
    propassign list, List<AnyRef>(Integer(5))

;// -----
; Not Covariant.
FIRRTL version 3.3.0
circuit NotCovariant:
  class Class:

  module NotCovariant:
    output list : List<AnyRef>

    object obj of Class
    ; expected-error @below {{cannot propassign non-equivalent type '!firrtl.list<class<@Class()>>' to '!firrtl.list<anyref>'}}
    propassign list, List<Inst<Class>>(obj)

;// -----
; Not Contravariant.
FIRRTL version 3.3.0
circuit NotContravariant:
  class Class:

  module NotContravariant:
    output list : List<Inst<Class>>

    object obj of Class
    ; expected-error @below {{cannot propassign non-equivalent type '!firrtl.list<anyref>' to '!firrtl.list<class<@Class()>>'}}
    propassign list, List<AnyRef>(obj)

;// -----
; Double: must have digit before point.
FIRRTL version 3.3.0
circuit DoubleNegPeriod:
  module DoubleNegPeriod:
    output d : Double
    ; expected-error @below {{unexpected character after sign}}
    propassign d, Double(-.5)

;// -----
; Double: must have digit after point.
FIRRTL version 3.3.0
circuit DoublePeriodEnd:
  module DoublePeriodEnd:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(0.)

;// -----
; Double: Not an integer.
FIRRTL version 3.3.0
circuit DoubleInteger:
  module DoubleInteger:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(0)

;// -----
; Double: don't support NaN or inf.
FIRRTL version 3.3.0
circuit DoubleNaN:
  module DoubleNaN:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(NaN)

;// -----
; Double: Don't support hex.
FIRRTL version 3.3.0
circuit DoubleHex:
  module DoubleHex:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(0x0)

;// -----
; Double: Don't suuport FIRRTL-y radix.
FIRRTL version 3.3.0
circuit DoubleRadix:
  module DoubleRadix:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(0hABC)
