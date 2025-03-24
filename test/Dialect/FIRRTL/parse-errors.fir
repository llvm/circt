; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

FIRRTL version 4.0.0
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
FIRRTL version 2.0.0 circuit test :
  module test :

;// -----

; expected-error @below {{FIRRTL version must be >=2.0.0}}
FIRRTL version 0.1.0
circuit test :
  module test :

;// -----

FIRRTL version 4.0.0
circuit test test : ; expected-error {{expected ':' in circuit definition}}

;// -----

; expected-error @+1 {{unexpected character}}
@

;// -----

FIRRTL version 4.0.0
circuit nameConflict :
  module nameConflict :
    input c: Clock   ; expected-note {{previous definition here}}
    input r: Reset
    input c: Analog  ; expected-error {{redefinition of name 'c'}}

;// -----

FIRRTL version 4.0.0
circuit nameConflict :
  extmodule nameConflict :
    input c: Clock   ; expected-note {{previous definition here}}
    input r: Reset
    input c: Analog  ; expected-error {{redefinition of name 'c'}}

;// -----

FIRRTL version 4.0.0
circuit invalid_name :
  public module invalid_name :
    input c: UInt<4>
    connect out, c   ; expected-error {{use of unknown declaration 'out'}}

;// -----

FIRRTL version 4.0.0
circuit invalid_name :
  public module invalid_name :
    output out: UInt<4>
    connect out, c   ; expected-error {{use of unknown declaration 'c'}}

;// -----

FIRRTL version 4.0.0
circuit subfield_of_ground_type :
  public module subfield_of_ground_type :
    input a: UInt<1>
    output b: UInt<1>
    ; expected-error @+1 {{subfield requires bundle or object operand}}
    connect b, a.a

;// -----

FIRRTL version 2.0.0
circuit invalid_name :
  module invalid_name :
   input out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}
   ; expected-error @+1 {{unknown field 'xx' in type '!firrtl.bundle<member: bundle<"0": bundle<clock: clock, reset: uint<1>>>>}}
   out_0.xx <= out_0.yy

;// -----

FIRRTL version 2.0.0
circuit invalid_name :
  module invalid_name :
   input out_0 : SInt<8>[5]
   ; expected-error @+1 {{out of range index '5' in vector type '!firrtl.vector<sint<8>, 5>'}}
   out_0[4] <= out_0[5]

;// -----

FIRRTL version 4.0.0
circuit invalid_add :
  public module invalid_add :
   input in : SInt<8>
   input c : Clock
   ; expected-error @+1 {{second operand must be an integer type, not '!firrtl.clock'}}
   node n = add(in, c)
;// -----

FIRRTL version 4.0.0
circuit trailing_comma :
  public module trailing_comma :
   input in0 : SInt<8>
   input in1 : SInt<8>
   ; expected-error @+1 {{expected ')'}}
   node n = add(in0, in1,)

;// -----

FIRRTL version 4.0.0
circuit no_commas :
  public module no_commas :
   input in0 : SInt<8>
   input in1 : SInt<8>
   ; expected-error @+1 {{expected ','}}
   node n = add(in0 in1)

;// -----

FIRRTL version 4.0.0
circuit invalid_add :
  public module invalid_add :
   input in : SInt<8>
   input c : Clock
   ; expected-error @+1 {{expected ')'}}
   node n = add(in, in, in)

;// -----

FIRRTL version 2.0.0
circuit invalid_int_literal :
  module invalid_int_literal :
   node n = add(UInt<8>("hAX"), UInt<10>(42))  ; expected-error {{invalid character in integer literal}}

;// -----
; When scopes are local to the body
FIRRTL version 2.0.0
circuit invalid_name :
  module invalid_name :
    input reset : UInt<1>
    output out : UInt<1>
    when reset :
      node n4 = reset
    out <= n4   ; expected-error {{use of unknown declaration 'n4'}}

;// -----

FIRRTL version 4.0.0
circuit invalid_inst :
  public module invalid_inst :

    ; expected-error @+1 {{use of undefined module name 'some_module' in instance}}
    inst xyz of some_module

;// -----

FIRRTL version 4.2.0
circuit class_inst :
  class some_class :
  public module class_inst :
    ; expected-error @below {{cannot create instance of class 'some_class', did you mean object?}}
    inst xyz of some_class

;// -----

FIRRTL version 4.0.0
circuit MyParameterizedExtModule :
  extmodule MyParameterizedExtModule :
    parameter DEFAULT = 0
    parameter DEFAULT = 32 ; expected-error {{redefinition of parameter 'DEFAULT'}}

;// -----

FIRRTL version 4.0.0
circuit invalid_name :
  public module invalid_name :
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    node n4 = add(bf, bf)  ; expected-error {{operands must be integer types, not '!firrtl.bundle<int_1 flip: uint<1>, int_out: uint<2>>' and '!firrtl.bundle<int_1 flip: uint<1>, int_out: uint<2>>'}}

;// -----

FIRRTL version 4.0.0
circuit invalid_bits :
  public module invalid_bits:
     input a: UInt<8>
     output b: UInt<4>
     ; expected-error @+1 {{high must be equal or greater than low, but got high = 4, low = 7}}
     connect b, bits(a, 4, 7)

;// -----

FIRRTL version 4.0.0
circuit test :
  public module invalid_add :
   input in1 : SInt<5>
   input in2 : UInt<5>
   node n = add(in1, in2)  ; expected-error {{operand signedness must match}}

;// -----

FIRRTL version 4.0.0
circuit invalid_node_not_passive :
  public module invalid_node_not_passive :
    input a : { a: UInt<5>, flip b: UInt<5>}
    output b : { a: UInt<5>, flip b: UInt<5>}
    ; expected-error @+1 {{Node cannot be analog and must be passive or passive under a flip}}
    node n = a

;// -----

FIRRTL version 4.0.0
circuit invalid_node_analog :
  public module invalid_node_analog :
    input a : Analog<1>
    output b : Analog<1>
    ; expected-error @+1 {{Node cannot be analog and must be passive or passive under a flip}}
    node n = a

;// -----

FIRRTL version 4.0.0
circuit Issue418:
  public module Issue418:
    input a: UInt<1>
    output b: UInt<1>

    ; expected-error @+1 {{expected ')'}}
    connect b, not(a, a)

;// -----

FIRRTL version 4.0.0
circuit Issue3799:
  public module Issue3799:
    output a: UInt<0>
    connect a, UInt<0>(0) ; ok
    connect a, UInt<0>(-0) ; ok
    ; expected-error @below {{zero bit constant must be zero}}
    connect a, UInt<0>(1)

;// -----

FIRRTL version 4.0.0
circuit Issue3799:
  public module Issue3799:
    output a: SInt<0>
    connect a, SInt<0>(0) ; ok
    connect a, SInt<0>(-0) ; ok
    ; expected-error @below {{zero bit constant must be zero}}
    connect a, SInt<0>(1)


;// -----

FIRRTL version 4.0.0
circuit Issue426:
  public module Issue426:
    output a: UInt<1>
    ; expected-error @+1 {{initializer too wide for declared width}}
    connect a, UInt<1>(2)

;// -----

FIRRTL version 4.0.0
circuit Issue426:
  public module Issue426:
    output a: SInt<1>
    connect a, SInt<1>(0) ; ok
    connect a, SInt<1>(-1) ; ok
    ; expected-error @+1 {{initializer too wide for declared width}}
    connect a, SInt<1>(1)
;// -----

FIRRTL version 4.0.0
circuit Issue426:
  public module Issue426:
    output a: SInt<1>
    ; expected-error @+1 {{initializer too wide for declared width}}
    connect a, SInt<1>(-2)

;// -----

FIRRTL version 4.0.0
circuit circuit:
  public module circuit :
    input in: UInt<80>
    inst xyz of circuit
    node n = xyz
    node m = n     ; expected-error {{expected '.' in field reference}}

;// -----

FIRRTL version 4.0.0
circuit circuit:
  public module circuit :
    input in: UInt<80>
    inst xyz of circuit
    node n = xyz.foo   ; expected-error {{use of invalid field name 'foo' on bundle value}}

;// -----

FIRRTL version 4.0.0
circuit NonAscii:
  public module NonAscii:
    input clk: Clock
    input enable: UInt<1>
    printf(clk, enable, "’") ; expected-error {{string characters must be 7-bit ASCII}}

;// -----

FIRRTL version 4.0.0
circuit UnicodeEscape:
  public module UnicodeEscape:
    input clk: Clock
    input enable: UInt<1>
    printf(clk, enable, "\u0065") ; expected-error {{unicode escape not supported in string}}

;// -----

FIRRTL version 2.0.0
circuit PrintfExpr:
  module PrintfExpr:
    input clk: Clock
    input enable: UInt<1>
    output out : UInt<1>
    out <= printf(clk, enable, "b1") ; expected-error {{unexpected token in statement}}

;// -----

FIRRTL version 4.0.0
circuit ProbeFlipType:
  public module ProbeFlipType:
    output p : Probe<{a: UInt, flip b: UInt}> ; expected-error {{probe inner type must be passive}}

;// -----

FIRRTL version 4.0.0
circuit NestedProbes:
  public module NestedProbes:
    output p : Probe<Probe<UInt>> ; expected-error {{invalid probe inner type, must be base-type}}

;// -----

FIRRTL version 4.0.0
circuit BadCommas:
  public module BadCommas:
    output p : Probe<UInt<1>,> ; expected-error {{expected layer name}}
    wire w : UInt<1>
    invalidate w
    define p = probe(w)

;// -----

FIRRTL version 4.0.0
circuit BadCommas2:
  public module BadCommas2:
    output p : Probe<UInt<1>, A,> ; expected-error {{expected '>' to end reference type}}
    wire w : UInt<1>
    invalidate w
    define p = probe(w)

;// -----

FIRRTL version 3.1.0

circuit ProbeOfProp:
  module ProbeOfProp:
    output p : Probe<Integer> ; expected-error {{invalid probe inner type, must be base-type}}

;// -----

FIRRTL version 4.0.0
circuit RefSubaccess:
  public module RefSubaccess:
    input in : UInt<1>[3]
    input sel : UInt<2>
    output out : UInt<1>

    connect out, read(probe(in)[sel]) ; expected-error {{expected ')' in 'read'}}

;// -----

FIRRTL version 4.0.0
circuit ProbeBase:
  extmodule ProbeGen:
    output p : Probe<UInt<2>>

  public module ProbeBase:
    input in : UInt<1>
    output out : UInt<1>

    inst pg of ProbeGen
    connect out, probe(pg.p) ; expected-error {{expected base-type expression in 'probe', got}}

;// -----

FIRRTL version 4.0.0
circuit ProbeProbe:
  public module ProbeProbe:
    input in : UInt<1>
    output out : UInt<1>

    connect out, probe(probe(in)) ; expected-error {{expected static reference expression in 'probe'}}

;// -----

FIRRTL version 4.0.0
circuit ReadBase:
  public module ReadBase:
    input in : UInt<1>
    output out : UInt<1>

    connect out, read(in) ; expected-error {{expected reference-type expression in 'read', got}}

;// -----

FIRRTL version 4.0.0
circuit DefineBaseLHS:
  public module DefineBaseLHS:
    input in : UInt<1>
    output out : UInt<1>

    define out = probe(in) ; expected-error {{expected reference-type expression in 'define' target (LHS), got}}

;// -----

FIRRTL version 4.0.0
circuit DefineBaseRHS:
  public module DefineBaseRHS:
    input in : UInt<1>
    output out : Probe<UInt<1>>

    define out = in ; expected-error {{expected reference-type expression in 'define' source (RHS), got}}

;// -----

FIRRTL version 4.0.0
circuit RefSubaccess:
  public module RefSubaccess:
    input in : UInt<1>[3]
    input sel : UInt<2>
    output out : UInt<1>

    connect out, read(probe(in[sel])) ; expected-error {{subaccess not allowed here}}

;// -----

FIRRTL version 4.0.0
circuit DefineTwice:
  public module DefineTwice:
    input in: UInt<1>
    output out : Probe<UInt<1>>
    define out = probe(in) ; expected-error {{destination reference cannot be reused by multiple operations, it can only capture a unique dataflow}}
    define out = probe(in)

;// -----

FIRRTL version 4.0.0
circuit DefineIntoSubindex:
  public module DefineIntoSubindex:
    input in : UInt<1>[3]
    output out : Probe<UInt<1>[3]>

    define out[0] = probe(in) ; expected-error {{cannot define into a sub-element of a reference}}

;// -----

FIRRTL version 4.0.0
circuit DefineIntoProbe:
  public module DefineIntoProbe:
    input in : UInt<1>
    output out : UInt<1>

    define probe(out) = probe(in) ; expected-error {{expected static reference expression in 'define'}}

;// -----

FIRRTL version 4.0.0
circuit InProbeTop:
  public module InProbeTop:
    input p : Probe<UInt<1>> ; expected-error {{input probe not allowed}}

;// -----

FIRRTL version 4.0.0
circuit InProbeExt:
  extmodule BadExtMod:
    input p : Probe<UInt<1>> ; expected-error {{references in ports must be output on extmodule and intmodule}}

  module InProbeExt:
    inst e of BadExtMod

;// -----

FIRRTL version 3.0.0
circuit RefDupe:
  extmodule RefDupe:
    output p : Probe<UInt<2>>
    ref p is "x.y"
    ref p is "a.z" ; expected-error {{duplicate ref statement for 'p'}}

;// -----
; Either use ref statements or ABI, can't partially use ref statements.

FIRRTL version 3.0.0
circuit MissingRef:
  extmodule MissingRef:
    output p : Probe<UInt<2>> ; expected-error {{no ref statement found for ref port "p"}}
    output q : Probe<UInt<2>>
    ref q is "foo.bar"

;// -----

FIRRTL version 3.0.0
circuit UnusedRef:
  extmodule UnusedRef:
    ref p is "x.y" ; expected-error {{unused ref statement}}

;// -----

FIRRTL version 3.0.0
circuit RefNotString:
  extmodule RefNotString:
    output p : Probe<UInt<2>>
    ref p is a.b ; expected-error {{expected string in ref statement}}

;// -----

FIRRTL version 3.0.0
circuit RefForAggElement:
  extmodule RefForAggElement:
    output p : { x: Probe<UInt<2>> }
    ref p.x is a.b ; expected-error {{ref statements for aggregate elements not yet supported}}

;// -----

FIRRTL version 4.0.0
circuit ConnectWrongType:
  public module ConnectWrongType:
    input a: UInt<1>
    output b: SInt<1>

    connect b, a ; expected-error {{cannot connect non-equivalent type}}

;// -----
FIRRTL version 4.0.0

circuit RefStatementDeprecated:
  extmodule RefStatementDeprecated:
    output p : Probe<UInt<1>>
    ref p is "x.y" ; expected-error {{ref statements were removed in FIRRTL 4.0.0, but the specified FIRRTL version was 4.0.0}}

;// -----

FIRRTL version 3.0.0
circuit ConnectAlternateWrongType:
  module ConnectAlternateWrongType:
    input a: UInt<1>
    output b: SInt<1>

    connect b, a ; expected-error {{cannot connect non-equivalent type}}

;// -----

FIRRTL version 4.0.0
circuit ProbeConnect:
  public module ProbeConnect:
    input in : UInt<1>
    output p : Probe<UInt<1>>

    connect p, probe(in) ; expected-error {{cannot connect reference or property types}}

;// -----

FIRRTL version 3.0.0
circuit ProbeConnectAlternate:
  module ProbeConnectAlternate:
    input in : UInt<1>
    output p : Probe<UInt<1>>

    connect p, probe(in) ; expected-error {{cannot connect reference or property types}}

;// -----

FIRRTL version 4.0.0
circuit ProbeMem:
  public module ProbeMem:
    output out : Probe<UInt<3>> ; (wrong probe type, can't probe memories)

    mem memory:
      data-type => UInt<8>
      depth => 4
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    invalidate memory.r
    invalidate memory.w
    define out = probe(memory.r) ; expected-error {{cannot probe memories or their ports}}


;// -----

FIRRTL version 3.0.0
circuit ProbeSMem:
  module ProbeSMem:
    input clock : Clock
    input reset : Reset
    output io : { flip addr : UInt<3>, dataOut : UInt<8>}
    output out : Probe<UInt<8>>

    smem mem : UInt<8> [8]
    infer mport read = mem[io.addr], clock
    connect io.dataOut, read
    define out = probe(read) ; expected-error {{cannot probe memories or their ports}}
;// -----

FIRRTL version 3.0.0
circuit SmemNoRuwInfo:
  module SmemNoRuwInfo:
    input clock : Clock
    input reset : Reset
    output io : { flip addr : UInt<3>, dataOut : UInt<8>}

    smem mem : UInt<8> [8] @[foo.fir 1:1]
    infer mport read = mem[io.addr], clock
    connect io.dataOut, read


;// -----

FIRRTL version 4.0.0
circuit LeadingRead:
  public module LeadingRead:
    input in : UInt<1>
    read(probe(in)) ; expected-error {{unexpected read() as start of statement}}

;// -----

FIRRTL version 4.0.0
circuit LeadingProbe:
  public module LeadingProbe:
    input in : UInt<1>
    probe(in) ; expected-error {{unexpected probe() as start of statement}}

;// -----

FIRRTL version 4.0.0
circuit ForceProbe:
  public module ForceProbe:
    input in : UInt<1>
    force_initial(probe(in), UInt<1>(1)) ; expected-error {{expected rwprobe-type expression for force_initial destination, got '!firrtl.probe<uint<1>>'}}

;// -----

FIRRTL version 4.0.0
circuit ForceWithLiteral:
  public module ForceWithLiteral:
    input in : UInt<1>
    node n = in
    force_initial(rwprobe(n), 1) ; expected-error {{expected source expression in force_initial}}

;// -----

FIRRTL version 4.0.0
circuit ForceBadTypes:
  public module ForceBadTypes:
    input in : { a : UInt<5>, b : UInt<4> }
    node n = in
    ; expected-error @below {{incompatible force_initial source of type '!firrtl.uint<4>' cannot target destination '!firrtl.rwprobe<bundle<a: uint<5>, b: uint<4>>>'}}
    force_initial(rwprobe(n), in.b)

;// -----

FIRRTL version 4.0.0
circuit ForceBadTypesWithFlips:
  public module ForceBadTypesWithFlips:
    output out : { flip a : UInt<5> }
    wire w : { flip a : UInt<5> }
    connect out, w
    ; expected-error @below {{expected passive value for force_initial source, got '!firrtl.bundle<a flip: uint<5>>'}}
    force_initial(rwprobe(w), out)

;// -----

FIRRTL version 4.0.0
circuit RWProbeConst:
  extmodule RWProbeConst:
    output p : RWProbe<{a: const UInt}> ; expected-error {{rwprobe cannot contain const}}

;// -----

FIRRTL version 4.0.0
circuit RWProbeConstPort:
  public module RWProbeConstPort:
    input in : const UInt<1>[2]
    output p : RWProbe<UInt<1>>
    define p = rwprobe(in[1]) ; expected-error {{cannot force target of type '!firrtl.const.uint<1>'}}

;// -----

FIRRTL version 2.0.0
circuit PartialConnect_v2p0p0:
  module PartialConnect_v2p0p0:
    input a: UInt<1>
    output b: UInt<1>

    b <- a ; expected-error {{unexpected character after sign}}

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

FIRRTL version 4.0.0
circuit NodeProbe:
  extmodule Ref:
    output x : Probe<UInt<1>>
  public module NodeProbe:
    inst r of Ref
    node n = r.x ; expected-error {{Node cannot be analog and must be passive or passive under a flip}}

;// -----

FIRRTL version 4.0.0
circuit ConstProbe:
  public module ConstProbe:
    output x : const Probe<UInt<1>> ; expected-error {{only hardware types can be 'const'}}

;// -----

FIRRTL version 4.0.0
circuit MultiConst:
  public module MultiConst:
    output x : const const UInt<1> ; expected-error {{'const' can only be specified once on a type}}

;// -----

FIRRTL version 4.0.0
circuit DefinePromoteToRW:
  extmodule Ref:
    output ro : Probe<UInt<1>>
  public module DefinePromoteToRW:
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

FIRRTL version 4.0.0
circuit UnknownTypeAlias:
  public module UnknownTypeAlias:
    output x : Foo ; expected-error {{type identifier `Foo` is not declared}}

;// -----

FIRRTL version 4.0.0
circuit DuplicateAlias:
  type Foo = UInt
  type Foo = UInt; expected-error {{type alias `Foo` is already defined}}
  public module DuplicateAlias:

;// -----

FIRRTL version 4.0.0
circuit KeywordTypeName:
  type Clock = UInt; expected-error {{cannot use keyword 'Clock' for type alias name}}

;// -----

FIRRTL version 4.0.0
circuit KeywordTypeName:
  type module = UInt; expected-error {{cannot use keyword 'module' for type alias name}}

;// -----

FIRRTL version 3.1.0
circuit UnsupportedVersionDeclGroups:
  ; expected-error @below {{optional groups are a FIRRTL 3.2.0+ feature, but the specified FIRRTL version was 3.1.0}}
  declgroup A, bind:

;// -----

FIRRTL version 4.0.0
circuit RemovedDeclGroups:
  ; expected-error @below {{optional groups were removed in FIRRTL 3.3.0, but the specified FIRRTL version was 4.0.0}}
  declgroup A, bind:

;// -----

FIRRTL version 3.2.0
circuit UnsupportedVersionLayer:
  ; expected-error @below {{layers are a FIRRTL 3.3.0+ feature, but the specified FIRRTL version was 3.2.0}}
  layer A, bind:

;// -----

FIRRTL version 3.1.0
circuit UnsupportedVersionGroups:
  module UnsupportedVersionGroups:
    ; expected-error @below {{optional groups are a FIRRTL 3.2.0+ feature, but the specified FIRRTL version was 3.1.0}}
    group A:

;// -----

FIRRTL version 4.0.0
circuit RemovedGroups:
  public module RemovedGroups:
    ; expected-error @below {{optional groups were removed in FIRRTL 3.3.0, but the specified FIRRTL version was 4.0.0}}
    group A:

;// -----

FIRRTL version 3.2.0
circuit UnsupportedLayerBlock:
  module UnsupportedLayerBlock:
    ; expected-error @below {{layers are a FIRRTL 3.3.0+ feature, but the specified FIRRTL version was 3.2.0}}
    layerblock A:

;// -----

FIRRTL version 4.0.0
circuit UnsupportedLayerConvention:
  ; expected-error @below {{unknown convention 'foo'}}
  layer A, foo :

;// -----

FIRRTL version 4.0.0
circuit LayerAndCircuitShareName:
  ; expected-note @below {{see existing symbol definition here}}
  layer LayerAndCircuitShareName, bind :

  ; expected-error @below {{redefinition of symbol named 'LayerAndCircuitShareName'}}
  public module LayerAndCircuitShareName:

;// -----

FIRRTL version 4.0.0
circuit LayerEmptyOutputDir:
  ; expected-error @below {{output directory must not be blank}}
  layer MyLayer, bind, "":

  public module LayerEmptyOutputDir:

;// -----

FIRRTL version 4.2.0
circuit Top:
  ; expected-error @below {{class cannot be the top of a circuit}}
  class Top:

;// -----

FIRRTL version 4.2.0
circuit Top:
  ; expected-error @below {{extclass cannot be the top of a circuit}}
  extclass Top:

;// -----

FIRRTL version 4.2.0
circuit Top:
  public module Top:
  class Foo:
    ;; expected-error @below {{ports on classes must be properties}}
    input a: UInt<8>

;// -----

FIRRTL version 4.2.0
circuit Top:
  public module Top:
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
FIRRTL version 3.1.0

circuit Top:
  module Top:
    input a : Integer
    input b : Integer
    output c : Integer
    ; expected-error @below {{Integer arithmetic expressions are a FIRRTL 4.0.0+ feature, but the specified FIRRTL version was 3.1.0}}
    propassign c, integer_add(a, b)

;// -----
FIRRTL version 3.1.0

circuit Top:
  module Top:
    input a : Integer
    input b : Integer
    output c : Integer
    ; expected-error @below {{Integer arithmetic expressions are a FIRRTL 4.0.0+ feature, but the specified FIRRTL version was 3.1.0}}
    propassign c, integer_mul(a, b)

;// -----
FIRRTL version 3.1.0

circuit Top:
  module Top:
    input a : Integer
    input b : Integer
    output c : Integer
    ; expected-error @below {{Integer arithmetic expressions are a FIRRTL 4.0.0+ feature, but the specified FIRRTL version was 3.1.0}}
    propassign c, integer_shr(a, b)

;// -----
FIRRTL version 3.1.0

circuit Top:
  module Top:
    input a : Integer
    input b : Integer
    output c : Integer
    ; expected-error @below {{Integer arithmetic expressions are a FIRRTL 4.0.0+ feature, but the specified FIRRTL version was 3.1.0}}
    propassign c, integer_shl(a, b)

;// -----
FIRRTL version 4.0.0

; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  public module Top :
    input a : Integer
    output b : List<Integer>

    ; expected-error @below {{unexpected expression of type '!firrtl.integer' in List concat expression}}
    propassign b, list_concat(a)

;// -----
FIRRTL version 4.0.0

; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  public module Top :
    input a : List<Integer>
    input b : List<String>
    output c : List<Integer>

    ; expected-error @below {{unexpected expression of type '!firrtl.list<string>' in List concat expression of type '!firrtl.list<integer>'}}
    propassign c, list_concat(a, b)

;// -----
FIRRTL version 4.0.0

; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  public module Top :
    output c : List<Integer>

    ; expected-error @below {{need at least one List to concatenate}}
    propassign c, list_concat()

;// -----
FIRRTL version 4.2.0

circuit Top:
  public module Top:
    ; expected-error @below {{unknown class 'Missing'}}
    input in : Inst<Missing>

;// -----
FIRRTL version 4.2.0

circuit Top:
  public module Top:
    ; expected-error @below {{use of undefined class name 'Missing' in object}}
    object x of Missing

;// -----
FIRRTL version 4.2.0

circuit Top:
  class MyClass:
    skip

  public module Top:
    output str : String
    object x of MyClass
    ; expected-error @below {{unknown field 'missing' in type '!firrtl.class<@MyClass()>'}}
    propassign str, x.missing

;// -----
FIRRTL version 3.0.0

circuit Top:
  ; expected-error @below {{classes are a FIRRTL 4.2.0+ feature, but the specified FIRRTL version was 3.0.0}}
  class MyClass:
    skip

  module Top:
    object x of MyClass

;// -----
FIRRTL version 3.0.0

circuit Top:
  module Top:
    ; expected-error @below {{object statements are a FIRRTL 4.2.0+ feature, but the specified FIRRTL version was 3.0.0}}
    object x of MyClass

;// -----
FIRRTL version 3.0.0

circuit Top:
  module Top:
    ; expected-error @below {{Inst types are a FIRRTL 4.2.0+ feature, but the specified FIRRTL version was 3.0.0}}
    output o: Inst<MyClass>

;// -----
; Lists not yet supported.
FIRRTL version 3.1.0
circuit ListOfInt:
   module ListOfInt:
     ; expected-error @below {{Lists are a FIRRTL 4.0.0+ feature}}
     output a : List<Integer>

;// -----
FIRRTL version 4.0.0
circuit ListOfUInt:
   public module ListOfUInt:
     ; expected-error @below {{expected property type}}
     output a : List<UInt>

;// -----
; Wrong type of elements in List expression.
FIRRTL version 4.0.0
circuit ListOfUInt:
   public module ListOfUInt:
     output a : List<String>
     ; expected-error @below {{unexpected expression of type '!firrtl.integer' in List expression of type '!firrtl.string'}}
     propassign a, List<String>(Integer(5))

;// -----
; List should not be leading a statement.
FIRRTL version 4.0.0
circuit LeadingList:
  public module LeadingList:
    ; expected-error @below {{unexpected List<>() as start of statement}}
    List<String>()

;// -----
; Path should not be leading a statement.
FIRRTL version 4.0.0
circuit LeadingPath:
  public module LeadingPath:
    ; expected-error @below {{unexpected path() as start of statement}}
    path("")

;// -----
FIRRTL version 3.1.0
circuit PathVersion:
  module PathVersion:
    ; expected-error @below {{Paths are a FIRRTL 4.2.0+ feature, but the specified FIRRTL version was 3.1.0}}
    output path : Path
    propassign path, path("")

;// -----
; Path operation should have a single string argument.
FIRRTL version 4.2.0
circuit BadPathExpr:
  public module BadPathExp:
    output path : Path
    ; expected-error @below {{expected target string in path expression}}
    propassign path, path()

;// -----
; Path operation should have a single string argument.
FIRRTL version 4.2.0
circuit BadPathExpr:
  public module BadPathExp:
    output path : Path
    ; expected-error @below {{expected ')' in path expression}}
    propassign path, path("hello", "goodbye")

;// -----
; Path operation should have a single string argument.
FIRRTL version 4.2.0
circuit BadPathExpr:
  public module BadPathExp:
    output path : Path
    ; expected-error @below {{expected target string in path expression}}
    propassign path, path(UInt<1>(1))

;// -----
; CHeck version handling for Bool.
FIRRTL version 3.0.0

circuit Top:
  module Top:
    ; expected-error @below {{Bools are a FIRRTL 4.2.0+ feature, but the specified FIRRTL version was 3.0.0}}
    input in : Bool
    output out : Bool
    propassign out, in

;// -----
; Bool literal must be true or false.
FIRRTL version 4.2.0

circuit Top:
  public module Top:
    output out : Bool
    ; expected-error @below {{expected true or false in Bool expression}}
    propassign out, Bool(0)

;// -----
; Properties can't be const.
FIRRTL version 4.2.0

circuit Top:
  public module Top:
    ; expected-error @below {{only hardware types can be 'const'}}
    output out : const Bool

;// -----
; AnyRef not yet supported.
FIRRTL version 3.1.0
circuit AnyRef:
   module AnyRef:
     ; expected-error @below {{AnyRef types are a FIRRTL 4.2.0+ feature}}
     output a : AnyRef

;// -----
; Only objects are valid as AnyRef
FIRRTL version 4.2.0
circuit AnyRef:
  public module AnyRef:
    output out : AnyRef
    ; expected-error @below {{cannot propassign non-equivalent type '!firrtl.integer' to '!firrtl.anyref'}}
    propassign out, Integer(5)

;// -----
; Only objects are valid as AnyRef in Lists
FIRRTL version 4.2.0
circuit AnyRefList:
  public module AnyRefList:
    output list : List<AnyRef>
    ; expected-error @below {{unexpected expression of type '!firrtl.integer' in List expression of type '!firrtl.anyref'}}
    propassign list, List<AnyRef>(Integer(5))

;// -----
; Not Covariant.
FIRRTL version 4.2.0
circuit NotCovariant:
  class Class:

  public module NotCovariant:
    output list : List<AnyRef>

    object obj of Class
    ; expected-error @below {{cannot propassign non-equivalent type '!firrtl.list<class<@Class()>>' to '!firrtl.list<anyref>'}}
    propassign list, List<Inst<Class>>(obj)

;// -----
; Not Contravariant.
FIRRTL version 4.2.0
circuit NotContravariant:
  class Class:

  public module NotContravariant:
    output list : List<Inst<Class>>

    object obj of Class
    ; expected-error @below {{cannot propassign non-equivalent type '!firrtl.list<anyref>' to '!firrtl.list<class<@Class()>>'}}
    propassign list, List<AnyRef>(obj)

;// -----
; Double: must have digit before point.
FIRRTL version 4.2.0
circuit DoubleNegPeriod:
  public module DoubleNegPeriod:
    output d : Double
    ; expected-error @below {{unexpected character after sign}}
    propassign d, Double(-.5)

;// -----
; Double: must have digit after point.
FIRRTL version 4.2.0
circuit DoublePeriodEnd:
  public module DoublePeriodEnd:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(0.)

;// -----
; Double: Not an integer.
FIRRTL version 4.2.0
circuit DoubleInteger:
  public module DoubleInteger:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(0)

;// -----
; Double: don't support NaN or inf.
FIRRTL version 4.2.0
circuit DoubleNaN:
  public module DoubleNaN:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(NaN)

;// -----
; Double: Don't support hex.
FIRRTL version 4.2.0
circuit DoubleHex:
  public module DoubleHex:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(0x0)

;// -----
; Double: Don't suuport FIRRTL-y radix.
FIRRTL version 4.2.0
circuit DoubleRadix:
  public module DoubleRadix:
    output d : Double
    ; expected-error @below {{expected floating point in Double expression}}
    propassign d, Double(0hABC)

;// -----
FIRRTL version 3.2.0
circuit PublicModuleUnsupported:
  ; expected-error @below {{public modules are a FIRRTL 3.3.0+ feature}}
  public module PublicModuleUnsupported:

;// -----
FIRRTL version 4.0.0
circuit PublicNonModule:
  ; expected-error @below {{only modules may be public}}
  public extmodule Foo:
  public module PublicNonModule:

;// -----
FIRRTL version 4.0.0
circuit RWProbeExprOOB :
  public module RWProbeExprOOB :
    output p : RWProbe<UInt<1>>
    wire x : UInt<1>[2]
    invalidate x

    ; expected-error @below {{out of range index '100' for vector type '!firrtl.vector<uint<1>, 2>'}}
    define p = rwprobe(x[100])

   input out_0 : SInt<8>[5]
   connect out_0[4], out_0[5]

;// -----

FIRRTL version 3.9.0
circuit Foo:
  ; expected-error @below {{option groups/instance choices are a FIRRTL 4.2.0+ feature}}
  option Platform:
    FPGA
    ASIC

;// -----

FIRRTL version 3.9.0
circuit Foo:

  module Foo:
    ; expected-error @below {{option groups/instance choices are a FIRRTL 4.2.0+ feature}}
    instchoice inst of DefaultTarget Platform :
      FPGA => FPGATarget
      ASIC => ASICTarget

;// -----

FIRRTL version 4.2.0
circuit Foo:
  option Platform:
    FPGA

  module DefaultTarget:
    input clock: Clock

  module FPGATarget:
    input clock: UInt<1>

  public module Foo:
    ; expected-error @below {{instance case port 'clock' type does not match the default module port}}
    instchoice inst of DefaultTarget, Platform :
      FPGA => FPGATarget

;// -----

FIRRTL version 4.2.0
circuit Foo:
  option Platform:
    FPGA

  module DefaultTarget:
    input clock: Clock

  module FPGATarget:
    input x: UInt<1>

  public module Foo:
    ; expected-error @below {{instance case module port 'x' does not match the default module port 'clock'}}
    instchoice inst of DefaultTarget, Platform :
      FPGA => FPGATarget

;// -----

FIRRTL version 4.2.0
circuit Foo:
  option Platform:
    FPGA

  module DefaultTarget:
  module FPGATarget:

  public module Foo:
    ; expected-error @below {{use of undefined option case 'ASIC'}}
    instchoice inst of DefaultTarget, Platform :
      ASIC => FPGATarget

;// -----

FIRRTL version 4.2.0
circuit Foo:
  module DefaultTarget:
  module FPGATarget:

  public module Foo:
    ; expected-error @below {{use of undefined option group 'Platform'}}
    instchoice inst of DefaultTarget, Platform :
      ASIC => FPGATarget

;// -----

FIRRTL version 4.2.0
circuit Foo:
  ; expected-error @below {{duplicate option case definition 'FPGA'}}
  option Platform:
    FPGA
    FPGA

;// -----
FIRRTL version 3.1.0
circuit Foo:

  module Foo:
    ; expected-error @below {{colored probes are a FIRRTL 4.0.0+ feature, but the specified FIRRTL version was 3.1.0}}
    output a: Probe<UInt<1>, A>

;// -----
FIRRTL version 4.0.0
circuit Foo:
  layer A, bind:
  public module Foo:
    output a: Probe<
    ; expected-error @below {{expected probe data type}}
;// -----
FIRRTL version 4.0.0
circuit Foo:
  layer A, bind:
  public module Foo:
    ; expected-error @below {{expected '<' in reference type}}
    output a: Probe X

;// -----
FIRRTL version 4.0.0
circuit Foo:
  layer A, bind:
  public module Foo:
    output a: Probe
    ; expected-error @below {{expected '<' in reference type}}
;// -----
FIRRTL version 4.0.0
circuit Foo:
  layer A, bind:
  public module Foo:
    ; expected-error @below {{expected probe data type}}
    output a: Probe<>

;// -----
FIRRTL version 4.0.0
circuit Foo:
  layer A, bind:
  public module Foo:
    ; expected-error @below {{expected '>' to end reference type}}
    output a: Probe<UInt<1>
;// -----
FIRRTL version 4.0.0
circuit Foo:
  layer A, bind:
  public module Foo:
    ; expected-error @below {{expected '>' to end reference type}}
    output a: Probe<UInt<1>, A

;// -----
FIRRTL version 4.0.0
circuit Foo:
  layer A, bind:
  public module Foo:
    ; expected-error @below {{expected layer name}}
    output a: Probe<UInt<1>, A.>

;// -----
FIRRTL version 4.0.0
circuit UnknownWidthPublic:
  public module UnknownWidthPublic:
    ; expected-error @below {{public module port must have known width}}
    output a: UInt

;// -----
FIRRTL version 4.0.0
circuit UnknownWidthExt:
  extmodule UnknownWidthExt:
    ; expected-error @below {{extmodule port must have known width}}
    output a: { x: UInt }

;// -----
FIRRTL version 4.0.0
circuit AbstractResetPublic:
  public module AbstractResetPublic:
    ; expected-error @below {{public module port must have concrete reset type}}
    input r: Reset

;// -----
FIRRTL version 4.0.0
circuit PrivateMainModule:
  ; expected-error @below {{private main modules were removed in FIRRTL 4.0.0}}
  module PrivateMainModule:

;// -----
FIRRTL version 3.0.0
circuit IntModuleNoIntrinsic:
  intmodule test:
  ; expected-error @below {{expected 'intrinsic'}}
  module IntModule:

;// -----
FIRRTL version 3.0.0
circuit IntModuleBadIntrinsic:
  intmodule test:
    ; expected-error @below {{expected intrinsic name}}
    intrinsic = 0
  module IntModule:

;// -----
FIRRTL version 4.0.0
circuit GenericIntExprNoRet:
  public module GenericIntExprNoRet:
    ; expected-error @below {{expected ':' in intrinsic expression}}
    node n = intrinsic(dummy)

;// -----
FIRRTL version 4.0.0
circuit GenericIntNotPassive:
  public module GenericIntNotPassive:
    ; expected-error @below {{must be a passive base type}}
    intrinsic(dummy : { flip x : UInt<1> })

;// -----
FIRRTL version 4.0.0
circuit GenericIntBadId:
  public module GenericIntBadId:
    ; expected-error @below {{expected intrinsic identifier}}
    intrinsic(5)

;// -----
FIRRTL version 4.0.0
circuit GenericIntUntermStr:
  public module GenericIntUntermStr:
    ; expected-error @below {{unterminated string}}
    intrinsic<key = "val
      " ; end for syntax highlighting
;// -----
FIRRTL version 4.0.0
circuit GenericIntParamRedef:
  public module GenericIntParamRedef:
    ; expected-error @below {{redefinition of parameter 'x'}}
    intrinsic(dummy<x = 5, x = 3>)

;// -----
FIRRTL version 4.0.0
circuit IntrinsicModule:
  ; expected-error @below {{intrinsic modules were removed in FIRRTL 4.0.0}}
  intmodule MyIntModule :
    intrinsic = testIntrinsic1

;// -----
FIRRTL version 3.0.0
circuit RegWith:
  module RegWith:
    input clock: Clock

    ; expected-error @below {{'reg with' registers were removed in FIRRTL 3.0.0}}
    reg r_0 : UInt<5>, clock with :
      reset => (UInt<1>(0h0), r_0)

;// -----
FIRRTL version 3.0.0
circuit IsInvalid:
  module IsInvalid:
    output a: UInt<1>

    ; expected-error @below {{'is invalid' statements were removed in FIRRTL 3.0.0}}
    a is invalid

;// -----
FIRRTL version 3.0.0
circuit IsInvalid:
  module IsInvalid:
    input a: UInt<1>
    output b: UInt<1>

    ; expected-error @below {{'<=' connections were removed in FIRRTL 3.0.0}}
    b <= a

;// -----
FIRRTL version 4.0.0
circuit Foo:
  ; expected-error @below {{inline layers are a FIRRTL 4.1.0+ feature}}
  layer A, inline:
  public module Foo:

;// -----
FIRRTL version 4.1.0
circuit Top:
  public module Top:
    ; expected-error @below {{contracts are a FIRRTL 4.2.0+ feature}}
    contract:

// -----
FIRRTL version 4.0.0
circuit DuplicateWhenElse:
  public module DuplicateWhenElse:
    input reset: UInt<1>
    output _t: UInt<1>
    wire _t_2: UInt<1>

    when reset :
      node n4 = _t_2
      connect _t, n4
    else :
      ; expected-error @+1 {{redefinition of name 'n4' - FIRRTL has flat namespace and requires all declarations in a module to have unique names}}
      node n4 = _t_2   ; 'n4' is duplicated across scopes
      connect _t, n4

// -----
FIRRTL version 4.0.0
circuit SameScopeRedef:
  public module SameScopeRedef:
    input reset: UInt<1>
    output _t: UInt<1>
    wire _t_2: UInt<1>

    ; expected-note @+1 {{previous definition here}}
    node x = _t_2
    ; expected-error @+1 {{redefinition of name 'x'}}
    node x = _t_2   ; 'x' is duplicated in same scope
    connect _t, x

// -----
FIRRTL version 4.0.0
circuit NestedWhenRedef:
  public module NestedWhenRedef:
    input reset: UInt<1>
    output _t: UInt<1>
    wire _t_2: UInt<1>

    when reset :
      ; expected-note @+1 {{previous definition here}}
      node val = _t_2
      when reset :
        ; expected-error @+1 {{redefinition of name 'val'}}
        node val = _t_2   ; 'val' is duplicated in nested when
      connect _t, val
