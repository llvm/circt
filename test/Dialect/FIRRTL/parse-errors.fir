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
    ; expected-error @+1 {{subfield requires bundle operand}}
    b <= a.a

;// -----

circuit invalid_name :
  module invalid_name :
   input out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}
   ; expected-error @+1 {{unknown field 'xx' in bundle type '!firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>'}}
   out_0.xx <- out_0.yy

;// -----

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

circuit Issue426:
  module Issue426:
    ; expected-error @+1 {{zero bit constants are not allowed}}
    node x = UInt<0>("h0")

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

circuit Issue886:
  module Issue886 :
    input a: SInt<1>
    input b: UInt<42>
    node n = validif(a, b)  ; expected-error {{first operand should have UInt type}}

;// -----

circuit Issue886:
  module Issue886 :
    input a: UInt<2>
    input b: UInt<42>
    node n = validif(a, b)  ; expected-error {{first operand should have 'uint<1>' type}}

;// -----

circuit Issue886:
  module Issue886 :
    input a: UInt<1>
    input b: UInt<42>
    node n = validif(a, b, b)  ; expected-error {{operation requires two operands}}

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
