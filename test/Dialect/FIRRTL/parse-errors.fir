; RUN: circt-translate -parse-fir -verify-diagnostics --split-input-file %s 

circuit test :
  extmodule MyModule :

    ; expected-error @+1 {{unterminated string}}
    parameter FORMAT = "tilelink_timeou

;// -----

circuit test test : ; expected-error {{expected ':' in circuit definition}}

;// -----

; expected-error @+1 {{unexpected character}}
@

;// -----

circuit test :
  module nameConflict :
    input c: Clock   ; expected-note {{previous definition here}}
    input r: Reset
    input c: Analog  ; expected-error {{redefinition of name 'c'}}

;// -----

circuit test :
  extmodule nameConflict :
    input c: Clock   ; expected-note {{previous definition here}}
    input r: Reset
    input c: Analog  ; expected-error {{redefinition of name 'c'}}

;// -----

circuit test :
  module invalid_name :
    input c: UInt
    out <= c         ; expected-error {{use of unknown declaration 'out'}}

;// -----

circuit test :
  module invalid_name :
    output out: UInt
    out <= c         ; expected-error {{use of unknown declaration 'c'}}

;// -----

circuit test :
  module invalid_name :
   input out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}
   ; expected-error @+1 {{unknown field 'xx' in bundle type '!firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>'}}
   out_0.xx <- out_0.yy

;// -----

circuit test :
  module invalid_name :
   input out_0 : SInt<8>[5]
   ; expected-error @+1 {{out of range index '5' in vector type '!firrtl.vector<sint<8>, 5>'}}
   out_0[4] <- out_0[5]

;// -----

circuit test :
  module invalid_add :
   input in : SInt<8>
   input c : Clock
   ; expected-error @+1 {{second operand must be an integer type, not '!firrtl.clock'}}
   node n = add(in, c)

;// -----

circuit test :
  module invalid_add :
   input in : SInt<8>
   input c : Clock
   ; expected-error @+1 {{operation requires two operands and no constants}}
   node n = add(in, in, in)

;// -----

circuit test :
  module invalid_int_literal :
   node n = add(UInt<8>("hAX"), UInt<10>(42))  ; expected-error {{invalid character in integer literal}}

;// -----
; When scopes are local to the body
circuit test :
  module invalid_name :
    input reset : UInt<1>
    output out : UInt<1>
    when reset :
      node n4 = reset
    out <- n4   ; expected-error {{use of unknown declaration 'n4'}}

;// -----

circuit test :
  module invalid_inst :

    ; expected-error @+1 {{use of undefined module name 'some_module' in instance}}
    inst xyz of some_module
 
;// -----

circuit test :
  extmodule MyParameterizedExtModule :
    parameter DEFAULT = 0
    parameter DEFAULT = 32 ; expected-error {{redefinition of parameter 'DEFAULT'}}

;// -----

circuit test :
  module invalid_name :
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    node n4 = add(bf, bf)  ; expected-error {{operands must be integer types, not '!firrtl.bundle<int_1: uint<1>, int_out: uint<2>>' and '!firrtl.bundle<int_1: uint<1>, int_out: uint<2>>'}}

;// -----

circuit test :
  module invalid_bits:
     input a: UInt<8>
     output b: UInt<4>
     ; expected-error@+1 {{high must be equal or greater than low, but got high = 4, low = 7}};
     b <= bits(a, 4, 7)

;// -----

circuit test :
  module invalid_add :
   input in1 : SInt
   input in2 : UInt
   node n = add(in1, in2)  ; expected-error {{operand signedness must match}}
