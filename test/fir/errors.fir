; RUN: spt-translate -import-firrtl -verify-diagnostics --split-input-file %s 

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
    out <= c         ; expected-error {{use of invalid name 'out'}}

;// -----

circuit test :
  module invalid_name :
    output out: UInt
    out <= c         ; expected-error {{use of invalid name 'c'}}

;// -----

circuit test :
  module invalid_name :
   input out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}
   out_0.xx <- out_0.yy  ; expected-error {{invalid subfield 'xx' for type '!firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>'}}

;// -----

circuit test :
  module invalid_name :
   input out_0 : SInt<8>[5]
   out_0[4] <- out_0[5]  ; expected-error {{invalid subindex '5' for type '!firrtl.vector<sint<8>, 5>'}}

