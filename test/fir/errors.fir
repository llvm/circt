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

