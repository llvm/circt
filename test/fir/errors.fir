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

