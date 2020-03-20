; expected-error {{firrtl parsing not implemented}}

; RUN: spt-translate -import-firrtl -verify-diagnostics --split-input-file %s 

module MyModule :

  ; expected-error @+1 {{unterminated string}}
  parameter FORMAT = "tilelink_timeou


; ------

; expected-error @+1 {{unexpected character}}
@

; ------

; expected-error @+1 {{unexpected character}}
!
