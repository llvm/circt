; RUN: circt-translate -import-firrtl -verify-diagnostics %s

; This must end the file, need a `\` directly before EOF (no newline).
; Put other tests in annotations-errors.fir
FIRRTL version 4.0.0

; expected-error @below {{unterminated inline annotation}}
circuit test :%[[{

\