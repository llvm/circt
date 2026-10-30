; RUN: circt-translate --import-firrtl --verify-diagnostics --split-input-file %s

FIRRTL version 7.0.0
circuit WrongElement:
  public module WrongElement:
    input items: MutableCollection<Integer>
    ; expected-error @+1 {{push element type does not match collection element type}}
    push items, String("wrong")

;// -----

FIRRTL version 7.0.0
circuit WrongTarget:
  public module WrongTarget:
    input items: Integer
    ; expected-error @+1 {{push requires a MutableCollection handle}}
    push items, Integer(1)

;// -----

FIRRTL version 7.0.0
circuit NotEmpty:
  public module NotEmpty:
    output items: MutableCollection<Integer>
    ; expected-error @+1 {{expected ')' in empty MutableCollection expression}}
    propassign items, MutableCollection<Integer>(Integer(1))
