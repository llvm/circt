; Check that a file does not exist if there is no error
; RUN: firtool %s  --output-error-diagnostics=%t && not cat %t
FIRRTL version 5.0.0
circuit test :
  public module test :
    skip
