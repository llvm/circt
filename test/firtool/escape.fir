; RUN: firtool %s -verilog | FileCheck %s


circuit unicode_escape_in_string :
  module unicode_escape_in_string :
    input clock : Clock
    input en : UInt<1>
    ; CHECK: "PreservesUnicode \u2019s"
    printf(clock, en, "PreservesUnicode \u2019s")

    ; CHECK: "PreservesWhitespace \n \t \b"
    printf(clock, en, "PreservesWhitespace \n \t \b")

    ; CHECK: "PreservesQuotes \' \""
    printf(clock, en, "PreservesQuotes \' \"")
