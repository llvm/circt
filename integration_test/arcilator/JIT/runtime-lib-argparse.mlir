// RUN: arcilator %s --run --jit-entry=main > %t.out 2>%t.err
// RUN: FileCheck %s --check-prefixes=STDOUT --input-file %t.out
// RUN: FileCheck %s --check-prefixes=STDERR --input-file %t.err
// REQUIRES: arcilator-jit

// Test the parser of the runtime argument string

// STDOUT:      [ArcRuntime] Argument string for instance ID 0: debug;simple;empty=;;noQuote=foo;quote="foo \\ \" bar ";semi="a;b;c"
// STDOUT-NEXT: [ArcRuntime] Parsed argument(s):
// STDOUT-NEXT: [ArcRuntime]   debug
// STDOUT-NEXT: [ArcRuntime]   empty = ""
// STDOUT-NEXT: [ArcRuntime]   noQuote = "foo"
// STDOUT-NEXT: [ArcRuntime]   quote = "foo \ " bar "
// STDOUT-NEXT: [ArcRuntime]   semi = "a;b;c"
// STDOUT-NEXT: [ArcRuntime]   simple

// STDERR:      [ArcRuntime] WARNING: Malformed runtime argument: Invalid key, ignoring
// STDERR-NEXT: [ArcRuntime] WARNING: Malformed runtime argument: Invalid key, ignoring
// STDERR-NEXT: [ArcRuntime] WARNING: Malformed runtime argument: Invalid key, ignoring
// STDERR-NEXT: [ArcRuntime] WARNING: Malformed runtime argument: Unquoted value contains forbidden character '"' for key "badVal", ignoring
// STDERR-NEXT: [ArcRuntime] WARNING: Malformed runtime argument: Unexpected content after closing quote for key "badVal2", ignoring
// STDERR-NEXT: [ArcRuntime] WARNING: Malformed runtime argument: Invalid escape sequence in quoted value for key "badVal3", ignoring
// STDERR-NEXT: [ArcRuntime] WARNING: Malformed runtime argument: Unterminated quoted value for key "unterm", ignoring

// STDOUT:      [ArcRuntime] Argument string for instance ID 1: debug;=badKey;=;"=;badVal=x"";badVal2=""x;badVal3="\x";foo;unterm="
// STDOUT-NEXT: [ArcRuntime] Parsed argument(s):
// STDOUT-NEXT: [ArcRuntime]   debug
// STDOUT-NEXT: [ArcRuntime]   foo

// STDERR:      [ArcRuntime] WARNING: Malformed runtime argument: Truncated escape sequence in quoted value for key "truncEscape", ignoring
// STDOUT-NOT:  [ArcRuntime] Argument string
// STDOUT-NOT:  [ArcRuntime] Parsed argument

// STDOUT:      [ArcRuntime] Argument string for instance ID 3: workDir=tmpWorkDir;traceFile=overridden;debug;traceFile=overriding
// STDOUT-NEXT: [ArcRuntime] Parsed argument(s):
// STDOUT-NEXT: [ArcRuntime]   debug
// STDOUT-NEXT: [ArcRuntime]   traceFile = "overriding"
// STDOUT-NEXT: [ArcRuntime]   workDir = "tmpWorkDir"
// STDOUT-NEXT: [ArcRuntime] Working directory for instance ID 3: {{.+}}{{/|\\}}tmpWorkDir


hw.module @dummy() {
  hw.output
}

func.func @main() {
  arc.sim.instantiate @dummy as %model runtime ("debug;simple;empty=;;noQuote=foo;quote=\"foo \\\\ \\\" bar \";semi=\"a;b;c\"") {
  }
  arc.sim.instantiate @dummy as %model runtime ("debug;=badKey;=;\"=;badVal=x\"\";badVal2=\"\"x;badVal3=\"\\x\";foo;unterm=\"") {
  }
  arc.sim.instantiate @dummy as %model runtime ("foo;truncEscape=\"\\") {
  }
  arc.sim.instantiate @dummy as %model runtime ("workDir=tmpWorkDir;traceFile=overridden;debug;traceFile=overriding") {
  }
  return
}
