// RUN: arcilator %s --run --jit-entry=main --jit-object-file=%t && test -s %t
// REQUIRES: arcilator-jit

func.func @main() {
  return
}
