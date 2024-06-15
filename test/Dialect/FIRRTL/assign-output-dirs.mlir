// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-assign-output-dirs{output-dir=/path/to/output}))' %s | FileCheck %s

firrtl.circuit "AssignOutputDirs" {
  // Directory Tree
  //        R
  //    A       B
  //  C   D

  firrtl.module public @AssignOutputDirs() {}

  // R -> R
  // CHECK: firrtl.module private @ByR() {
  firrtl.module private @ByR() {}

  // R & A -> R
  // CHECK: firrtl.module private @ByRA() {
  firrtl.module private @ByRA() {}

  // R & C -> R
  // CHECK: firrtl.module private @ByRC() {
  firrtl.module private @ByRC() {}

  // A -> A
  // CHECK: firrtl.module private @ByA() attributes {output_file = #hw.output_file<"A{{/|\\\\}}">} {
  firrtl.module private @ByA() {}
  
  // A & B -> R
  // firrtl.module private @ByAB() {
  firrtl.module private @ByAB() {}

  // C & D -> A
  // CHECK: firrtl.module private @ByCD() attributes {output_file = #hw.output_file<"A{{/|\\\\}}">} {
  firrtl.module private @ByCD() {}

  // A & C -> A
  // CHECK: firrtl.module private @ByAC() attributes {output_file = #hw.output_file<"A{{/|\\\\}}">} {
  firrtl.module private @ByAC() {}

  // B & C -> R
  // CHECK: firrtl.module private @ByBC() {
  firrtl.module private @ByBC() {}

  firrtl.module @InR() {
    firrtl.instance r  @ByR()
    firrtl.instance ra @ByRA()
    firrtl.instance rc @ByRC()
  }

  firrtl.module @InA() attributes {output_file = #hw.output_file<"A/foo">} {
    firrtl.instance ra @ByRA()
    firrtl.instance ab @ByAB()
    firrtl.instance a  @ByA()
    firrtl.instance ac @ByAC()
  }

  firrtl.module @InB() attributes {output_file = #hw.output_file<"B/foo">} {
    firrtl.instance ab @ByAB()
    firrtl.instance bc @ByBC()
  }

  firrtl.module @InC() attributes {output_file = #hw.output_file<"A/C/">} {
    firrtl.instance cd @ByCD()
    firrtl.instance bc @ByBC()
  }

  firrtl.module @InD() attributes {output_file = #hw.output_file<"A/D/">} {
    firrtl.instance byCD @ByCD()
  }

  // CHECK: firrtl.module private @ByDotDot() attributes {output_file = #hw.output_file<"{{.*(/|\\\\)}}path{{/|\\\\}}to{{/|\\\\}}">} {
  firrtl.module private @ByDotDot() {}

  firrtl.module @InDotDot() attributes {output_file = #hw.output_file<"../">} {
    firrtl.instance byDotDot @ByDotDot()
  }

  // Absolute output directory tests

  // CHECK firrtl.module private @ByOutputA() {output_file = #hw.output_file<"A{{/|\\\\}}">} {}
  firrtl.module private @ByOutputA() {}

  firrtl.module @InOutputA() attributes {output_file = #hw.output_file<"/path/to/output/A/foo">} {
    firrtl.instance byOutputA @ByOutputA()
  }

  // CHECK: firrtl.module private @ByYZ() attributes {output_file = #hw.output_file<"{{.*(/|\\\\)}}X{{/|\\\\}}">} {
  firrtl.module private @ByYZ() {}

  firrtl.module @InY() attributes {output_file = #hw.output_file<"/X/Y/">} {
    firrtl.instance byYZ @ByYZ()
  }
  firrtl.module @InZ() attributes {output_file = #hw.output_file<"/X/Z/">} {
    firrtl.instance byYZ @ByYZ()
  }
}
