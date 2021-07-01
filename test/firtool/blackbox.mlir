// RUN: rm -rf %t
// RUN: firtool %s --blackbox-resource-path=%S/.. | firtool --format=mlir --split-verilog -o=%t --blackbox-path=%S --blackbox-resource-path=%S/..
// RUN: FileCheck %s --check-prefix=VERILOG-TOP < %t/test_mod.sv
// RUN: FileCheck %s --check-prefix=VERILOG-FOO < %t/magic/blackbox-inline.v
// RUN: FileCheck %s --check-prefix=VERILOG-HDR < %t/magic/blackbox-inline.svh
// RUN: FileCheck %s --check-prefix=VERILOG-BAR < %t/magic/blackbox-resource.v
// RUN: FileCheck %s --check-prefix=VERILOG-GIB < %t/magic/blackbox-path.v
// RUN: FileCheck %s --check-prefix=LIST-TOP < %t/filelist.f
// RUN: FileCheck %s --check-prefix=LIST-BLACK-BOX < %t/magic.f

// LIST-TOP: test_mod.sv

// LIST-BLACK-BOX:      magic/blackbox-inline.v
// LIST-BLACK-BOX-NEXT: magic/blackbox-path.v
// LIST-BLACK-BOX-NEXT: magic/blackbox-resource.v

firrtl.circuit "test_mod" attributes {annotations = [
  // Black box processing should honor only the last annotation.
  {class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "ignore_me_plz"},
  {class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "magic"},
  {class = "firrtl.transforms.BlackBoxResourceFileNameAnno", resourceFileName = "definitely_bad.f"},
  {class = "firrtl.transforms.BlackBoxResourceFileNameAnno", resourceFileName = "magic.f"}
]} {
  // VERILOG-TOP-LABEL: module test_mod
  // VERILOG-TOP-NEXT:    ExtInline foo
  // VERILOG-TOP-NEXT:    ExtResource bar
  // VERILOG-TOP-NEXT:    ExtPath gib
  // VERILOG-TOP-NEXT:  endmodule
  firrtl.module @test_mod() {
    firrtl.instance @ExtInline {name = "foo", portNames = []}
    firrtl.instance @ExtResource {name = "bar", portNames = []}
    firrtl.instance @ExtPath {name = "gib", portNames = []}
  }

  // VERILOG-FOO-LABEL: module ExtInline(); endmodule
  // VERILOG-FOO-NOT:   module ExtInline(); endmodule
  // VERILOG-HDR-LABEL: `define SOME_MACRO
  // VERILOG-HDR-NOT:   `define SOME_MACRO
  firrtl.extmodule @ExtInline() attributes {annotations = [
    // Inline file shall be emitted.
    {
      class = "firrtl.transforms.BlackBoxInlineAnno",
      name = "blackbox-inline.v",
      text = "module ExtInline(); endmodule\n"
    },
    // Duplicate inline annotations will not be emitted.
    {
      class = "firrtl.transforms.BlackBoxInlineAnno",
      name = "blackbox-inline.v",
      text = "module ExtInline(); endmodule\n"
    },
    // Verilog header `*.svh` shall be excluded from the file list.
    {
      class = "firrtl.transforms.BlackBoxInlineAnno",
      name = "blackbox-inline.svh",
      text = "`define SOME_MACRO\n"
    }
  ]}

  // VERILOG-BAR-LABEL: module ExtResource(); endmodule
  // VERILOG-BAR-NOT:   module ExtResource(); endmodule
  firrtl.extmodule @ExtResource() attributes {annotations = [{
    class = "firrtl.transforms.BlackBoxResourceAnno",
    resourceId = "firtool/blackbox-resource.v"
  }]}

  // Duplicate resources will not be copied.
  // VERILOG-BAR-NOT:   module ExtResource(); endmodule
  firrtl.extmodule @DuplicateExtResource() attributes {annotations = [{
    class = "firrtl.transforms.BlackBoxResourceAnno",
    resourceId = "firtool/blackbox-resource.v"
  }]}

  // VERILOG-GIB-LABEL: module ExtPath(); endmodule
  // VERILOG-GIB-NOT:   module ExtPath(); endmodule
  firrtl.extmodule @ExtPath() attributes {annotations = [{
    class = "firrtl.transforms.BlackBoxPathAnno",
    path = "blackbox-path.v"
  }]}
}
