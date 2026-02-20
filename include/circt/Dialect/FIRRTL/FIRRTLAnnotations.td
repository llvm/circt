//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the table of FIRRTL annotations supported by CIRCT.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target Types
//===----------------------------------------------------------------------===//

// Base class for all target types. Each target type specifies the MLIR
// operation it represents.
class TargetType<string opName = ""> {
  // The name of the MLIR operation this target type represents
  // e.g., "WireOp", "FModuleOp", "FExtModuleOp"
  string operation = opName;
}

// Module-level target types.
def Module : TargetType<"FModuleOp">;
def ExtModule : TargetType<"FExtModuleOp">;

// Component-level target types.
def Wire : TargetType<"WireOp">;
def Node : TargetType<"NodeOp">;
def Reg : TargetType<"RegOp">;
def RegReset : TargetType<"RegResetOp">;
def Instance : TargetType<"InstanceOp">;
def Mem : TargetType<"MemOp">;
def CombMem : TargetType<"CombMemOp">;
def MemoryPort : TargetType<"MemoryPortOp">;
def SeqMem : TargetType<"SeqMemOp">;

//===----------------------------------------------------------------------===//
// Parameter Type Classes
//===----------------------------------------------------------------------===//

// DAG operator for annotation members.
def ins;

// Base class for annotation parameter types.
class ParamType;

def StringParam : ParamType;

def IntegerParam : ParamType;

def BooleanParam : ParamType;

// Object parameter type.
// Can be used in two ways:
// 1. With a DAG of fields: ObjectParam<(ins StringParam:$field1, IntegerParam:$field2)>
//    This creates a structured object with named, typed fields
// 2. As a base class for reusable object types:
//    def MyObjectType : ObjectParam<(ins StringParam:$x, IntegerParam:$y)>;
class ObjectParam<dag fields = (ins)> : ParamType {
  dag objectFields = fields;
  code description = "";
}

// Array parameter type with element type.
// Example: ArrayParam<StringParam> for an array of strings
class ArrayParam<ParamType elementType> : ParamType {
  ParamType element = elementType;
}

// Target parameter type (FIRRTL reference target)
def TargetParam : ParamType;

// Enum parameter (string with allowed values)
class EnumParam<list<string> values> : ParamType {
  list<string> allowedValues = values;
}

// Union parameter (one of several allowed types)
// Used to specify that a field can be one of multiple specific types.
// Example: UnionParam<[AugmentedGroundType, AugmentedVectorType, AugmentedBundleType]>
// Example: UnionParam<[StringParam, IntegerParam]>
class UnionParam<list<ParamType> types> : ParamType {
  list<ParamType> allowedTypes = types;
}

// Decorates a parameter as optional with an optional default value.
class Optional<ParamType paramType, string defaultVal = ""> : ParamType {
  ParamType baseType = paramType;
  string defaultValue = defaultVal;
}

// Helper class to attach documentation to a parameter type. This can ONLY be
// used at the outermost layer of annotation members and ObjectParam fields.
class Doc<ParamType paramType, string desc> {
  ParamType baseType = paramType;
  string documentation = desc;
}

//===----------------------------------------------------------------------===//
// Annotation Class
//===----------------------------------------------------------------------===//

// Base class for all FIRRTL annotations
// Matches Scala: trait Annotation
// Inherits from ParamType so annotations can be used as member types in other
// annotations.
class Annotation : ParamType {
  // The fully-qualified class name (e.g., "firrtl.transforms.DontTouchAnnotation")
  string className;

  // DAG of annotation members (parameters). For SingleTargetAnnotation, the
  // 'target' parameter is automatically added by the generator.
  // Format: (ins ParamType:$paramName, ParamType:$paramName, ...)
  dag members = (ins);

  // Description for documentation.
  string description = "";

  // Custom handler function name for LowerAnnotations pass
  // If empty, the handler is automatically generated from annotation metadata
  // If set, this custom handler function is used instead
  // e.g., "applyDUTAnno", "applyConventionAnno", "applyAttributeAnnotation"
  string customHandler = "";
}

// NoTargetAnnotation - Circuit-level annotations with no target. These
// annotations are always attached to the circuit and have no 'target' field in
// JSON.
class NoTargetAnnotation : Annotation {
}

// SingleTargetAnnotation - Annotations targeting exactly one named element. The
// generator automatically adds a 'target' parameter to the members dag.
class SingleTargetAnnotation : Annotation {
  // Whether non-local annotations are allowed (can span module boundaries)
  bit allowNonLocal = 0;

  // Whether port annotations are allowed
  bit allowPortTargets = 0;

  // List of allowed target types
  // Empty list means any named target is allowed
  list<TargetType> targets = [];
}

// IgnoredAnnotation - Annotations that are recognized but ignored by the
// compiler. These annotations are dropped during lowering and do not affect
// compilation. Use this for annotations that are only meaningful to external
// tools or for documentation.
class IgnoredAnnotation : Annotation { }

//===----------------------------------------------------------------------===//
// Annotation Definitions
//===----------------------------------------------------------------------===//

def DontTouchAnnotation : SingleTargetAnnotation {
  let className = "firrtl.transforms.DontTouchAnnotation";
  let allowNonLocal = 1;
  let allowPortTargets = 1;
  let targets = [Wire, Node, Reg, RegReset, Instance, Mem, CombMem, MemoryPort, SeqMem];
  let description = [{
    Prevents the removal of elements through optimization.

    This annotation is an optimization barrier. For example, it blocks constant
    propagation through it. This annotation also ensures that the name of the
    object is preserved, and not discarded or modified.

    Example:
    ```json
    {
      "class": "firrtl.transforms.DontTouchAnnotation",
      "target": "~Foo|Bar/d:Baz"
    }
    ```

    Scala documentation: https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/DontTouchAnnotation.html
  }];
}

def AttributeAnnotation : SingleTargetAnnotation {
  let className = "firrtl.AttributeAnnotation";
  let members = (ins
    Doc<StringParam, "An attribute">:$description
  );
  let description = [{
    Attaches SV attributes to a specified target. A reference target must be a
    wire, node, reg, or module.

    This annotation doesn't prevent optimizations so it's necessary to add
    `DontTouchAnnotation` if users want to preserve the target.

    Example:
    ```json
    {
      "class": "firrtl.AttributeAnnotation",
      "target": "~Foo|Foo>r",
      "description": "debug = \"true\""
    }
    ```

    Scala documentation: https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/AttributeAnnotation.html
  }];
  let customHandler = "applyAttributeAnnotation";
}

def DocStringAnnotation : SingleTargetAnnotation {
  let className = "firrtl.DocStringAnnotation";
  let members = (ins
    Doc<StringParam, "An attribute">:$description
  );
  let description = [{
    Attaches a comment to a specified target. A reference target must be a wire,
    node, reg, or module.

    This annotation doesn't prevent optimizations so it's necessary to add
    `DontTouchAnnotation` if users want to preserve the target.

    Example:
    ```json
    {
      "class": "firrtl.DocStringAnnotation",
      "target": "~Foo|Foo>r",
      "description": "comment"
    }
    ```

    Scala documentation: https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/DocStringAnnotation.html
  }];
}

def FlattenAnnotation : SingleTargetAnnotation {
  let className = "firrtl.transforms.FlattenAnnotation";
  let targets = [Module];
  let description = [{
    Indicates that the target should be flattened, which means that child
    instances will be recursively inlined.
    
    Example:
    ```json
    {
      "class": "firrtl.transforms.FlattenAnnotation",
      "target": "~Foo|Bar/d:Baz"
    }
    ```
    
    Scala documentation: https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/FlattenAnnotation.html
  }];
}

def InlineAnnotation : SingleTargetAnnotation {
  let className = "firrtl.passes.InlineAnnotation";
  let targets = [Module];
  let description = [{
    Indicates that the target should be inlined.

    Example:
    ```json
    {
      "class": "firrtl.passes.InlineAnnotation",
      "target": "~Foo|Bar/d:Baz"
    }
    ```

    Scala documentation: https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/passes/InlineAnnotation.html
  }];
}

//===----------------------------------------------------------------------===//
// BlackBox Annotations
//===----------------------------------------------------------------------===//

def BlackBox : SingleTargetAnnotation {
  let className = "firrtl.transforms.BlackBox";
  let targets = [ExtModule];
  let description = [{
    Attached to any external module created from any of the other blackbox
    annotations, such as `BlackBoxInlineAnno`.

    This is used when generating metadata about external modules to distinguish
    generated modules. This annotation is internal to the MLIR FIRRTL compiler.
    
    Example:
    ```json
    {
      "class": "firrtl.transforms.BlackBox",
      "target": "~Foo|Foo"
    }
    ```
  }];
}

def BlackBoxInlineAnno : SingleTargetAnnotation {
  let className = "firrtl.transforms.BlackBoxInlineAnno";
  let targets = [ExtModule];
  let members = (ins
    Doc<StringParam, "A full path to a file">:$name,
    Doc<StringParam, "Literal verilog code">:$text
  );
  let description = [{
    Specifies the black box source code (`text`) inline. Generates a file with
    the given `name` in the target directory.

    Example:
    ```json
    {
      "class": "firrtl.transforms.BlackBoxInlineAnno",
      "target": "~Foo|Foo",
      "name": "blackbox-inline.v",
      "text": "module ExtInline(); endmodule\n"
    }
    ```

    Scala documentation: https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/BlackBoxInlineAnno.html
  }];
}

def BlackBoxPathAnno : SingleTargetAnnotation {
  let className = "firrtl.transforms.BlackBoxPathAnno";
  let targets = [ExtModule];
  let members = (ins
    Doc<StringParam, "ModuleName target">:$path
  );
  let description = [{
    Specifies the file `path` as source code for the module. Copies the file to
    the target directory.

    Example:
    ```json
    {
      "class": "firrtl.transforms.BlackBoxPathAnno",
      "target": "~Foo|Foo",
      "path": "myfile.v"
    }
    ```

    Scala documentation: https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/BlackBoxPathAnno.html
  }];
}

def BlackBoxTargetDirAnno : NoTargetAnnotation {
  let className = "firrtl.transforms.BlackBoxTargetDirAnno";
  let members = (ins
    Doc<StringParam, "Output directory">:$targetDir
  );
  let description = [{
    Overrides the target directory into which black box source files are emitted.

    Example:
    ```json
    {
      "class": "firrtl.transforms.BlackBoxTargetDirAnno",
      "targetDir": "/tmp/circt/output"
    }
    ```
    
    Scala documentation: https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/BlackBoxTargetDirAnno.html
  }];
}

//===----------------------------------------------------------------------===//
// CIRCT-Specific Annotations
//===----------------------------------------------------------------------===//

def FullResetAnnotation : SingleTargetAnnotation {
  let className = "circt.FullResetAnnotation";
  let allowPortTargets = 1;
  let members = (ins
    Doc<EnumParam<["async", "sync"]>, "Type of reset">:$resetType
  );
  let description = [{
    The target must be a signal that is a reset. The type of the signal must be
    (or inferred to be) the same as the reset type specified in the annotation.

    Indicates that all reset-less registers which are children of the module
    containing the target will have the reset targeted attached, with a reset
    value of 0.

    The module containing the target of this annotation is not allowed to reside
    in multiple hierarchies.

    Example:
    ```json
    {
      "class": "circt.FullResetAnnotation",
      "target": "~Foo|Bar/d:Baz>reset",
      "resetType": "async"
    }
    ```
  }];
}

def ExcludeFromFullResetAnnotation : SingleTargetAnnotation {
  let className = "circt.ExcludeFromFullResetAnnotation";
  let allowNonLocal = 1;
  let targets = [Module];
  let description = [{
    This annotation indicates that the target module should be excluded from the
    FullResetAnnotation of a parent module.

    Example:
    ```json
    {
      "class": "circt.IgnoreFullAsyncResetAnnotation",
      "target": "~Foo|Bar/d:Baz"
    }
    ```
  }];
}

def ConventionAnnotation : SingleTargetAnnotation {
  let className = "circt.ConventionAnnotation";
  let members = (ins
    Doc<EnumParam<["scalarized"]>, "Port convention type">:$convention
  );
  let description = [{
    Specify the port convention for a module. The port convention controls how a
    module's ports are transformed, and how that module can be instantiated, in
    the output format.

    The options are:
    - `scalarized`: Convert aggregate ports (i.e. vector or bundles) into
      multiple ground-typed ports.

    Example:
    ```json
    {
      "class": "circt.ConventionAnnotation",
      "convention": "scalarized",
      "target": "~Foo|Bar/d:Baz"
    }
    ```
  }];
  let customHandler = "applyConventionAnno";
}

def BodyTypeLoweringAnnotation : SingleTargetAnnotation {
  let className = "circt.BodyTypeLoweringAnnotation";
  let members = (ins
    Doc<EnumParam<["scalarized", "internal"]>, "Type lowering convention">:$convention,
    Doc<BooleanParam, "Apply the convention to all modules in the hierarchy">:$includeHierarchy
  );
  let description = [{
    Specify the type lowering option for module internal signals. This is similar
    to the `Convention` annotation, but for internal signals rather than module
    ports.

    When `includeHierarchy` is `false`, it indicates the convention is applied
    only to the specified module. If `includeHierarchy` is `true`, the convention
    is applied to all modules in the hierarchy. If there are multiple annotation
    instances that specify conventions, the `scalarized` convention takes
    precedence over the `internal` convention.
    
    Example:
    ```json
    {
      "class": "circt.BodyTypeLoweringAnnotation",
      "convention": "scalarized",
      "target": "~Foo|Bar",
      "includeHierarchy": true
    }
    ```
  }];
  let customHandler = "applyBodyTypeLoweringAnno";
}

//===----------------------------------------------------------------------===//
// SiFive Enterprise Annotations
//===----------------------------------------------------------------------===//

def AddSeqMemPortAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation";
  let members = (ins
    Doc<StringParam, "The name of the port to insert">:$name,
    Doc<BooleanParam, "If true this is an input port, otherwise it is an output port">:$input,
    Doc<IntegerParam, "The width of the port">:$width
  );
  let description = [{
    This annotation causes an extra port to be added to all SRAM modules in the
    DUT. The extra port is a regular module port of unsigned integer type with
    the specified width. These extra ports are commonly used to implement SRAM
    features not represented by the FIRRTL memory op, such as MBIST. The added
    port will be wired to the DUT, where it will be tied to 0.

    Example:
    ```json
    {
      "class":"sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
      "name":"user_outputs",
      "input":false,
      "width":1
    }
    ```
  }];
}

def AddSeqMemPortsFileAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation";
  let members = (ins
    Doc<StringParam, "The filename to output to">:$filename
  );
  let description = [{
    This annotation is used to emit metadata about the extra ports created by
    `AddSeqMemPortAnnotation`. This file is emitted relative to the
    `MetadataDirAnnotation`. The file lists each SRAM and provides the mapping
    to where it is in the hierarchy, and gives its IO prefix at the DUT top
    level.

    Example:
    ```json
    {
      "class":"sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
      "filename":"SRAMPorts.txt"
    }
    ```
  }];
}

def VerbatimBlackBoxAnno : SingleTargetAnnotation {
  let className = "circt.VerbatimBlackBoxAnno";
  let targets = [ExtModule];
  let members = (ins
    Doc<ArrayParam<StringParam>, "Array of file paths">:$files
  );
  let description = [{
    Specifies verbatim black box source code with one or more files. Each file
    object in the `files` array contains:
    - `content`: The literal source code content
    - `output_file`: Path to the output file

    This annotation is used internally by CIRCT to represent partially lowered
    FIRRTL extmodules with verbatim content.

    Example:
    ```json
    {
      "class": "circt.VerbatimBlackBoxAnno",
      "target": "~Foo|MyBlackBox",
      "files": [
        {
          "content": "module MyBlackBox(\n  input clk,\n  output out\n);\n  assign out = clk;\nendmodule",
          "output_file": "blackbox.v"
        }
      ]
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Deprecated Annotations
//===----------------------------------------------------------------------===//

def FullAsyncResetAnnotation : SingleTargetAnnotation {
  let className = "sifive.enterprise.firrtl.FullAsyncResetAnnotation";
  let description = [{
    The target must be a signal that is or is inferred to be an asynchronous reset.

    Indicates that all reset-less registers which are children of the module
    containing the target will have the asynchronous reset targeted attached,
    with a reset value of 0.

    The module containing the target of this annotation is not allowed to reside
    in multiple hierarchies.

    Example:
    ```json
    {
      "class": "circt.FullResetAnnotation",
      "target": "~Foo|Bar/d:Baz>reset",
      "resetType": "async"
    }
    ```
  }];
  let customHandler = "convertToFullResetAnnotation";
}

def IgnoreFullAsyncResetAnnotation : SingleTargetAnnotation {
  let className = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation";
  let description = [{
    This annotation indicates that the target should be excluded from the
    FullAsyncResetAnnotation of a parent module.

    Example:
    ```json
    {
      "class": "circt.IgnoreFullAsyncResetAnnotation",
      "target": "~Foo|Bar/d:Baz"
    }
    ```
  }];
  let customHandler = "convertToExcludeFromFullResetAnnotation";
}

//===----------------------------------------------------------------------===//
// DUT and Hierarchy Annotations
//===----------------------------------------------------------------------===//

def MarkDUTAnnotation : SingleTargetAnnotation {
  let className = "sifive.enterprise.firrtl.MarkDUTAnnotation";
  let targets = [Module];
  let description = [{
    This annotation is used to mark the top module of the device under test. This
    can be used to distinguish modules in the test harness from modules in the DUT.

    Example:
    ```json
    {
      "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
      "target":"Core.Core"
    }
    ```
  }];
  let customHandler = "applyDUTAnno";
}

def InjectDUTHierarchyAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation";
  let members = (ins
    Doc<StringParam, "The name of the module containing original DUT logic">:$name,
    Doc<Optional<BooleanParam, "false">, "If true, then treat the newly created module as the DUT">:$moveDut
  );
  let description = [{
    This annotation can be used to add an extra level of hierarchy in the design
    under the DUT (indicated with a `MarkDUTAnnotation`). All logic in the
    original DUT will be moved into a module with the specified `name`. This is
    typically used in combination with `ExtractBlackBoxAnnotation` to not
    intermix the original DUT contents with extracted module instantiations.

    If the `moveDut` field is true, then the newly created module with the
    specified `name` will be treated as the design-under-test. The
    `MarkDUTAnnotation` will be moved to this module. If this field is false,
    then the design-under-test will not be changed.

    This annotation should only appear zero or once.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
      "name": "Logic",
      "moveDut": false
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Metadata and Output Directory Annotations
//===----------------------------------------------------------------------===//

def MetadataDirAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.MetadataDirAnnotation";
  let members = (ins
    Doc<StringParam, "The directory to place generated metadata in">:$dirname
  );
  let description = [{
    This annotation is used to define the directory where metadata should be
    emitted. When this annotation is not present, metadata will be emitted to
    the "metadata" directory by default.

    Example:
    ```json
    {
      "class":"sifive.enterprise.firrtl.MetadataDirAnnotation",
      "dirname":"build/metadata"
    }
    ```
  }];
}

def TestBenchDirAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.TestBenchDirAnnotation";
  let members = (ins
    Doc<StringParam, "The output directory">:$dirname
  );
  let description = [{
    This annotation is used to indicate where to emit the test bench modules
    generated by GrandCentral.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
      "dirname": "output/testbench"
    }
    ```
  }];
}

def ModuleHierarchyAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.ModuleHierarchyAnnotation";
  let members = (ins
    Doc<StringParam, "The full output file path">:$filename
  );
  let description = [{
    This annotation indicates that a module hierarchy JSON file should be emitted
    for the module hierarchy rooted at the design under test (DUT), as indicated
    by the `MarkDUTAnnotation`. See the SV attribute,
    `firrtl.moduleHierarchyFile`, for information about the JSON file format.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.ModuleHierarchyAnnotation",
      "filename": "./dir/hier.json"
    }
    ```
  }];
}

def TestHarnessHierarchyAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation";
  let members = (ins
    Doc<StringParam, "The full output file path">:$filename
  );
  let description = [{
    This annotation indicates that a module hierarchy JSON file should be emitted
    for the module hierarchy rooted at the circuit root module, which is assumed
    to be the test harness. See the SV attribute, `firrtl.moduleHierarchyFile`,
    for information about the JSON file format.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation",
      "filename": "./dir/testharness_hier.json"
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Deduplication Annotations
//===----------------------------------------------------------------------===//

def MustDeduplicateAnnotation : NoTargetAnnotation {
  let className = "firrtl.transforms.MustDeduplicateAnnotation";
  let members = (ins
    Doc<ArrayParam<StringParam>, "A list of module targets which must deduplicate">:$modules
  );
  let description = [{
    This annotation causes the deduplication pass to check that the listed
    modules are deduplicated with each other.

    Example:
    ```json
    {
      "class": "firrtl.transforms.MustDeduplicateAnnotation",
      "modules": ["~Foo|Bar", "~Foo|Baz"]
    }
    ```
  }];
}

def DedupGroupAnnotation : SingleTargetAnnotation {
  let className = "firrtl.transforms.DedupGroupAnnotation";
  let targets = [Module];
  let members = (ins
    Doc<StringParam, "The dedup group that the module belongs to">:$group
  );
  let description = [{
    This annotation assigns the targeted module to a dedup group. Modules that
    belong to a dedup group may only be deduplicated with modules that are part
    of the same group.

    Example:
    ```json
    {
      "class": "firrtl.transforms.DedupGroupAnnotation",
      "target": "~Foo|Bar",
      "group": "group1"
    }
    ```
  }];
  let customHandler = "applyWithoutTarget<false, FModuleOp, FExtModuleOp>";
}

//===----------------------------------------------------------------------===//
// Retiming Annotations
//===----------------------------------------------------------------------===//

def RetimeModuleAnnotation : SingleTargetAnnotation {
  let className = "freechips.rocketchip.util.RetimeModuleAnnotation";
  let targets = [Module];
  let description = [{
    This annotation is used to mark modules which should be retimed, and is
    generally just passed through to other tools.

    Example:
    ```json
    {
      "class": "freechips.rocketchip.util.RetimeModuleAnnotation",
      "target": "~Foo|Bar"
    }
    ```
  }];
  let customHandler = "applyWithoutTarget<false, FModuleOp, FExtModuleOp>";
}

def RetimeModulesAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.RetimeModulesAnnotation";
  let members = (ins
    Doc<StringParam, "The filename with full path where it will be written">:$filename
  );
  let description = [{
    This annotation triggers the creation of a file containing a JSON array
    containing the names of all modules annotated with the
    `RetimeModuleAnnotation`.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.RetimeModulesAnnotation",
      "filename": "retime_modules.json"
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Sitest BlackBox Annotations
//===----------------------------------------------------------------------===//

def SitestBlackBoxAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation";
  let members = (ins
    Doc<StringParam, "The file to write to">:$filename
  );
  let description = [{
    This annotation triggers the creation of a file containing a JSON array of
    the `defname`s of all external modules in the device under test which are
    not imported or inlined blackbox modules. This will only collect modules
    which are instantiated under a module annotated with `MarkDUTAnnotation`.

    If any external modules (including imported and inlined blackboxes) have a
    `SitestBlackBoxLibrariesAnnotation`, the libraries specified in that
    annotation will be included in the output.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
      "filename": "dut_blackboxes.json"
    }
    ```
  }];
}

def SitestTestHarnessBlackBoxAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation";
  let members = (ins
    Doc<StringParam, "The file to write to">:$filename
  );
  let description = [{
    This annotation triggers the creation of a file containing a JSON array of
    the `defname`s of all external modules in the test harness which are not
    imported or inlined blackbox modules. This will only collect modules which
    are not instantiated under a module annotated with `MarkDUTAnnotation`.

    If any external modules (including imported and inlined blackboxes) have a
    `SitestBlackBoxLibrariesAnnotation`, the libraries specified in that
    annotation will also be included in the output.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
      "filename": "testharness_blackboxes.json"
    }
    ```
  }];
}

def SitestBlackBoxLibrariesAnnotation : SingleTargetAnnotation {
  let className = "sifive.enterprise.firrtl.SitestBlackBoxLibrariesAnnotation";
  let targets = [ExtModule];
  let members = (ins
    Doc<ArrayParam<StringParam>, "Array of library names to include in blackbox metadata">:$libraries
  );
  let description = [{
    This annotation is used to specify additional library names that should be
    included in the blackbox metadata for an external module. When applied to an
    external module, the specified libraries will be added to the blackbox
    resource list in the generated metadata. Both the `defname` and any libraries
    specified by this annotation will be included in the metadata for
    non-imported and non-inlined blackboxes.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.SitestBlackBoxLibrariesAnnotation",
      "target": "~Foo|MyExtModule",
      "libraries": ["lib1", "lib2"]
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Instance Extraction Annotations
//===----------------------------------------------------------------------===//

def ExtractBlackBoxAnnotation : SingleTargetAnnotation {
  let className = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation";
  let members = (ins
    Doc<StringParam, "Output file to be filled with the applied hierarchy changes">:$filename,
    Doc<StringParam, "Prefix for the extracted instance">:$prefix,
    Doc<Optional<StringParam>, "Name of an optional wrapper module under which to group extracted instances">:$dest
  );
  let description = [{
    This annotation causes the `ExtractInstances` pass to move the annotated
    instance, or all instances if the annotation is on a module, upwards in the
    hierarchy. If the `dest` field is present and non-empty, the instances are
    placed in a module underneath the DUT (marked by `MarkDUTAnnotation`) with
    the name provided in that field. If the `dest` field is empty, the instances
    are extracted out of the DUT, such that the DUT gains additional ports that
    correspond to the extracted instance ports.

    This allows the DUT to be instantiated and custom implementations for the
    extracted instances to be provided at the instantiation site. Instances are
    never extracted out of the root module of the design.

    Applies to modules and instances.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
      "target": "~Foo|BlackBox",
      "filename": "blackbox_extract.txt",
      "prefix": "bb",
      "dest": "BlackBoxes"
    }
    ```
  }];
}

def ExtractClockGatesFileAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation";
  let members = (ins
    Doc<StringParam, "Output file to be filled with the applied hierarchy changes">:$filename,
    Doc<Optional<StringParam>, "Name of an optional wrapper module under which to group extracted instances">:$group
  );
  let description = [{
    This annotation causes the `ExtractInstances` pass to move instances of
    extmodules whose defnames end in `EICG_wrapper` upwards in the hierarchy,
    either out of the DUT if `group` is omitted or empty, or into a submodule of
    the DUT with the name given in `group`. The wiring prefix is hard-coded to
    `clock_gate`.

    Applies to the circuit.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation",
      "filename": "clock_gates.txt",
      "group": "ClockGates"
    }
    ```
  }];
}

def ExtractSeqMemsFileAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation";
  let members = (ins
    Doc<StringParam, "Output file to be filled with the applied hierarchy changes">:$filename,
    Doc<Optional<StringParam>, "Name of an optional wrapper module under which to group extracted instances">:$group
  );
  let description = [{
    This annotation causes the `ExtractInstances` pass to move memory instances
    upwards in the hierarchy, either out of the DUT if `group` is omitted or
    empty, or into a submodule of the DUT with the name given in `group`. The
    wiring prefix is hard-coded to `mem_wiring`.

    Applies to the circuit.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation",
      "filename": "seq_mems.txt",
      "group": "Memories"
    }
    ```
  }];
}

def OutputDirAnnotation : SingleTargetAnnotation {
  let className = "circt.OutputDirAnnotation";
  let targets = [Module];
  let members = (ins
    Doc<StringParam, "The output directory">:$dirname
  );
  let description = [{
    Specify the output directory for a module. The target must be a public
    module, and must be local.

    Example:
    ```json
    {
      "class": "circt.OutputDirAnnotation",
      "target": "~Foo|Bar",
      "dirname": "output/bar"
    }
    ```
  }];
  let customHandler = "applyOutputDirAnno";
}

//===----------------------------------------------------------------------===//
// Grand Central Annotations
//===----------------------------------------------------------------------===//

// Grand Central Object Type Definitions

def TargetTokenField : NoTargetAnnotation {
  let className = "firrtl.annotations.TargetToken$Field";
  let members = (ins
    Doc<UnionParam<[StringParam, IntegerParam]>, "Field name or index">:$value
  );
  let description = [{
    This is used to represent an index in to an aggregate type, such as an index
    or array.
  }];
}

// ReferenceTarget object - represents a target reference in FIRRTL
// Used in Grand Central annotations to reference circuit components
def ReferenceTarget : ObjectParam<(ins
  Doc<StringParam, "Circuit name">:$circuit,
  Doc<StringParam, "Module name">:$module,
  Doc<ArrayParam<StringParam>, "Path of instances from circuit to module">:$path,
  Doc<StringParam, "Reference name">:$ref,
  Doc<ArrayParam<TargetTokenField>, "Component path (field/index selections)">:$component
)> {
  let description = [{
    Represents a reference to a specific component in the FIRRTL circuit
    hierarchy. Used in Grand Central annotations to specify targets for
    interface connections.
  }];
}

// AugmentedType - Placeholder object representing any Grand Central augmented type.
// This is used in recursive type definitions where the actual type is one of:
// - AugmentedGroundType: A leaf type representing a hardware signal
// - AugmentedVectorType: An array of augmented types
// - AugmentedBundleType: A bundle/struct of augmented fields
// The actual type is determined by the "class" field in the JSON.
def AugmentedType : ObjectParam<> {
  let description = [{
    Represents any Grand Central augmented type (AugmentedGroundType,
    AugmentedVectorType, or AugmentedBundleType). The actual type is
    determined by the "class" field in the JSON object.
  }];
}

def AugmentedGroundType : NoTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.AugmentedGroundType";
  let members = (ins
    Doc<ReferenceTarget, "ReferenceTarget of the target component">:$ref
  );
  let description = [{
    Creates a SystemVerilog logic type for Grand Central views.

    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
      "ref": { ... }
    }
    ```
  }];
}

def AugmentedVectorType : NoTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.AugmentedVectorType";
  let members = (ins
    Doc<ArrayParam<AugmentedType>, "List of augmented types">:$elements
  );
  let description = [{
    Creates a SystemVerilog unpacked array for Grand Central views.


    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.AugmentedVectorType",
      "elements": [
        {"class": "sifive.enterprise.grandcentral.AugmentedGroundType", "ref": {...}},
        {"class": "sifive.enterprise.grandcentral.AugmentedGroundType", "ref": {...}}
      ]
    }
    ```
  }];
}

// AugmentedField object - represents a field in an AugmentedBundleType
// The tpe field uses AugmentedType to represent any augmented type.
def AugmentedField : ObjectParam<(ins
  Doc<StringParam, "Field name">:$name,
  Doc<Optional<StringParam>, "Field description">:$description,
  Doc<AugmentedType, "Field type">:$tpe
)> {
  let description = [{
    A field in an augmented bundle type. This can provide a small description of what the field in the bundle is.
  }];
}

def AugmentedBundleType : NoTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.AugmentedBundleType";
  let members = (ins
    Doc<StringParam, "The name of the SystemVerilog interface">:$defName,
    Doc<ArrayParam<AugmentedField>, "List of AugmentedFields">:$elements
  );
  let description = [{
    Creates a SystemVerilog interface for each bundle type in Grand Central views.

    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
      "defName": "MyInterface",
      "elements": [
        {
          "name": "field1",
          "description": "First field",
          "tpe": {"class": "sifive.enterprise.grandcentral.AugmentedGroundType", "ref": {...}}
        }
      ]
    }
    ```
  }];
}

def ExtractGrandCentralAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation";
  let members = (ins
    Doc<StringParam, "Directory where Grand Central outputs go, except a bindfile">:$directory,
    Doc<Optional<StringParam>, "Optional filename with full path where the bindfile will be written">:$filename
  );
  let description = [{
    This annotation controls where to "extract" Grand Central collateral from
    the circuit. This annotation is mandatory and can only appear once if the
    full Grand Central transform pipeline is run.

    The directory member has no effect on the filename member, i.e., the
    directory will not be prepended to the filename.

    The filename is optional. When specified, all binds will be emitted to that
    file. When omitted, each module will have an associated bindfile in
    `directory` named `<module>-bind.sv`.

    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
      "directory": "gct-dir",
      "filename": "bindings.sv"
    }
    ```
  }];
}

def GrandCentralHierarchyFileAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation";
  let members = (ins
    Doc<StringParam, "A filename where a YAML representation of the interface should be placed">:$filename
  );
  let description = [{
    This annotation, if present, will emit a YAML representation of all
    interfaces that were generated by the Grand Central views pass.
    Equivalently, this is a different serialization of the information
    contained in all `ViewAnnotation`s.

    Example:
    ```json
    {
      "class" : "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
      "filename" : "directory/file.yaml"
    }
    ```
    The format of the produced YAML file is a one-to-one mapping of the
    SystemVerilog interface to YAML. Consider the following SystemVerilog
    interface produced by GrandCentral:
    
    ```verilog
    interface Foo;
      // A 4-bit type
      logic [3:0] a;
      // A 2D vector of an 8-bit type
      logic [7:0] b [1:0][0:0];
      // A 1D vector of instances of Bar
      Bar bar[4]();
    endinterface
    
    interface Bar;
      logic c;
    endinterface
    
    interface Baz;
      logic d;
    endinterface
    ```
    
    This will produce the following YAML representation:
    
    ```yaml
    - name: Foo
      fields:
        - name: a
          description: A 4-bit type
          dimensions: [  ]
          width: 4
        - name: b
          description: A 2D vector of an 8-bit type
          dimensions: [ 1, 2 ]
          width: 8
      instances:
        - name: bar
          description: A 1D vector of instances of Bar
          dimensions: [ 4 ]
          interface:
            name: Bar
            fields:
              - name: c
                dimensions: [ ]
                width: 1
            instances: []
    - name: Baz:
      fields:
        - name: d
          dimensions: [ ]
          width: 1
      instances: []
    ```
  }];
}

def SerializedViewAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation";
  let members = (ins
    Doc<StringParam, "Name of the view, no affect on output">:$name,
    Doc<StringParam, "Module target of an empty module to insert cross module references in to">:$companion,
    Doc<StringParam, "Module target of the module the interface will be referencing">:$parent,
    Doc<AugmentedBundleType, "AugmentedBundleType representing the interface">:$view
  );
  let description = [{
    These annotations are used to represent a SystemVerilog interface, a location
    in which it should be instantiated, and XMRs to drive the interface. Any XMR
    sources receive `DontTouchAnnotation` to prevent these from being
    inadvertently deleted.

    Either `ViewAnnotation` or `GrandCentralView$SerializedViewAnnotation` are
    the same in CIRCT. The latter has its "view" value serialized (again) to
    JSON and string-escaped. When CIRCT sees any JSON string it tries to
    recursively deserialize it. If this fails, this is deemed to be a string.
    If this succeeds, then the JSON is unpacked.

    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
      "name": "MyView",
      "companion": "~Foo|Companion",
      "parent": "~Foo|Parent",
      "view": {...}
    }
    ```
  }];
  let customHandler = "applyGCTView";
}

def ViewAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.ViewAnnotation";
  let members = (ins
    Doc<StringParam, "Name of the view">:$name,
    Doc<StringParam, "Module target of companion module">:$companion,
    Doc<StringParam, "Module target of parent module">:$parent,
    Doc<AugmentedBundleType, "View definition">:$view
  );
  let description = [{
    Annotation for Grand Central views. Similar to SerializedViewAnnotation but
    with the view field not serialized to JSON.

    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.ViewAnnotation",
      "name": "MyView",
      "companion": "~Foo|Companion",
      "parent": "~Foo|Parent",
      "view": {...}
    }
    ```
  }];
  let customHandler = "applyGCTView";
}

def MemTapAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.MemTapAnnotation";
  let members = (ins
    Doc<ArrayParam<StringParam>, "List of memory tap paths">:$keys
  );
  let description = [{
    Annotation for memory taps in Grand Central. This annotation has custom
    resolution logic that resolves 'source' and 'sink' fields internally,
    so it does not use standard target resolution.

    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.MemTapAnnotation",
      "keys": ["mem_tap_1", "mem_tap_2"]
    }
    ```
  }];
  let customHandler = "applyGCTMemTaps";
}

def CompanionAnnotation : SingleTargetAnnotation {
  let className = "sifive.enterprise.grandcentral.ViewAnnotation.companion";
  let targets = [Module];
  let customHandler = "applyWithoutTarget<>";
  let description = [{
    Companion annotation for Grand Central views. Not in SFC (Scala FIRRTL Compiler).
    This annotation is created by applyGCTView and has a 'target' field that points
    to the companion module. It is applied to modules and processed by the GrandCentral pass.

    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.ViewAnnotation.companion",
      "target": "~Foo|Companion"
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Chisel Annotations
//===----------------------------------------------------------------------===//

def ForceNameAnnotation : SingleTargetAnnotation {
  let className = "chisel3.util.experimental.ForceNameAnnotation";
  let allowNonLocal = 1;
  let targets = [Module, ExtModule];
  let members = (ins
    Doc<StringParam, "The name to force">:$name
  );
  let description = [{
    Forces a specific name for the annotated component, preventing name changes
    during compilation.

    Example:
    ```json
    {
      "class": "chisel3.util.experimental.ForceNameAnnotation",
      "target": "~Foo|Bar",
      "name": "ForcedName"
    }
    ```
  }];
}

def DecodeTableAnnotation : IgnoredAnnotation {
  let className = "chisel3.util.experimental.decode.DecodeTableAnnotation";
  let members = (ins
    Doc<StringParam, "Decode table specification">:$table
  );
  let description = [{
    Annotation for decode table specifications in Chisel.

    Example:
    ```json
    {
      "class": "chisel3.util.experimental.decode.DecodeTableAnnotation",
      "table": "..."
    }
    ```
  }];
}

def TraceNameAnnotation : SingleTargetAnnotation {
  let className = "chisel3.experimental.Trace$TraceNameAnnotation";
  let members = (ins
    Doc<StringParam, "Trace name">:$name
  );
  let description = [{
    Specifies a trace name for the annotated signal.

    Example:
    ```json
    {
      "class": "chisel3.experimental.Trace$TraceNameAnnotation",
      "target": "~Foo|Bar>signal",
      "name": "trace_signal"
    }
    ```
  }];
  let customHandler = "applyTraceName";
}

def TraceAnnotation : SingleTargetAnnotation {
  let className = "chisel3.experimental.Trace$TraceAnnotation";
  let allowNonLocal = 1;
  let members = (ins
    Doc<StringParam, "Chisel type of the signal">:$chiselType,
    Doc<StringParam, "Trace name">:$target
  );
  let description = [{
    Marks a signal for tracing with its Chisel type information.

    Example:
    ```json
    {
      "class": "chisel3.experimental.Trace$TraceAnnotation",
      "target": "~Foo|Bar>signal",
      "chiselType": "UInt<8>",
      "target": "trace_signal"
    }
    ```
  }];
}

def ModulePrefixAnnotation : SingleTargetAnnotation {
  let className = "chisel3.ModulePrefixAnnotation";
  let targets = [Module];
  let members = (ins
    Doc<StringParam, "Prefix to add to module name">:$prefix
  );
  let description = [{
    Adds a prefix to the module name during compilation.

    Example:
    ```json
    {
      "class": "chisel3.ModulePrefixAnnotation",
      "target": "~Foo|Bar",
      "prefix": "MyPrefix_"
    }
    ```
  }];
  let customHandler = "applyModulePrefixAnno";
}

//===----------------------------------------------------------------------===//
// Additional FIRRTL Annotations
//===----------------------------------------------------------------------===//

def NoDedupAnnotation : SingleTargetAnnotation {
  let className = "firrtl.transforms.NoDedupAnnotation";
  let targets = [Module, ExtModule];
  let description = [{
    Prevents deduplication of the annotated module. Both regular modules and
    external modules can be deduplicated, so this annotation can target either.

    Example:
    ```json
    {
      "class": "firrtl.transforms.NoDedupAnnotation",
      "target": "~Foo|Bar"
    }
    ```
  }];
}

def RunFIRRTLTransformAnnotation : IgnoredAnnotation {
  let className = "firrtl.stage.RunFirrtlTransformAnnotation";
  let members = (ins
    Doc<StringParam, "Transform class name">:$transformClass
  );
  let description = [{
    Specifies a FIRRTL transform to run during compilation. This annotation
    is recognized but ignored by CIRCT.

    Example:
    ```json
    {
      "class": "firrtl.stage.RunFirrtlTransformAnnotation",
      "transformClass": "firrtl.transforms.MyCustomTransform"
    }
    ```
  }];
}

def LoadMemoryAnnotation : SingleTargetAnnotation {
  let className = "firrtl.annotations.LoadMemoryAnnotation";
  let targets = [Mem, CombMem, SeqMem];
  let members = (ins
    Doc<StringParam, "File path to load memory contents from">:$fileName,
    Doc<StringParam, "Hex or binary format">:$hexOrBinary
  );
  let description = [{
    Specifies a file to load memory contents from during simulation.

    Example:
    ```json
    {
      "class": "firrtl.annotations.LoadMemoryAnnotation",
      "target": "~Foo|Bar>mem",
      "fileName": "mem_init.hex",
      "hexOrBinary": "h"
    }
    ```
  }];
  let customHandler = "applyLoadMemoryAnno<false>";
}

def MemoryFileInlineAnnotation : SingleTargetAnnotation {
  let className = "firrtl.annotations.MemoryFileInlineAnnotation";
  let targets = [Mem, CombMem, SeqMem];
  let members = (ins
    Doc<StringParam, "File path for inline memory">:$filename,
    Doc<StringParam, "Hex or binary format">:$hexOrBinary
  );
  let description = [{
    Specifies inline memory file loading for simulation.

    Example:
    ```json
    {
      "class": "firrtl.annotations.MemoryFileInlineAnnotation",
      "target": "~Foo|Bar>mem",
      "filename": "mem_inline.hex",
      "hexOrBinary": "h"
    }
    ```
  }];
  let customHandler = "applyLoadMemoryAnno<true>";
}

def WiringSinkAnnotation : SingleTargetAnnotation {
  let className = "firrtl.passes.wiring.SinkAnnotation";
  let members = (ins
    Doc<StringParam, "Pin name">:$pin
  );
  let description = [{
    Marks a component as a wiring sink for the WiringTransform.

    Example:
    ```json
    {
      "class": "firrtl.passes.wiring.SinkAnnotation",
      "target": "~Foo|Bar>sink",
      "pin": "myPin"
    }
    ```
  }];
  let customHandler = "applyWiring";
}

def WiringSourceAnnotation : SingleTargetAnnotation {
  let className = "firrtl.passes.wiring.SourceAnnotation";
  let members = (ins
    Doc<StringParam, "Pin name">:$pin
  );
  let description = [{
    Marks a component as a wiring source for the WiringTransform.

    Example:
    ```json
    {
      "class": "firrtl.passes.wiring.SourceAnnotation",
      "target": "~Foo|Bar>source",
      "pin": "myPin"
    }
    ```
  }];
  let customHandler = "applyWiring";
}

//===----------------------------------------------------------------------===//
// Additional SiFive Enterprise Annotations
//===----------------------------------------------------------------------===//

def ExtractAssertionsAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.ExtractAssertionsAnnotation";
  let members = (ins
    Doc<StringParam, "Output directory for extracted assertions">:$directory
  );
  let description = [{
    Triggers extraction of assertions to a separate file.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.ExtractAssertionsAnnotation",
      "directory": "assertions"
    }
    ```
  }];
}

def ExtractAssumptionsAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.ExtractAssumptionsAnnotation";
  let members = (ins
    Doc<StringParam, "Output directory for extracted assumptions">:$directory
  );
  let description = [{
    Triggers extraction of assumptions to a separate file.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.ExtractAssumptionsAnnotation",
      "directory": "assumptions"
    }
    ```
  }];
}

def ExtractCoverageAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.ExtractCoverageAnnotation";
  let members = (ins
    Doc<StringParam, "Output directory for extracted coverage">:$directory
  );
  let description = [{
    Triggers extraction of coverage statements to a separate file.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.ExtractCoverageAnnotation",
      "directory": "coverage"
    }
    ```
  }];
}

def TestHarnessPathAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.TestHarnessPathAnnotation";
  let members = (ins
    Doc<StringParam, "Path to test harness">:$path
  );
  let description = [{
    Specifies the path to the test harness module.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.TestHarnessPathAnnotation",
      "path": "TestHarness.v"
    }
    ```
  }];
}

def ConvertMemToRegOfVecAnnotation : NoTargetAnnotation {
  let className = "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$";
  let description = [{
    Circuit-level annotation that converts all memories in the design to register vectors.
    This annotation does not target specific memories - it applies to the entire circuit.

    Example:
    ```json
    {
      "class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RocketChip Annotations
//===----------------------------------------------------------------------===//

def InternalVerifBlackBoxAnnotation : SingleTargetAnnotation {
  let className = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation";
  let targets = [ExtModule];
  let description = [{
    Marks an external module as an internal verification black box for RocketChip.
    When instantiated inside the DUT, the module is marked for coverage extraction.

    Example:
    ```json
    {
      "class": "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation",
      "target": "~Foo|VerifBlackBox"
    }
    ```
  }];
}
