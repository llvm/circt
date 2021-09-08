# FIRRTL Annotations

The Scala FIRRTL Compiler (SFC) provides a mechanism to encode arbitrary
metadata and associate it with zero or more "things" in a FIRRTL circuit.  This
mechanism is an _Annotation_ and the association is described using one or more
_Targets_.  Annotations should be viewed an extension to the FIRRTL IR
specification, and can greatly affect the meaning and interpretation of the IR.

Annotations are represented as a dictionary, with a "class" field which
describes which annotation it is, and a "target" field which represents the IR
object it is attached to. The annotation's class matches the name of a Java
class in the Scala Chisel/FIRRTL code base. Annotations may have arbitrary
additional fields attached. Some annotation classes extend other annotations,
which effectively means that the subclass annotation implies to effect of the
parent annotation.

Annotations are serializable to JSON and either live in a separate file (e.g.,
during the handoff between Chisel and the SFC) or are stored in-memory (e.g.,
during SFC-based compilation).  The SFC pass API requires that passes describe
which targets in the circuit they update.  SFC infrastructure then
automatically updates annotations so they are always synchronized with their
corresponding FIRRTL IR.

An example of an annotation is the `DontTouchAnnotation`, which can be used to
indicate to the compiler that a wire "foo" should not be optimized away.

```json
{
  "class":"firrtl.transforms.DontTouchAnnotation",
  "target""~MyCircuit|MyModule>foo"
}
```

Some annotations have more complex interactions with the IR. For example the
[BoringUtils](https://www.chisel-lang.org/api/latest/chisel3/util/experimental/BoringUtils$.html)
provides FIRRTL with annotations which can be used to wire together any two
things across the module instance hierarchy.

## Motivation

Historically, annotations grew out of three choices in the design of FIRRTL IR:

1) FIRRTL IR is not extensible with user-defined IR nodes.
2) FIRRTL IR is not parameterized.
3) FIRRTL IR does not support in-IR attributes.

Annotations have then been used for all manner of extensions including:

1) Encoding SystemVerilog nodes into the IR using special printfs, an example of
   working around (1) above.
2) Setting the reset vector of different, identical CPU cores, an example of
   working around (2) above.
3) Encoding sources and sinks that should be wired together by an SFC pass, an
   example of (3) above.

## Targets

A circuit is described, stored, and optimized in a folded representation. For
example, there may be multiple instances of a module which will eventually
become multiple physical copies of that module on the die. 

Targets are a mechanism to identify specific hardware in specific instances of
modules in a FIRRTL circuit.  A target consists of a circuit, a root module, an
optional instance hierarchy, and an optional reference. A target can only
identify hardware with a name, e.g., a circuit, module, instance, register,
wire, or node. References may further refer to specific fields or subindices in
aggregates. A target with no instance hierarchy is local. A target with an
instance hierarchy is non-local.

Targets use a shorthand syntax of the form:
```
target ::= “~” (circuit) (“|” (module) (“/” (instance) “:” (module) )* (“>” (ref) )?)?
```

A reference is a name inside a module and one or more qualifying tokens that
encode subfields (of a bundle) or subindices (of a vector):
```
reference ::= (name) ("[" (index) "]" | "." (field))*
```

Targets are specific enough to refer to any specific module in a folded,
unfolded, or partially folded representation. 

To show some examples of what these look like, consider the following example
circuit. This consists of four instances of module `Baz`, two instances of
module `Bar`, and one instance of module `Foo`:

```firrtl
circuit Foo:
  module Foo:
    inst a of Bar
    inst b of Bar
  module Bar:
    inst c of Baz
    inst d of Baz
  module Baz:
    skip
```

| Folded Module   | Unfolded Modules  |
| --------------- | ----------------- | 
| <img title="Folded Modules" src="includes/img/firrtl-folded-module.png"/> | <img title="Unfolded Modules" src="includes/img/firrtl-unfolded-module.png"/> |

Using targets (or multiple targets), any specific module, instance, or
combination of instances can be expressed. Some examples include:

| Target                                 | Description                                                |
| --------                               | -------------                                              |
| <code>~Foo</code>                      | refers to the whole circuit                                |
| <code>~Foo&#124;Foo</code>             | refers to the top module                                   |
| <code>~Foo&#124;Bar</code>             | refers to module `Bar` (or both instances of module `Bar`) |
| <code>~Foo&#124;Foo/a:Bar</code>       | refers just to one instance of module `Bar`                |
| <code>~Foo&#124;Foo/b:Bar/c:Baz</code> | refers to one instance of module `Baz`                     |
| <code>~Foo&#124;Bar/d:Baz</code>       | refers to two instances of module `Baz`                    |

If a target does not contain an instance path, it is a _local_ target.  A local
target points to all instances of a module.  If a target contains an instance
path, it is a _non-local_ target.  A non-local target _may_ not point to all
instances of a module.  Additionally, a non-local target may have an equivalent
local target representation.

## Inline Annotations

The MLIR FIRRTL compiler supports an inline format for annotations as an
extension to the FIRRTL syntax. These inline annotations are helpful for making
single-file annotated FIRRTL code. This is not supported by the Scala FIRRTL
compiler.  

Inline annotations are attached to the `circuit`, and are JSON wrapped in `%[`
and `]`.

```firrtl
circuit Foo: %[[{"a":"a","target":"~Foo"}]]
  module Foo:
    skip
```

## Annotations in CIRCT

We plan to provide full support for annotations in CIRCT.  The FIRRTL dialect
current supports:

1) All non-local annotations can be parsed and applied to the correct circuit
   component.
2) Annotations, with and without references, are copied to the correct ground
   type in the `LowerTypes` pass.

Annotations can be parsed using the `--annotation-file` command line argument
to the `firtool` utility.  Alternatively, we provide a non-standard way of
encoding annotations in the FIRRTL IR textual representation.  We provide this
non-standard support primarily to make test writing easier.  As an example of
this, consider the following JSON annotation file:

```json
[
  {
    "target": "~Foo|Foo",
    "hello": "world"
  }
]
```

This can be equivalently, in CIRCT, expressed as:

```firrtl
circuit Foo: %[[{"target":"~Foo|Foo","hello":"world"}]]
  module Foo:
    skip
```

During parsing, annotations are "scattered" into the MLIR representation as
operation or port attributes.  As an example of this, the above parses into the
following MLIR representation:

```mlir
firrtl.circuit "Foo"  {
  firrtl.module @Foo() attributes {annotations = [{hello = "world"}]} {
    firrtl.skip
  }
}
```

Targets without references have their targets stripped during scattering since
target information is redundant once annotation metadata is attached to the IR.
Targets with references have the reference portion of the target included in
the attribute.  The `LowerTypes` pass then uses this reference information to
attach annotation metadata to only the _lowered_ portion of a targeted circuit
component.

Annotations are expected to be fully removed via custom transforms, conversion
to other MLIR operations, or dropped. A warning will be emitted if there are
any unused annotations still in the circuit. For example, the `ModuleInliner`
pass removes `firrtl.passes.InlineAnnotation` by inlining annotated modules or
instances. JSON Annotations map to the builtin MLIR attributes. An annotation
is implemented using a DictionaryAttr, which holds the class, target, any
annotation specific data. 

## Annotations

Annotations here are written in their JSON format. A "reference target"
indicates that the annotation could target anything object in the hierarchy,
although there may be further restrictions in the annotation.

### [BlackBoxInlineAnno](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/BlackBoxInlineAnno.html)

| Property   | Type   | Description                            |
| ---------- | ------ | -------------                          |
| class      | string | `firrtl.transforms.BlackBoxInlineAnno` |
| target     | string | An ExtModule name target               |
| name       | string | A full path to a file                  |
| text       | string | Literal verilog code.                  |

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

### [BlackBoxPathAnno](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/BlackBoxPathAnno.html)

| Property   | Type   | Description                          |
| ---------- | ------ | -------------                        |
| class      | string | `firrtl.transforms.BlackBoxPathAnno` |
| target     | string | An ExtModule name target             |
| path       | string | ModuleName target                    |

Specifies the file `path` as source code for the module. Copies the file
to the target directory.

Example:
```json
{
  "class": "firrtl.transforms.BlackBoxPathAnno",
  "target": "~Foo|Foo",
  "path": "myfile.v"
}
```

### [BlackBoxResourceAnno](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/BlackBoxResourceAnno.html)

| Property   | Type   | Description                              |
| ---------- | ------ | -------------                            |
| class      | string | `firrtl.transforms.BlackBoxResourceAnno` |
| target     | string | An ExtModule name target                 |
| path       | string | ModuleName target                        |

Specifies the file `path` as source code for the module. In contrast to
the `BlackBoxPathAnno`, the file is searched for in the black box resource
search path. This is a remnant of the Scala origins of FIRRTL. Copies the
file to the target directory.

Example:
```json
{
  "class": "firrtl.transforms.BlackBoxResourceAnno",
  "target": "~Foo|Foo",
  "resourceId": "myfile.v"
}
```

### [BlackBoxResourceFileNameAnno](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/BlackBoxResourceFileNameAnno.html)

| Property         | Type   | Description                              |
| ----------       | ------ | -------------                            |
| class            | string | `firrtl.transforms.BlackBoxFileNameAnno` |
| resourceFileName | string | Output filename                          |

Specifies the output file name for the list of black box source files that
is generated as a collateral of the pass.

Example:
```json
{
  "class": "firrtl.transforms.BlackBoxResourceFileNameAnno",
  "resourceFileName": "FileList.f"
}
```

### [BlackBoxTargetDirAnno](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/BlackBoxTargetDirAnno.html)

| Property   | Type   | Description                               |
| ---------- | ------ | -------------                             |
| class      | string | `firrtl.transforms.BlackBoxTargetDirAnno` |
| targetDir  | string | Output directory                          |

Overrides the target directory into which black box source files are
emitted.

Example:
```json
{
  "class": "firrtl.transforms.BlackBoxTargetDirAnno",
  "targetDir": "/tmp/circt/output"
}
```

### [DontTouchAnnotation](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/DontTouchAnnotation.html)

| Property   | Type   | Description                             |
| ---------- | ------ | -------------                           |
| class      | string | `firrte.transforms.DontTouchAnnotation` |
| target     | string | Reference target                        |

The `DontTouchAnnotation` prevents the removal of elements through
optimization. This annotation is an optimization barrier, for 
example, it blocks constant propagation through it.
This annotation also ensures that the name of the object is
preserved, and not discarded or modified. 

Example:
```json
{
  "class": "firrtl.transforms.DontTouchAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### [FlattenAnnotation](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/FlattenAnnotation.html)

| Property   | Type   | Description                           |
| ---------- | ------ | -------------                         |
| class      | string | `firrtl.transforms.FlattenAnnotation` |
| target     | string | Reference target                      |

Indicates that the target should be flattened, which means that child instances
will be recursively inlined.

Example:
```json
{
  "class": "firrtl.transforms.FlattenAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### FullAsyncResetAnnotation

| Property   | Type   | Description                                         |
| ---------- | ------ | -------------                                       |
| class      | string | `sifive.enterprise.firrtl.FullAsyncResetAnnotation` |
| target     | string | Reference target                                    |

Indicates that all reset-less registers which are children of the target will
have an asynchronous reset attached, with a reset value of 0.

A module targeted by this annotation is not allowed to reside in multiple
hierarchies.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.FullAsyncResetAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### IgnoreFullAsyncResetAnnotation

| Property   | Type   | Description                                               |
| ---------- | ------ | -------------                                             |
| class      | string | `sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation` |
| target     | string | Reference target                                          |

This annotation indicates that the target should be excluded from the
FullAsyncResetAnnotation of a parent module.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### [InlineAnnotation](https://www.chisel-lang.org/api/firrtl/latest/firrtl/passes/InlineAnnotation.html)

| Property   | Type   | Description                      |
| ---------- | ------ | -------------                    |
| class      | string | `firrtl.passes.InlineAnnotation` |
| target     | string | Reference target                 |

Indicates that the target should be inlined.

Example:
```json
{
  "class": "firrtl.passes.InlineAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### NestedPrefixModulesAnnotation

| Property   | Type   | Description                                              |
| ---------- | ------ | -------------                                            |
| class      | string | `sifive.enterprise.firrtl.NestedPrefixModulesAnnotation` |
| prefix     | string | Prefix to use                                            |
| inclusive  | bool   | Whether this prefix is inclusive of the target           |

Prefixes all module names under the target with the required prefix.  If
`inclusive` is true, it includes the target module in the renaming.  If
`inclusive` is false, it will only rename modules instantiated underneath the
target module.  If a module is required to have two different prefixes, it will
be cloned.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
  "prefix": "MyPrefix_",
  "inclusive": true
}
```

##  Grand Central

Grand Central provides annotations for creating cross module references and
SystemVerilog interfaces.

### Views

Grand Central views are used from Chisel to allow users to encapsulate monitor
logic that gets emitted separately from the DUT. The generated interfaces
provide a stable view of modules which are connected to the target module
through SystemVerilog bind statements.

#### TargetToken$Field

| Property   | Type              | Description                            |
| ---------- | ------            | -------------                          |
| class      | string            | `firrtl.annotations.TargetToken$Field` |
| value      | string or integer | Index or element name                  |

This is used to represent an index in to an aggregate type, such as an index or
array.

#### ReferenceTarget

| Property   | Type   | Description                                              |
| ---------- | ------ | -------------                                            |
| circuit    | string | Name of the encapsulating circuit                        |
| module     | string | Name of the root module of this reference                |
| path       | array  | Path through instance and Modules                        |
| ref        | string | Name of the component                                    |
| component  | array  | List of TargetToken$Field subcomponent of this reference |

A reference target is a JSON serialization of a regular reference target
string.

#### UnknownGroundType

| Property   | Type   | Description                                                          |
| ---------- | ------ | -------------                                                        |
| class      | string | `sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$` |

This represents an unknown FIRRTL ground type.

#### AugmentedGroundType

| Property   | Type   | Description                                          |
| ---------- | ------ | -------------                                        |
| class      | string | `sifive.enterprise.grandcentral.AugmentedGroundType` |
| ref        | object | ReferenceTarget of the target component              |
| tpe        | object | UnknownGroundType                                    |

Creates a SystemVerilog logic type.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
  "ref": {
    "circuit": "GCTInterface",
    "module": "GCTInterface",
    "path": [],
    "ref": "a",
    "component": [
      {
        "class": "firrtl.annotations.TargetToken$Field",
        "value": "_2"
      },
      {
        "class": "firrtl.annotations.TargetToken$Index",
        "value": 0
      }
    ]
  },
  "tpe": {
    "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
  }
}
```

#### AugmentedVectorType

| Property   | Type   | Description                                          |
| ---------- | ------ | -------------                                        |
| class      | string | `sifive.enterprise.grandcentral.AugmentedVectorType` |
| elements   | array  | List of augmented types.

Creates a SystemVerilog unpacked array.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.AugmentedVectorType",
  "elements": [
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
      "ref": {
        "circuit": "GCTInterface",
        "module": "GCTInterface",
        "path": [],
        "ref": "a",
        "component": []
      },
      "tpe": {
        "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
      }
    },
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
      "ref": {
        "circuit": "GCTInterface",
        "module": "GCTInterface",
        "path": [],
        "ref": "b",
        "component": []
      },
      "tpe": {
        "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
      }
    }
  ]
}
```

#### AugmentedField

| Property    | Type   | Description                        |
| ----------  | ------ | -------------                      |
| name        | string | Name of the field                  |
| description | string | A textual description of this type |
| tpe         | string | A nested augmented type            |

A field in an augmented bundle type.  This can provide a small description of
what the field in the bundle is.

#### AugmentedBundleType

| Property   | Type   | Description                                        |
| ---------- | ------ | -------------                                      |
| class      | string | sifive.enterprise.grandcentral.AugmentedBundleType |
| defName    | string | The name of the SystemVerilog interface            |
| elements   | array  | List of AugmentedFields                            |

Creates a SystemVerilog interface for each bundle type.

#### GrandCentralView$SerializedViewAnnotation

| Property    | Type     | Description                                                              |
| ----------- | -------- | ------------------                                                       |
| class       | string   | sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation |
| name        | string   | Name of the view, no affect on output                                    |
| companion   | string   | Module target of an empty module to insert cross module references in to |
| parent      | string   | Module target of the module the interface will be referencing            |
| view        | object   | AugmentedBundleType representing the interface                           |

Grand Central Interfaces are used to emit SystemVerilog interfaces with stable
names. `SerializedViewAnnotation` implies `DontTouchAnnotation` on any
`AugmentedGroundType.ref` target.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
  "name": "view",
  "companion": "~GCTInterface|view_companion",
  "parent": "~GCTInterface|GCTInterface",
  "view": {
    "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
    "defName": "ViewName",
    "elements": [
      {
        "name": "port",
        "description": "the port 'a' in GCTInterface",
        "tpe": {
          "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
          "ref": {
            "circuit": "GCTInterface",
            "module": "GCTInterface",
            "path": [],
            "ref": "a",
            "component": []
          },
          "tpe": {
            "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
          }
        }
      }
    ]
  }
}
```

#### ExtractGrandCentralAnnotation

| Property  | Type   | Description                                                  |
| ---       | ---    | ---                                                          |
| class     | string | sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation |
| directory | string | Directory where Grand Central outputs go, except a bindfile  |
| filename  | string | Filename with full path where the bindfile will be written   |

This annotation controls where to "extract" Grand Central collateral from the
circuit.  This annotation is mandatory and can only appear once if the full
Grand Central transform pipeline is run in the SFC.  (An error is generated by
the `ExtractGrandCentralCode` transform.)

The directory member has no effect on the filename member, i.e., the directory
will not be prepended to the filename.

### Data Taps

Grand Central Taps are a tool for representing cross module references. They
enable users to "tap" into signal anywhere in the module hierarchy and treat
them as local, read-only signals.

`DataTaps` annotations are used to fill in the body of a FIRRTL external module
with cross-module references to other modules.  Each `DataTapKey` corresponds
to one output port on the `DataTapsAnnotation` external module.

#### ReferenceDataTapKey

| Property    | Type     | Description                                          |
| ----------- | -------- | ------------------                                   |
| class       | string   | `sifive.enterprise.grandcentral.ReferenceDataTapKey` |
| source      | string   | Reference target of the source signal.               |
| portName    | string   | Reference target of the data tap black box port      |

This key allows tapping a target in FIRRTL.

#### DataTapModuleSignalKey

| Property     | Type     | Description                                           |
| -----------  | -------- | ------------------                                    |
| class        | string   | sifive.enterprise.grandcentral.DataTapModuleSignalKey |
| module       | string   | ExtModule name of the target black box                |
| internalPath | string   | The path within the module                            |
| portName     | string   | Reference target of the data tap black box port       |

This key allows tapping a point by name in a blackbox.

#### LiteralDataTapKey

| Property    | Type     | Description                                      |
| ----------- | -------- | ------------------                               |
| class       | string   | sifive.enterprise.grandcentral.LiteralDataTapKey |
| literal     | string   | FIRRTL constant literal                          |
| portName    | string   | Reference target of the data tap black box port  |

This key allows the creation of a FIRRTL literal.

#### DataTapsAnnotation

| Property    | Type     | Description                                                       |
| ----------- | -------- | ------------------                                                |
| class       | string   | sifive.enterprise.grandcentral.DataTapsAnnotation                 |
| blackbox    | string   | ExtModule name of the black box with ports referenced by the keys |
| keys        | array    | List of DataTapKeys                                               |

The `DataTapsAnnotation` is a collection of all the data taps in a circuit.
This will cause a data tap module to be emitted.  The `DataTapsAnnotation`
implies `DontTouchAnnotation` on any `ReferenceDataTapKey.source` target.

Example:

```json
{
  "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
  "blackBox": "~GCTDataTap|DataTap",
  "keys": [
    {
      "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      "source": "~GCTDataTap|GCTDataTap>r",
      "portName": "~GCTDataTap|DataTap>_0"
    },
    {
      "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      "module":"~GCTDataTap|BlackBox",
      "internalPath":"baz.qux",
      "portName":"~GCTDataTap|DataTap>_1"
    },
    {
      "class":"sifive.enterprise.grandcentral.LiteralDataTapKey",
      "literal":"UInt<16>(\"h2a\")",
      "portName":"~GCTDataTap|DataTap>_3"
    }
  ]
}
```

### Memory Taps

Memory taps are a special version of data taps which are used for targeting the
FIRRTL memory vectors.

#### MemTapAnnotation

| Property    | Type             | Description                                                            |
| ----------- | --------         | ------------------                                                     |
| class       | string           | sifive.enterprise.grandcentral.MemTapAnnotation                        |
| taps        | array of strings | An array of components corresponding to the elements of the tap vector |
| source      | string           | Reference target to a FIRRTL memory element                            |

`MemoryTapAnnotation` is used to create a data tap to a FIRRTL memory. The
contents of the MemTap module are the cross-module references to each row of
the tapped memory. Attaching this annotation to memories with aggregate data
types is not supported.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.MemTapAnnotation",
  "taps":[
    "GCTMemTap.MemTap.mem[0]",
    "GCTMemTap.MemTap.mem[1]"
  ],
  "source":"~GCTMemTap|GCTMemTap>mem"
}
```

