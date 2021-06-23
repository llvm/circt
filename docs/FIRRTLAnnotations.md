# FIRRTL Annotations

This document describes annotations and their uses in FIRRTL.  Annotations are
an extension to the FIRRTL IR that greatly affect the meaning of the IR.
Annotations are typically stored in a separate JSON file, which will be paired
with a FIRRTL IR file.  Annotations are interpreted and sometimes consumed by
certain FIRRTL passes to modify the IR. Annotations are implemented in the
Scala FIRRTL compiler, although it is not documented in the FIRRTL
specification. 

Annotations are represented as a dictionary, with a "class" field which
describes which annotation it is, and a "target" field which represents the IR
object it is attached to. The annotation's class matches the name of a Java
class in Chisel/FIRRTL. Annotations may have arbitrary additional data
attached.

An example of an annotation is the `DontTouchAnnotation`, which can be used to
indicate to the compiler that a wire "foo" should not be optimized away.
```json
{
  "class":"firrtl.transforms.DontTouchAnnotation",
  "target""~MyCircuit|MyModule>foo"
}

```

## Targets

Annotations and Targets solve a problem unique to hardware. Namely, a circuit
is described, stored, and optimized in a folded representation. However, the
same circuit is eventually unfolded onto a physical die. During compilation it
may exist in a partially folded state. Metadata is attached to folded,
unfolded, or partially folded representations and must be preserved.

Note: that fold is defined in the functional programming sense where a function
is recursively applied to a “larger” data structure to build a “smaller” one.
The SFC’s DedupModules and EliminateTargetPaths (also called “duplication”)
transforms are types of folds and unfolds, respectively. In the folded
representation a module may be instantiated multiple times. Each instance may
be the same or may be parametric. Figure 1 shows a folded representation of a
circuit hierarchy of three modules: Top, Foo, and Bar. Top instantiates two
copies of Foo. Foo instantiates two copies of Bar. This is typically the
representation used in a Hardware Description Language (HDL) like Verilog or
VHDL.

Targets are a mechanism to identify specific hardware in specific instances in
a FIRRTL circuit, regardless of foldedness. A target consists of a circuit, a
root module, an optional instance hierarchy, and an optional reference. A
target can only identify hardware with a name, e.g., a circuit, module,
instance, register, wire, or node. References may further refer to specific
fields or subindices in aggregates. A target with no instance hierarchy is
local. A target with an instance hierarchy is non-local.

Targets use a shorthand syntax of the form:

```
target ::= “~” (circuit) (“|” (module) (“/” (instance) “:” (module) )* (“>” (ref) )?)?
```

A ref (reference) is a string followed by optional subfield/subindex expressions:

```
ref ::= (name) ("[" (index) "]" | "." (field))*
```

Targets are specific enough to refer to any specific module in a folded,
unfolded, or partially folded representation. By example the following targets
in Figure 1 map to one or more modules in Figure 2:

```firrtl
circuit Top:
  module Top:
    inst a of Foo
  module Foo:
    inst c of Bar
  module Bar:
    skip
```

| Target                      | Unfolded Modules                 |
|-----------------------------|----------------------------------|
| "~Top&#124;Top"             | Top                              |
| "~Top&#124;Foo"             | Top.a_Foo, Top.b_Foo             |
| "~Top&#124;Top/a:Foo"       | Top.a_Foo                        |
| "~Top&#124;Top/a:Foo/c:Bar" | Top.a_Foo.c_Bar                  |
| "~Top&#124;Foo/d:Bar"       | Top.a_Foo.d_Bar, Top.b_Foo.d_Bar |

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

JSON Annotations map to the builtin MLIR attributes. An annotation is
implemented using a DictionaryAttr, which holds the class, target, any
annotation specific data.

Consider the following example files:
```firrtl
circuit Foo :
  module Bar :
    skip
  module Foo :
    inst bar of Bar
```

```json
[
  {
    "class": "firrtl.stage.TargetDirAnnotation",
    "directory": "/foo/bar/baz"
  },
  {
    "class": "logger.LogLevelAnnotation",
    "globalLogLevel": "info"
  },
  {
    "class": "firrtl.passes.Foo",
    "target": "~Foo"
  },
  {
    "class": "firrl.FakeAnnotation",
    "target": "~Foo|Foo"
  },
  {
    "class": "firrtl.passes.InlineAnnotation",
     "target": "~Foo|Foo>bar"
  }
]
```

This, when read into MLIR, will populate the following attributes:

```mlir
module  {
  firrtl.circuit "Foo" attributes {annotations = [{class = "firrtl.stage.TargetDirAnnotation", directory = "/foo/bar/baz"}, {class = "logger.LogLevelAnnotation", globalLogLevel = "info"}, {class = "firrtl.passes.Foo"}]}  {
    firrtl.module @Bar() {
    }
    firrtl.module @Foo() attributes {annotations = [{class = "firrtl.FakeAnnotation"}]} {
      firrtl.instance @Bar  {annotations = [{class = "firrtl.passes.InlineAnnotation"}], name = "bar"}
    }
  }
}
```

## Annotations In Scala

An annotation is a metadata container. An annotation may be associated with
zero or more targets. Annotations are maintained separately from FIRRTL IR.
Annotations additionally define how they are updated if names in a FIRRTL
circuit change. In Scala, an annotation is a product type with an abstract
update method:

```scala
trait Annotation extends Product {
  def update(renames: RenameMap): Seq[Annotation]
}
```

The update method takes a `RenameMap` as a parameter. All SFC transforms (passes)
return a `RenameMap` that describes every modification that transform made to
targets in the circuit. A `RenameMap` is an associative array of a target to zero
or more targets. In Scala this could be written as:

```scala
type RenameMap extends Map[Target, Seq[Target]]
```

This provides semantics of deletion (an empty value in the map), name changes
(a value of a single new target), duplication (a value with multiple targets,
e.g., during aggregate lowering), and combination (multiple keys that map to
values with overlapping members).

Concrete annotation implementations typically do not implement the Annotation
trait directly. Commonly, one of the following traits is implemented instead.
These represent annotations that are associated with no targets, one target,
and multiple targets:

```scala
trait NoTargetAnnotation extends Annotation {
  def update(renames: RenameMap): Seq[NoTargetAnnotation] = Seq(this)
}

trait SingleTargetAnnotation[A <: Target] extends Annotation {
  def target: A
  def duplicate(newTarget: A): Annotation
  def update(renames: RenameMap): Seq[Annotation] = /* uses ‘duplicate’ */
}

trait MultiTargetAnnotation extends Annotation {
  def targets: Seq[Seq[Target]]
  def duplicate(newTargets: Seq[Seq[Target]]): Annotation
  def update(renames: RenameMap): Seq[Annotation] = /* uses ‘duplicate’ */
}
```

Note that there are uses of custom annotations that do not extend one of these
traits. E.g., SeqMemMetadataAnnotation has two targets and implements a custom
update method.

## Annotations

### [DontTouchAnnotation](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/DontTouchAnnotation.html)

The `DontTouchAnnotation` prevents the removal of elements through
optimization. This annotation also ensures that the name of the object is
preserved, and not discarded or modified. The annotation can be applied to
registers, wires, and nodes.

```
class ::= firrtl.transforms.DontTouchAnnotation
```

### [FlattenAnnotation](https://www.chisel-lang.org/api/firrtl/latest/firrtl/transforms/FlattenAnnotation.html)

Tags an annotation to be consumed by this transform

```
class ::= firrtl.transforms.FlattenAnnotation

```

### [InlineAnnotation](https://www.chisel-lang.org/api/firrtl/latest/firrtl/passes/InlineAnnotation.html)

Indicates that something should be inlined.
```
class ::= firrtl.passes.DontTouchAnnotation
```



