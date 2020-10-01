[Back to table of contents](index.md#Table-of-contents)

# ESI Type System: Basic Types

Here we present an overview of the base type system. Extensions to this to
support streaming or MMIO are described in those sections.

## Booleans, Integers, Fixed & Floating Point Numbers

**Keywords**: `bool`, `byte`, `bit`, `uint`, `int`, `fixed`, `float`, `ufixed`, `ufloat`

These types are all **parameterized** ([discussed
later](#parameterized-types)) by the number of bits they consume. For
instance, to get a 7-bit unsigned integer, use `uint<7>` alias. For a signed
fixed point number with 2 bits of whole part and 10 bits of fraction (for a
total of 13 bits, including the sign bit), use `fixed<2, 10>`.

As much as possible, the ESI dialect uses the [standard MLIR
types](https://mlir.llvm.org/docs/LangRef/#standard-types).

### Examples

| MLIR syntax | Description |
| :--- | :--- |
| `!esi.float<true, 10, 21>` | Signed floating point number -- 1 sign bit, 10 bits magnitude, and 21 bits mantissa |
| `ui9` | Unsigned integer -- 9 bits |
| `!esi.fixed<false, 0, 10>` | Unsigned fixed point -- 0 sign bits, 0 whole bits, 10 fraction bits |
| `i101` | Signed integer -- 1 sign bit, 100 whole bits |
| `!esi.fixes<true, 0, 32>` | Signed fixed point -- 1 sign bit, 0 whole bits, 32 fraction bits |

Parameterized floating point and fixed point types do not have a natural
mapping into Verilog's type system. Language mappings (not described here)
define how non-native types are mapped.

## Enums

**Keyword**: `enum`

`Enums` specify a list of symbols which are automatically mapped to an
appropriately-sized `uint`, optionally, the specific numeric value of the
options. If specific values are not specified, the compiler assigns
values in-order, starting at 0 and skipping any values which have been
explicitly assigned. This is consistent with most software languages.

### Enum Example

```c++
enum Features {
    DDR, // Will get assigned 1
    Network = 0,
    PCIe, // Will get assigned 2
}
```

```mlir
!esi.enum<["DDR", "Network", "PCIe"]>
```

## Arrays

Any type can be made into an array of itself. Arrays must be statically
sized. Multidimensional arrays are supported but must be statically
sized in all dimensions. Note: in most languages by default, arrays will
be presented in one clock cycle so users are advised to use [Data
Windows](streaming.md#data-windows) on large arrays.

### Array Examples

| MLIR syntax | Description | Total Size |
| - | - | - |
| `!esi.array<!esi.float<9, 22>, 10>` | 10 x `float<9, 22>` | 320 bits |
| `!esi.array<uint9, 12>` | 12 x `uint9` | 108 bits |
| `!esi.array<!esi.array<byte, 9>, 4>` | 4 x 9 x `byte` | 288 bits |
| `!esi.array<!esi.ufixed<0, 10>, 215>` | 215 x `ufixed<0, 10>` | 2150 bits |

## Structs

Structs are C-like or SystemVerilog-like aggregate constructs.

The alignmant of members in `structs` in hardware has yet to be decided.
Alignment in software is expected to be C-compatible layout. There will be
alignment specifiers but they have yet to be worked out.

```c++
struct {
    int i;
    float f;
    struct {
        bool b;
    } s;
}
```

```mlir
!esi.struct<
  {i, i32},
  {f, !esi.float<1, 9, 22>},
  {s, !esi.struct<
    {b, bool}> } >
```

## Unions

Unions in ESI are _descriminated_, much like SystemVerilog unions. In other
words, a tag header is implictly included to inform the designer which of the
members is valid.

### Union Examples

```c++
union GenericUnion {
    // This union is discriminated so it can be queried for the member
    // it represents
    struct {
        uint8 val;
    } data; // "data" is the tag symbol

    struct {} end; // "end" is the tag symbol
}
```

```mlir
!esi.union<
  {data, !esi.struct<{val, u8}>},
  {end, none}>
```

## Lists

Lists are used to reason about variably-sized data. There are two types of
lists: those for which the size is known before transmission begins and those
for which it isn't (a variably sized list, or just **list**). For now, only
the latter is supported. On the wire, lists must generally be ended with an
end-of-list symbol. As such, a list wherein the length is known in advance is
an optimization.

When lists are members of other data types (e.g. `structs`), they must be
completely transmitted and read in the order in which they appear. Anything
appearing after said list will not be available until after the entirety of
the list is read.

### List Examples

```c++
list struct {
    fixed<4, 12> x;
    fixed<4, 12> y;
} // a list of coordinates
```

```mlir
!esi.list<!esi.struct< {x, !esi.fixed<true, 4, 12>}, {y, !esi.fixed<true, 4, 12>} >>
```

## String

Strings are variable length constructs, with the width of each item in the
'list' determined by the encoding. The reason we break out strings as a
separate data type (vs a list of `ui8`) is the semantics of the encoding
matter quite a bit at compile time. If a particular module only processes
ASCII strings, the design doesn't want to send in UTF8.

This would be better dealt with by a parameterized struct but those will not
be supported initially.

```mlir
!esi.string<ASCII>
!esi.string<UTF8>
!esi.string<UTF16>
!esi.string<UTF32>
```

## Message Pointer

Message pointers are pointers to data _within_ the same message. They
contain: a) the type of the data being pointed and b) the offset of data to
which it is pointing **in bits** relative to the **location of the pointer**,
*not* an absolute location. The offset can (optionally) be negative
indicating that the data has previously occurred. As of now, message pointers
cannot specify the guaranteed alignment, though that is desirable.

```mlir
!esi.ptr<i4, true> // Pointer to a 4-bit signed integer, negative offset allowed
!esi.ptr<!esi.struct<...>, false> // Pointer to a struct, guaranteed positive offset
```

## Parameterized Types

**This is not currently supported!**

Any `struct` or `union` can include **parameters**. Parameters can be either
constant (compile time computable) integers or type names. These are like a
very simple version of C++ templates or C\# generics.

### Parameterized Types Examples

```c++
struct PrefixedList<type T> {
    uint10 header;
    list T data;
}
```

```c++
struct ArrayOf<type T, int N> {
    T[N] data;
}
```
