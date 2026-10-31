# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s
"""Collect ordinary OM objects in a class field, without domain machinery."""

import circt
from circt.dialects import om
from circt.ir import Context, Location, Module, Type


def main():
  with Context() as ctx, Location.unknown():
    circt.register_dialects(ctx)
    module = Module.parse("""
    module {
      om.class @Component(%name: !om.string, %id: !om.integer)
          -> (name: !om.string, id: !om.integer) {
        om.class.fields %name, %id : !om.string, !om.integer
      }

      // The constructor allocates an empty collection for each catalog.
      om.class @Catalog(%name: !om.string)
          -> (name: !om.string, components: !om.collection<!om.class.type<@Component>>) {
        %components = om.collection : !om.collection<!om.class.type<@Component>>
        om.class.fields %name, %components : !om.string, !om.collection<!om.class.type<@Component>>
      }

      // Registration has an effect even though the object has no fields and
      // its result is unused. The destination is an ordinary class field.
      om.class @Register(%catalog: !om.class.type<@Catalog>,
                         %component: !om.class.type<@Component>) {
        %components = om.object.field %catalog["components"] : (!om.class.type<@Catalog>) -> !om.collection<!om.class.type<@Component>>
        om.collection_push %components, %component : !om.collection<!om.class.type<@Component>>, !om.class.type<@Component>
        om.class.fields
      }

      om.class @Assembly(%catalog: !om.class.type<@Catalog>)
          -> (catalog: !om.class.type<@Catalog>) {
        %aluName = om.constant "alu" : !om.string
        %cacheName = om.constant "cache" : !om.string
        %one = om.constant #om.integer<1 : si32> : !om.integer
        %two = om.constant #om.integer<2 : si32> : !om.integer
        %alu = om.object @Component(%aluName, %one) : (!om.string, !om.integer) -> !om.class.type<@Component>
        %cache = om.object @Component(%cacheName, %two) : (!om.string, !om.integer) -> !om.class.type<@Component>
        %a = om.object @Register(%catalog, %alu) : (!om.class.type<@Catalog>, !om.class.type<@Component>) -> !om.class.type<@Register>
        %b = om.object @Register(%catalog, %cache) : (!om.class.type<@Catalog>, !om.class.type<@Component>) -> !om.class.type<@Register>
        om.class.fields %catalog : !om.class.type<@Catalog>
      }

      om.class @UnknownCollections(%missing: !om.collection<!om.integer>)
          -> (items: !om.collection<!om.integer>, missing: !om.collection<!om.integer>) {
        %unknown = om.unknown : !om.integer
        %items = om.collection %unknown : !om.collection<!om.integer>
        om.collection_push %missing, %unknown : !om.collection<!om.integer>, !om.integer
        om.class.fields %items, %missing : !om.collection<!om.integer>, !om.collection<!om.integer>
      }
    }
    """)
    evaluator = om.Evaluator(module)
    catalog = evaluator.instantiate("Catalog", "example")
    assert isinstance(catalog.components, om.List)
    assert len(catalog.components) == 0

    assembly = evaluator.instantiate("Assembly", catalog)
    populated = assembly.catalog
    assert populated.name == "example"
    assert isinstance(populated.components, om.List)
    assert all(
        isinstance(component, om.Object) for component in populated.components)
    # Collection order is unspecified. Ordinary object fields remain accessible
    # through each element of the returned list.
    components = sorted(
        (component.name, component.id) for component in populated.components)
    assert components == [("alu", 1), ("cache", 2)]

    # API inputs are values: the populated catalog comes back through Assembly,
    # while the caller's original catalog is unchanged.
    assert len(catalog.components) == 0

    # The same list APIs expose unknown elements, while unknown membership is
    # wrapped as Unknown instead of an Object or List.
    unknowns = evaluator.instantiate(
        "UnknownCollections",
        om.Unknown(Type.parse("!om.collection<!om.integer>")))
    assert isinstance(unknowns.items, om.List)
    assert len(unknowns.items) == 1
    assert isinstance(unknowns.items[0], om.Unknown)
    assert isinstance(unknowns.missing, om.Unknown)

    # CHECK: Catalog: example
    # CHECK-NEXT: Components: [('alu', 1), ('cache', 2)]
    # CHECK-NEXT: Input catalog: []
    print("Catalog:", populated.name)
    print("Components:", components)
    print("Input catalog:", list(catalog.components))


if __name__ == "__main__":
  main()
