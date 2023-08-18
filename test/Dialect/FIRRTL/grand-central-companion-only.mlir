// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-grand-central{instantiate-companion-only},symbol-dce))' -split-input-file %s | FileCheck %s

// This is the main test that includes different interfaces of different
// types. All the interfaces share a common, simple circuit that provides two
// RefType signals, "foo" and "bar".

firrtl.circuit "InterfaceGroundType" attributes {
  annotations = [
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          name = "foo",
          id = 1 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "multi\nline\ndescription\nof\nbar",
          name = "bar",
          id = 2 : i64
        }
      ],
      id = 0 : i64,
      name = "GroundView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "foo",
              id = 4 : i64
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "bar",
              id = 5 : i64
            }
          ],
          name = "vector"
        }
      ],
      id = 3 : i64,
      name = "VectorView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "BundleView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "Bundle",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "foo",
              id = 7 : i64
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "bar",
              id = 8 : i64
            }
          ],
          name = "bundle"
        }
      ],
      id = 6 : i64,
      name = "BundleView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfBundleView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedBundleType",
              defName = "Bundle2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo",
                  id = 10 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar",
                  id = 11 : i64
                }
              ],
              name = "bundle2"
            }
          ],
          name = "vector"
        }
      ],
      id = 9 : i64,
      name = "VectorOfBundleView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfVectorView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo",
                  id = 13 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar",
                  id = 14 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "baz",
                  id = 15 : i64
                }
              ],
              name = "vector2"
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo",
                  id = 16 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar",
                  id = 17 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "baz",
                  id = 18 : i64
                }
              ],
              name = "vector2"
            }
          ],
          name = "vector"
        }
      ],
      id = 12 : i64,
      name = "VectorOfVectorView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "ZeroWidthView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 20 : i64,
          name = "zerowidth"
        }
      ],
      id = 19 : i64,
      name = "ZeroWidthView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "ConstantView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",

          name = "foo",
          id = 22 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          name = "bar",
          id = 23 : i64
        }
      ],
      id = 21 : i64,
      name = "ConstantView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "UnsupportedView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedStringType",
          name = "string"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedBooleanType",
          name = "boolean"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedIntegerType",
          name = "integer"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedDoubleType",
          name = "double"
        }
      ],
      id = 24 : i64,
      name = "UnsupporteView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfVerbatimView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector4",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                }
              ],
              name = "vector4"
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector4",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                }
              ],
              name = "vector4"
            }
          ],
          name = "vectorOfVerbatim"
        }
      ],
      id = 25 : i64,
      name = "VectorOfVerbatimView"
    },
    {
      class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
      directory = "gct-dir",
      filename = "bindings.sv"
    },
    {
      class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
      filename = "gct.yaml"
    }
  ]
} {
  // CHECK: firrtl.module @Companion() attributes {comment = "VCS coverage exclude_file", output_file = #hw.output_file<"gct-dir/", excludeFromFileList, includeReplicatedOps>} {
  firrtl.module @Companion() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "GroundView",
        id = 0 : i64,
        name = "GroundView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorView",
        id = 3 : i64,
        name = "VectorView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "BundleView",
        id = 6 : i64,
        name = "BundleView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorOfBundleView",
        id = 9 : i64,
        name = "VectorOfBundleView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorOfVectorView",
        id = 12 : i64,
        name = "VectorOfVectorView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "ZeroWidthView",
        id = 19 : i64,
        name = "ZeroWidthView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "ConstantView",
        id = 21 : i64,
        name = "ConstantView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "UnsupportedView",
        id = 24 : i64,
        name = "UnsupportedView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorOfVerbatimView",
        id = 25 : i64,
        name = "VectorOfVerbatimView"
      }
    ]
  } {

    // These are dummy references created for the purposes of the test.
    %_ui0 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<0>
    %_ui1 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<1>
    %_ui2 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<2>
    %ref_ui0 = firrtl.ref.send %_ui0 : !firrtl.uint<0>
    %ref_ui1 = firrtl.ref.send %_ui1 : !firrtl.uint<1>
    %ref_ui2 = firrtl.ref.send %_ui2 : !firrtl.uint<2>

    %ui1 = firrtl.ref.resolve %ref_ui1 : !firrtl.probe<uint<1>>
    %foo = firrtl.node %ui1 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 1 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 4 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 5 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 7 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 10 : i64
        }
      ]
    } : !firrtl.uint<1>

    %ui2 = firrtl.ref.resolve %ref_ui2 : !firrtl.probe<uint<2>>
    %bar = firrtl.node %ui2 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 2 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 8 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 11 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 13 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 14 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 15 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 16 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 17 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 18 : i64
        }
      ]
    } : !firrtl.uint<2>

    %ui0 = firrtl.ref.resolve %ref_ui0 : !firrtl.probe<uint<0>>
    %baz = firrtl.node %ui0 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 20 : i64
        }
      ]
    } : !firrtl.uint<0>

    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c-1_si2 = firrtl.constant -1 : !firrtl.sint<2>

    %node_c0_ui1 = firrtl.node %c0_ui1 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 22 : i64
        }
      ]
    } : !firrtl.uint<1>

    %node_c-1_si2 = firrtl.node %c-1_si2 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 23 : i64
        }
      ]
    } : !firrtl.sint<2>

  }
  // CHECK: firrtl.instance companion {output_file = #hw.output_file<"bindings.sv", excludeFromFileList>} @Companion()
  firrtl.module @InterfaceGroundType() {
    firrtl.instance companion @Companion()
  }
}
