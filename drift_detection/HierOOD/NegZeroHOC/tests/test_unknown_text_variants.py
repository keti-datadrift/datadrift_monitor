import unittest

from negzerohoc.prompt_text import build_parent_unknown_text


class MinimalHierarchy:
    id_node_list = ["root"]
    node_ancestors = {"root": []}


class UnknownTextVariantTest(unittest.TestCase):
    def test_generic_aircraft_text_is_parent_independent(self):
        text = build_parent_unknown_text(
            "fgvc-aircraft",
            MinimalHierarchy(),
            "root",
            variant="generic",
        )
        self.assertEqual(text, "a photo of an unknown aircraft")

    def test_literal_not_is_a_separate_explicit_negation(self):
        text = build_parent_unknown_text(
            "fgvc-aircraft",
            MinimalHierarchy(),
            "root",
            variant="literal_not",
        )
        self.assertIn(" not ", text)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            build_parent_unknown_text(
                "fgvc-aircraft",
                MinimalHierarchy(),
                "root",
                variant="unsupported",
            )


if __name__ == "__main__":
    unittest.main()
