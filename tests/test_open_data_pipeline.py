import unittest

from screenlot.open_data import extract_release_year, format_imdb_id


class OpenDataPipelineTests(unittest.TestCase):
    def test_format_imdb_id_from_numeric_string(self) -> None:
        self.assertEqual(format_imdb_id("114709"), "tt0114709")

    def test_format_imdb_id_preserves_existing_prefix(self) -> None:
        self.assertEqual(format_imdb_id("tt2301451"), "tt2301451")

    def test_extract_release_year(self) -> None:
        self.assertEqual(extract_release_year("Toy Story (1995)"), "1995")
        self.assertEqual(extract_release_year("Unknown Title"), "")


if __name__ == "__main__":
    unittest.main()

