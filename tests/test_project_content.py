from pathlib import Path
import unittest

from screenlot.content import (
    CONTRIBUTORS,
    SCREENLOT_BANNER,
    SCREENLOT_BANNER_DARK,
    SCREENLOT_BANNER_LIGHT,
    SCREENLOT_LOGO,
    STREAMLIT_CONCEPT,
)
from screenlot.runtime import PACKAGED_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScreenLotContentTests(unittest.TestCase):
    def test_brand_assets_exist(self) -> None:
        self.assertTrue(SCREENLOT_LOGO.exists())
        self.assertTrue(SCREENLOT_BANNER.exists())
        self.assertTrue(SCREENLOT_BANNER_DARK.exists())
        self.assertTrue(SCREENLOT_BANNER_LIGHT.exists())
        self.assertTrue(STREAMLIT_CONCEPT.exists())

    def test_contributors_have_images(self) -> None:
        self.assertGreaterEqual(len(CONTRIBUTORS), 5)
        for contributor in CONTRIBUTORS:
            self.assertIsInstance(contributor.image_path, Path)
            self.assertTrue(contributor.image_path.exists(), contributor.name)

    def test_companion_notebooks_exist(self) -> None:
        self.assertTrue((PROJECT_ROOT / "notebooks" / "ScreenLot_Modeling_Workbook.ipynb").exists())
        self.assertTrue((PROJECT_ROOT / "notebooks" / "ScreenLot_EDA_Workbook.ipynb").exists())

    def test_packaged_demo_data_exists(self) -> None:
        self.assertTrue((PACKAGED_DATA_DIR / "ratings.csv").exists())
        self.assertTrue((PACKAGED_DATA_DIR / "movies.csv").exists())
        self.assertTrue((PACKAGED_DATA_DIR / "links.csv").exists())


if __name__ == "__main__":
    unittest.main()
