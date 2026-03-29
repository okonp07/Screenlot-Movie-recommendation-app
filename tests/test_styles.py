from __future__ import annotations

import unittest

from screenlot.styles import (
    build_global_css,
    normalize_theme_mode,
    plotly_layout,
    plotly_template,
)


class ScreenLotStyleTests(unittest.TestCase):
    def test_invalid_theme_mode_defaults_to_dark(self) -> None:
        self.assertEqual(normalize_theme_mode("midnight"), "dark")
        self.assertEqual(normalize_theme_mode(None), "dark")

    def test_plotly_template_matches_theme(self) -> None:
        self.assertEqual(plotly_template("dark"), "plotly_dark")
        self.assertEqual(plotly_template("light"), "plotly_white")

    def test_light_plotly_layout_uses_white_or_transparent_backgrounds(self) -> None:
        layout = plotly_layout("light")
        self.assertEqual(layout["paper_bgcolor"], "rgba(255, 255, 255, 0)")
        self.assertEqual(layout["plot_bgcolor"], "#ffffff")

    def test_light_theme_css_contains_light_tokens(self) -> None:
        css = build_global_css("light")
        self.assertIn("color-scheme: light", css)
        self.assertIn("--screenlot-app-bg-start: #fcfaff;", css)
        self.assertIn("--screenlot-button-text: #221b2c;", css)


if __name__ == "__main__":
    unittest.main()
