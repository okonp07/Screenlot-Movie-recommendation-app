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
        self.assertEqual(layout["plot_bgcolor"], "rgba(255, 255, 255, 0)")
        self.assertEqual(layout["font"]["color"], "#000000")
        self.assertEqual(layout["legend"]["bgcolor"], "rgba(255, 255, 255, 0)")

    def test_light_theme_css_contains_light_tokens(self) -> None:
        css = build_global_css("light")
        self.assertIn("color-scheme: light", css)
        self.assertIn("--screenlot-app-bg-start: #fcfaff;", css)
        self.assertIn("--screenlot-button-text: #221b2c;", css)
        self.assertIn("--screenlot-table-text: #000000;", css)
        self.assertIn("--screenlot-table-row-bg: rgba(255, 255, 255, 0);", css)
        self.assertIn("--screenlot-wordmark-filter: brightness(0) saturate(100%) opacity(0.92);", css)


if __name__ == "__main__":
    unittest.main()
