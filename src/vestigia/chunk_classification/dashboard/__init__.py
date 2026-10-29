"""Local Streamlit dashboard for interactive chunk classification.

Requires the ``dashboard`` extra. Start it with ``vestigia-dashboard``.
"""

import sys
from pathlib import Path

STREAMLIT_OPTIONS = {
    "theme.base": "light",
    "theme.primaryColor": "#1D5FB0",
    "theme.backgroundColor": "#FFFFFF",
    "theme.secondaryBackgroundColor": "#F4F6F8",
    "theme.textColor": "#1F2933",
    # Fonts load from Google Fonts; offline, the system sans and mono are used.
    "theme.font": "IBM Plex Sans:https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap, sans-serif",
    "theme.codeFont": "IBM Plex Mono:https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap, monospace",
    "theme.baseFontSize": "18",
    "theme.headingFontSizes": "1.3rem",
    "theme.headingFontWeights": "600",
    "theme.baseRadius": "0.375rem",
    "theme.borderColor": "#E3E7EC",
    "client.toolbarMode": "minimal",
}


def main() -> None:
    """Run the dashboard in the local browser.

    Examples
    --------
    From a shell with the ``dashboard`` extra installed::

        vestigia-dashboard
    """
    from streamlit.web import cli

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).with_name("app.py")),
        "--browser.gatherUsageStats=false",
        *(f"--{key}={value}" for key, value in STREAMLIT_OPTIONS.items()),
        *sys.argv[1:],
    ]
    sys.exit(cli.main())
