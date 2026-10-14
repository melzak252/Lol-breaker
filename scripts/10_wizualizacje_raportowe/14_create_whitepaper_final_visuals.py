"""Generate portfolio-oriented SVG figures for the final whitepaper.

The script intentionally uses only the Python standard library, so it can run in
minimal thesis/portfolio environments without installing plotting packages.
"""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "docs" / "assets" / "whitepaper_final"


def _setup_output_dir() -> None:
    """Create the output directory for generated whitepaper assets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _text(x_pos: int, y_pos: int, value: str, size: int = 18, weight: str = "400") -> str:
    """Create an SVG text element.

    Args:
        x_pos: X coordinate of the text anchor.
        y_pos: Y coordinate of the baseline.
        value: Text value to render.
        size: Font size in pixels.
        weight: CSS font-weight value.

    Returns:
        SVG text element as a string.
    """
    return (
        f'<text x="{x_pos}" y="{y_pos}" text-anchor="middle" '
        f'font-family="Inter, Segoe UI, Arial" font-size="{size}" '
        f'font-weight="{weight}" fill="#102027">{escape(value)}</text>'
    )


def _box(
    x_pos: int,
    y_pos: int,
    width: int,
    height: int,
    title: str,
    subtitle: str,
    fill: str,
) -> str:
    """Create a rounded SVG box with title and subtitle.

    Args:
        x_pos: Left x coordinate.
        y_pos: Top y coordinate.
        width: Box width.
        height: Box height.
        title: Main label.
        subtitle: Secondary label.
        fill: Fill color.

    Returns:
        SVG fragment containing the box and labels.
    """
    center_x = x_pos + width // 2
    return "\n".join(
        [
            (
                f'<rect x="{x_pos}" y="{y_pos}" width="{width}" height="{height}" '
                f'rx="18" fill="{fill}" stroke="#263238" stroke-width="2" />'
            ),
            _text(center_x, y_pos + 44, title, size=20, weight="700"),
            _text(center_x, y_pos + 76, subtitle, size=14, weight="400"),
        ]
    )


def _write_svg(path: Path, width: int, height: int, body: str) -> None:
    """Write a complete SVG document to disk.

    Args:
        path: Output SVG file path.
        width: SVG viewport width.
        height: SVG viewport height.
        body: SVG body content.
    """
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#FAFAFA" />
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def create_roadmap_visual() -> Path:
    """Create a high-level roadmap figure for the modelling process.

    Returns:
        Path to the generated SVG file.
    """
    stages = [
        ("Dane", "GOL.GG + odds", "#B3E5FC"),
        ("Model", "ratingi + forma", "#C8E6C9"),
        ("Rynek", "opening odds", "#FFF9C4"),
        ("Hybryda", "model + cena", "#D1C4E9"),
        ("Decyzja", "EV + staking", "#FFE0B2"),
        ("Walidacja", "CLV + stress", "#FFCDD2"),
    ]
    body_parts = [
        _text(750, 54, "Roadmapa POC: od danych do decyzji dodatniego EV", 28, "800"),
    ]
    for index, (title, subtitle, fill) in enumerate(stages):
        x_pos = 55 + index * 235
        body_parts.append(_box(x_pos, 128, 180, 106, title, subtitle, fill))
        if index < len(stages) - 1:
            arrow_x = x_pos + 190
            body_parts.append(
                f'<line x1="{arrow_x}" y1="181" x2="{arrow_x + 38}" y2="181" '
                'stroke="#455A64" stroke-width="4" marker-end="url(#arrow)" />'
            )
    body_parts.extend(
        [
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
            'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#455A64" /></marker></defs>',
            _text(750, 330, "To nie jest tylko model zwycięzcy. To proces: cena → przewaga → ryzyko → walidacja.", 18),
        ]
    )
    output_path = OUTPUT_DIR / "poc_roadmap.svg"
    _write_svg(output_path, 1500, 390, "\n".join(body_parts))
    return output_path


def create_decision_funnel_visual() -> Path:
    """Create an educational funnel from predictions to accepted bets.

    Returns:
        Path to the generated SVG file.
    """
    steps = [
        ("Wszystkie mecze", "dane sportowe + kursy", 920, "#E3F2FD"),
        ("Predykcja", "p_model i p_market", 800, "#BBDEFB"),
        ("Hybryda", "kalibracja ceny", 680, "#C8E6C9"),
        ("EV > 5%", "po podatku i slippage", 560, "#FFF59D"),
        ("Staking", "Kelly 0.25 + cap", 440, "#FFCC80"),
        ("Kontrola", "CLV + stress testy", 320, "#EF9A9A"),
    ]
    body_parts = [_text(500, 52, "Decision funnel: kiedy predykcja staje się zakładem", 26, "800")]
    for index, (title, subtitle, width, fill) in enumerate(steps):
        x_pos = (1000 - width) // 2
        y_pos = 95 + index * 76
        body_parts.append(
            f'<rect x="{x_pos}" y="{y_pos}" width="{width}" height="52" rx="14" '
            f'fill="{fill}" stroke="#263238" stroke-width="2" />'
        )
        body_parts.append(
            f'<text x="{x_pos + 26}" y="{y_pos + 33}" font-family="Inter, Segoe UI, Arial" '
            f'font-size="18" font-weight="700" fill="#102027">{escape(title)}</text>'
        )
        body_parts.append(
            f'<text x="{x_pos + width - 26}" y="{y_pos + 33}" text-anchor="end" '
            f'font-family="Inter, Segoe UI, Arial" font-size="15" fill="#455A64">{escape(subtitle)}</text>'
        )
    body_parts.append(
        _text(500, 590, "Każda warstwa odrzuca pozorne value: brak ceny, brak EV albo zbyt duże ryzyko.", 17)
    )
    output_path = OUTPUT_DIR / "decision_funnel.svg"
    _write_svg(output_path, 1000, 640, "\n".join(body_parts))
    return output_path


def create_poc_vs_production_visual() -> Path:
    """Create a visual separation between POC evidence and production work.

    Returns:
        Path to the generated SVG file.
    """
    body_parts = [
        _text(600, 52, "Co udowadnia POC, a czego jeszcze nie udowadnia?", 26, "800"),
        _box(90, 120, 410, 135, "POC / backtest", "pipeline, EV, CLV, stress", "#C8E6C9"),
        _box(700, 120, 410, 135, "Produkcja", "forward test, fill, limity", "#FFCDD2"),
        '<line x1="525" y1="188" x2="675" y2="188" stroke="#455A64" stroke-width="5" marker-end="url(#arrow)" />',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#455A64" /></marker></defs>',
        _text(600, 335, "Najuczciwsza teza: historyczny edge i roadmapa budowy systemu, ale nie deklaracja skalowalnej strategii.", 17),
    ]
    output_path = OUTPUT_DIR / "poc_vs_production.svg"
    _write_svg(output_path, 1200, 400, "\n".join(body_parts))
    return output_path


def create_profit_anatomy_visual() -> Path:
    """Create a visual summary of profit anatomy by odds bucket.

    Returns:
        Path to the generated SVG file.
    """
    segments = [
        ("Near-even", 176, 11.17, 63.64, 2.08, "#BBDEFB"),
        ("Underdog", 401, 25.94, 46.63, 3.11, "#C8E6C9"),
        ("Longshot", 236, 165.93, 37.29, 7.58, "#FFCC80"),
    ]
    max_yield = max(segment[2] for segment in segments)
    body_parts = [
        _text(600, 52, "Anatomia zysku: nie accuracy, tylko cena", 26, "800"),
        _text(600, 86, "Win Rate spada przy wyższych kursach, ale Yield rośnie, gdy rynek niedoszacowuje underdogi.", 15),
    ]
    for index, (name, bets, yield_value, win_rate, odds, fill) in enumerate(segments):
        x_pos = 135 + index * 330
        bar_height = int(230 * yield_value / max_yield)
        y_pos = 360 - bar_height
        body_parts.append(
            f'<rect x="{x_pos}" y="{y_pos}" width="190" height="{bar_height}" rx="12" '
            f'fill="{fill}" stroke="#263238" stroke-width="2" />'
        )
        body_parts.append(_text(x_pos + 95, 395, name, 20, "800"))
        body_parts.append(_text(x_pos + 95, y_pos - 14, f"Yield {yield_value:.1f}%", 18, "700"))
        body_parts.append(_text(x_pos + 95, 426, f"Bets: {bets} | WR: {win_rate:.1f}%", 14))
        body_parts.append(_text(x_pos + 95, 450, f"Avg odds: {odds:.2f}", 14))
    body_parts.append(
        '<line x1="90" y1="360" x2="1110" y2="360" stroke="#455A64" stroke-width="2" />'
    )
    body_parts.append(_text(600, 510, "Największy insight: strategia nie musi wygrywać większości kuponów, jeśli kupuje niedoszacowane prawdopodobieństwo.", 16))
    output_path = OUTPUT_DIR / "profit_anatomy.svg"
    _write_svg(output_path, 1200, 560, "\n".join(body_parts))
    return output_path


def main() -> None:
    """Generate all final whitepaper portfolio figures."""
    _setup_output_dir()
    paths = [
        create_roadmap_visual(),
        create_decision_funnel_visual(),
        create_poc_vs_production_visual(),
        create_profit_anatomy_visual(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
