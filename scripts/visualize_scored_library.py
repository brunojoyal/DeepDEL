#!/usr/bin/env python3
"""Visualize a scored 3-cycle DEL library as PNG images.

Given a scored library CSV with columns like::

    bb1_id,bb2_id,bb3_id,bb1_smiles,bb2_smiles,bb3_smiles,smiles,docking_score

this script writes two PNGs:

1. ``*_bb_pools.png``: the three BB pools, arranged as three columns by default
   or as three stacked rows separated by horizontal dividers via
   ``--pool-layout rows``. Each pool is laid out in a grid controllable with
   ``--pool-grid ROWS COLS``, and BB IDs can be hidden with ``--hide-bb-ids``.
   An optional outer margin and rectangular frame can be added with ``--margin``
   and ``--frame-width``; ``--divider-inset`` keeps the horizontal separators
   from touching the frame.
2. ``*_complete_del.png``: the complete DEL arranged as BB1 groups, each
   containing BB2 rows, each row containing BB3 products labeled with scores
   and SMILES. For libraries above ``--complete-del-threshold``, a random
   sampled-products PNG is written instead.

Example:
    python scripts/visualize_scored_library.py \
        data/scored_libraries/active_learning/20260511_103025_outer0_inner0_18.csv \
        --outdir outputs/library_visualizations

Render a 10x10x10 library's BB pools as three stacked 2x5 grids, framed:
    python scripts/visualize_scored_library.py \
        data/scored_libraries/active_learning/20260511_103025_outer0_inner0_18.csv \
        --outdir outputs/library_visualizations \
        --pool-layout rows --pool-grid 2 5 --hide-bb-ids \
        --margin 24 --frame-width 3 --divider-inset 10
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdMolDescriptors


REQUIRED_COLUMNS = (
    "bb1_id",
    "bb2_id",
    "bb3_id",
    "bb1_smiles",
    "bb2_smiles",
    "bb3_smiles",
    "smiles",
    "docking_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render BB pool and complete-DEL PNG visualizations from a scored library CSV."
    )
    parser.add_argument("--csv", type=Path, help="Input scored library CSV.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory for output PNGs. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--bb-pools-out",
        type=Path,
        default=None,
        help="Explicit output path for the BB pools PNG.",
    )
    parser.add_argument(
        "--complete-del-out",
        type=Path,
        default=None,
        help="Explicit output path for the complete DEL PNG.",
    )
    parser.add_argument(
        "--sample-out",
        type=Path,
        default=None,
        help="Explicit output path for the sampled-products PNG used for large libraries.",
    )
    parser.add_argument(
        "--no-complete-del",
        action="store_true",
        help="Skip rendering product PNGs entirely, including the sampled fallback.",
    )
    parser.add_argument(
        "--complete-del-threshold",
        type=int,
        default=1000,
        help="Maximum row count for rendering the full complete DEL PNG. Larger libraries use a random sampled-products PNG instead. Default: 1000.",
    )
    parser.add_argument(
        "--sample-grid",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLS"),
        default=(7, 7),
        help="Grid dimensions for sampled-products PNGs when the library exceeds --complete-del-threshold. Default: 7 7.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed for sampled-products PNG generation. Default: 0.",
    )
    parser.add_argument(
        "--bb-cell",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=(420, 360),
        help="Cell size for BB molecules in pixels. Default: 420 360.",
    )
    parser.add_argument(
        "--pool-layout",
        choices=("columns", "rows"),
        default="columns",
        help="Arrangement of the three BB pools: 'columns' (default) places them side by side with vertical dividers; 'rows' stacks them vertically with horizontal dividers.",
    )
    parser.add_argument(
        "--pool-grid",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLS"),
        default=None,
        help="Grid dimensions for each BB pool. E.g. '2 5' renders a 10-BB pool as 2 rows of 5. Default: 1 column when a pool has <=4 BBs, otherwise a near-square grid. Rows grow automatically when needed.",
    )
    parser.add_argument(
        "--hide-bb-ids",
        action="store_true",
        help="Omit the 'BB <id>' line from each BB-pool cell legend.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Outer whitespace padding around the BB pools PNG in pixels. Default: 0.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=0,
        help="Thickness in pixels of a rectangular frame drawn around the BB pools content (just inside the margin). 0 disables the frame. Default: 0.",
    )
    parser.add_argument(
        "--frame-color",
        default="black",
        help="Color of the rectangular frame. Default: black.",
    )
    parser.add_argument(
        "--divider-inset",
        type=int,
        default=0,
        help="Horizontal inset, in pixels, applied to the horizontal separators between stacked BB pools so they stop short of the frame. Default: 0.",
    )
    parser.add_argument(
        "--cell",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=(520, 430),
        help="Cell size for complete DEL molecules in pixels. Default: 520 430.",
    )
    parser.add_argument(
        "--score-format",
        default=".2f",
        help="Python format specifier for docking scores. Default: .2f.",
    )
    parser.add_argument(
        "--kekulize",
        action="store_true",
        help="Attempt kekulization before drawing molecules.",
    )
    parser.add_argument(
        "--sort-numeric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort BB ids numerically when possible. Default: true.",
    )
    parser.add_argument(
        "--score-colors",
        action="store_true",
        help="Enable score-based color borders on product molecules. Green = better binder (approaching --score-min), red = worse binder (approaching --score-max).",
    )
    parser.add_argument(
        "--score-min",
        type=float,
        default=-100.0,
        help="Docking score mapped to pure green. Default: -100.",
    )
    parser.add_argument(
        "--score-max",
        type=float,
        default=0.0,
        help="Docking score mapped to pure red. Default: 0.",
    )
    parser.add_argument(
        "--score-border-width",
        type=int,
        default=8,
        help="Thickness in pixels of the score-color border. Default: 8.",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a reasonably portable TrueType font, falling back to PIL default."""

    candidates = (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def sort_ids(values: Iterable[object], numeric: bool = True) -> List[object]:
    unique = list(dict.fromkeys(values))
    if not numeric:
        return sorted(unique, key=lambda x: str(x))

    def key(value: object) -> Tuple[int, float | str]:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    return sorted(unique, key=key)


def mol_from_smiles(smiles: object, *, kekulize: bool = False) -> Optional[Chem.Mol]:
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass
        return mol
    except Exception:
        return None


def validate_input(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError(f"{path} has no rows")


def default_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    outdir = args.outdir if args.outdir is not None else args.csv.parent
    stem = args.csv.stem
    bb_pools_out = args.bb_pools_out or (outdir / f"{stem}_bb_pools.png")
    complete_del_out = args.complete_del_out or (outdir / f"{stem}_complete_del.png")
    sample_out = args.sample_out or (outdir / f"{stem}_sampled_products.png")
    return bb_pools_out, complete_del_out, sample_out


def draw_grid(
    mols: Sequence[Optional[Chem.Mol]],
    legends: Sequence[str],
    *,
    mols_per_row: int,
    cell_size: Tuple[int, int],
    scores: Optional[Sequence[Optional[float]]] = None,
    score_min: float = -100.0,
    score_max: float = 0.0,
    score_border_width: int = 0,
) -> Image.Image:
    """Draw molecules into a grid with PIL-rendered legends.

    RDKit's built-in grid legends are compact and can collapse blank-line
    spacing. Rendering labels ourselves gives explicit pixel control over the
    gap between score and SMILES lines.

    When ``score_border_width > 0`` and ``scores`` is provided, a colored
    rectangle border is drawn around each cell whose color maps the molecule's
    docking score on a green (strong binders) to red (weak binders) gradient.
    """

    cell_w, cell_h = cell_size
    rows = math.ceil(len(mols) / mols_per_row)
    out = Image.new("RGB", (mols_per_row * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(out)
    legend_font = load_font(max(12, min(20, cell_w // 42)))
    line_h = text_size(draw, "Ag", legend_font)[1] + 10
    mol_h = int(cell_h * 0.58)
    label_top = mol_h + 12

    for i, (mol, legend) in enumerate(zip(mols, legends)):
        col = i % mols_per_row
        row = i // mols_per_row
        x0 = col * cell_w
        y0 = row * cell_h

        if mol is not None:
            mol_img = Draw.MolToImage(mol, size=(cell_w, mol_h)).convert("RGB")
            out.paste(mol_img, (x0, y0))

        # Draw score-based color border around the cell
        if score_border_width > 0 and scores is not None and i < len(scores):
            score = scores[i]
            if score is not None:
                score_val = pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0]
                if not pd.isna(score_val):
                    color = score_to_color(float(score_val), score_min, score_max)
                    draw.rectangle(
                        (x0, y0, x0 + cell_w - 1, y0 + cell_h - 1),
                        outline=color,
                        width=score_border_width,
                    )

        y = y0 + label_top
        for line in legend.split("\n"):
            if line:
                tw, _ = text_size(draw, line, legend_font)
                draw.text((x0 + (cell_w - tw) / 2, y), line, fill="black", font=legend_font)
            y += line_h

    return out


def wrap_legend_line(text: object, width: int = 52) -> str:
    """Wrap one legend field so long SMILES remain visible in grid cells."""

    if pd.isna(text):
        return ""
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=True, break_on_hyphens=False))


def format_mol_weight(smiles: object, fmt: str = ".2f") -> str:
    """Return an exact molecular-weight label for a SMILES string."""

    if pd.isna(smiles):
        return "MW nan Da"
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "MW nan Da"
    return f"MW {format(rdMolDescriptors.CalcExactMolWt(mol), fmt)} Da"


def make_bb_legend(bb_id: object, smiles: object, *, show_bb_id: bool = True) -> str:
    """Legend for a building block: optional ID, molecular weight, then SMILES."""

    prefix = f"BB {bb_id}\n" if show_bb_id else ""
    return f"{prefix}{format_mol_weight(smiles)}\n{wrap_legend_line(smiles)}"


def make_product_legend(score: object, smiles: object, score_format: str) -> str:
    """Legend for complete DEL products: docking score, molecular weight, and SMILES."""

    return f"score {format_score(score, score_format)}\n{format_mol_weight(smiles)}\n\n\n{wrap_legend_line(smiles)}"


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def add_header(image: Image.Image, title: str, *, font: ImageFont.ImageFont, pad: int = 12) -> Image.Image:
    title_h = max(32, text_size(ImageDraw.Draw(Image.new("RGB", (1, 1))), title, font)[1] + 2 * pad)
    out = Image.new("RGB", (image.width, image.height + title_h), "white")
    draw = ImageDraw.Draw(out)
    tw, th = text_size(draw, title, font)
    draw.text(((image.width - tw) / 2, (title_h - th) / 2), title, fill="black", font=font)
    out.paste(image, (0, title_h))
    return out


def add_frame(
    image: Image.Image,
    *,
    margin: int = 0,
    frame_width: int = 0,
    frame_color: str = "black",
    background: str = "white",
) -> Image.Image:
    """Add an outer margin and an optional rectangular frame around an image.

    ``margin`` adds ``background`` padding around the image. When
    ``frame_width > 0``, a rectangle is drawn around the content region (just
    inside the margin).
    """

    if margin <= 0 and frame_width <= 0:
        return image

    out_width = image.width + 2 * margin
    out_height = image.height + 2 * margin
    out = Image.new("RGB", (out_width, out_height), background)
    out.paste(image, (margin, margin))
    if frame_width > 0:
        draw = ImageDraw.Draw(out)
        draw.rectangle(
            (margin, margin, margin + image.width - 1, margin + image.height - 1),
            outline=frame_color,
            width=frame_width,
        )
    return out


def hstack(
    images: Sequence[Image.Image],
    *,
    gap: int = 24,
    background: str = "white",
    divider: bool = False,
    divider_color: str = "black",
    divider_width: int = 3,
) -> Image.Image:
    width = sum(img.width for img in images) + gap * max(0, len(images) - 1)
    height = max(img.height for img in images)
    out = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(out)
    x = 0
    for idx, img in enumerate(images):
        out.paste(img, (x, 0))
        if divider and idx < len(images) - 1:
            divider_x = x + img.width + gap // 2
            x0 = divider_x - divider_width // 2
            draw.rectangle((x0, 0, x0 + divider_width - 1, height), fill=divider_color)
        x += img.width + gap
    return out


def vstack(
    images: Sequence[Image.Image],
    *,
    gap: int = 24,
    background: str = "white",
    divider: bool = False,
    divider_color: str = "black",
    divider_width: int = 3,
    divider_inset: int = 0,
) -> Image.Image:
    width = max(img.width for img in images)
    height = sum(img.height for img in images) + gap * max(0, len(images) - 1)
    out = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(out)
    y = 0
    for idx, img in enumerate(images):
        out.paste(img, ((width - img.width) // 2, y))
        if divider and idx < len(images) - 1:
            divider_y = y + img.height + gap // 2
            y0 = divider_y - divider_width // 2
            inset = max(0, min(divider_inset, width // 2))
            draw.rectangle((inset, y0, width - inset, y0 + divider_width - 1), fill=divider_color)
        y += img.height + gap
    return out


def render_bb_pools(
    df: pd.DataFrame,
    out_path: Path,
    *,
    cell_size: Tuple[int, int],
    kekulize: bool,
    sort_numeric: bool,
    pool_layout: str = "columns",
    pool_grid: Optional[Tuple[int, int]] = None,
    hide_bb_ids: bool = False,
    margin: int = 0,
    frame_width: int = 0,
    frame_color: str = "black",
    divider_inset: int = 0,
) -> None:
    title_font = load_font(24, bold=True)
    pool_images: List[Image.Image] = []
    failures = 0

    for idx in (1, 2, 3):
        id_col = f"bb{idx}_id"
        smiles_col = f"bb{idx}_smiles"
        pool = df[[id_col, smiles_col]].drop_duplicates(subset=[id_col]).copy()
        pool_ids = sort_ids(pool[id_col].tolist(), numeric=sort_numeric)
        pool = pool.set_index(id_col, drop=False)

        mols: List[Optional[Chem.Mol]] = []
        legends: List[str] = []
        for bb_id in pool_ids:
            row = pool.loc[bb_id]
            mol = mol_from_smiles(row[smiles_col], kekulize=kekulize)
            if mol is None:
                failures += 1
            mols.append(mol)
            legends.append(make_bb_legend(bb_id, row[smiles_col], show_bb_id=not hide_bb_ids))

        if pool_grid is None:
            cols = 1 if len(mols) <= 4 else math.ceil(math.sqrt(len(mols)))
        else:
            rows, cols = pool_grid
            total = rows * cols
            if total < len(mols):
                rows = math.ceil(len(mols) / cols)
                total = rows * cols
            if total > len(mols):
                mols.extend([None] * (total - len(mols)))
                legends.extend([""] * (total - len(mols)))

        grid = draw_grid(mols, legends, mols_per_row=cols, cell_size=cell_size)
        pool_images.append(add_header(grid, f"BB{idx} pool", font=title_font))

    if pool_layout == "rows":
        combined = vstack(pool_images, gap=32, divider=True, divider_inset=divider_inset)
    else:
        combined = hstack(pool_images, gap=32, divider=True)

    combined = add_frame(combined, margin=margin, frame_width=frame_width, frame_color=frame_color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f"Saved BB pools PNG: {out_path}")
    if failures:
        print(f"Warning: {failures} BB molecule(s) failed to parse and were left blank.", file=sys.stderr)


def format_score(score: object, fmt: str) -> str:
    value = pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "nan"
    try:
        return format(float(value), fmt)
    except ValueError:
        return str(value)


def score_to_color(score: float, min_score: float, max_score: float) -> Tuple[int, int, int]:
    """Map a docking score to an RGB color on a green-to-red gradient.

    ``min_score`` maps to pure green (0, 255, 0) – strongest binders.
    ``max_score`` maps to pure red (255, 0, 0) – weakest binders.
    Scores outside the range are clamped.
    """

    if min_score >= max_score:
        return (128, 128, 0)
    clamped = max(min_score, min(max_score, score))
    fraction = (clamped - min_score) / (max_score - min_score)
    r = int(255 * fraction)
    g = int(255 * (1.0 - fraction))
    return (r, g, 0)


def render_complete_del(
    df: pd.DataFrame,
    out_path: Path,
    *,
    cell_size: Tuple[int, int],
    score_format: str,
    kekulize: bool,
    sort_numeric: bool,
    score_colors: bool = False,
    score_min: float = -100.0,
    score_max: float = 0.0,
    score_border_width: int = 0,
) -> None:
    title_font = load_font(26, bold=True)
    bb1_ids = sort_ids(df["bb1_id"].tolist(), numeric=sort_numeric)
    bb2_ids = sort_ids(df["bb2_id"].tolist(), numeric=sort_numeric)
    bb3_ids = sort_ids(df["bb3_id"].tolist(), numeric=sort_numeric)
    row_lookup = df.set_index(["bb1_id", "bb2_id", "bb3_id"], drop=False)
    bb1_sections: List[Image.Image] = []
    failures = 0

    for bb1_id in bb1_ids:
        bb2_rows: List[Image.Image] = []
        for bb2_id in bb2_ids:
            mols: List[Optional[Chem.Mol]] = []
            legends: List[str] = []
            scores: List[Optional[float]] = []
            for bb3_id in bb3_ids:
                key = (bb1_id, bb2_id, bb3_id)
                if key not in row_lookup.index:
                    mols.append(None)
                    legends.append("missing")
                    scores.append(None)
                    continue

                row = row_lookup.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                mol = mol_from_smiles(row["smiles"], kekulize=kekulize)
                if mol is None:
                    failures += 1
                mols.append(mol)
                legends.append(make_product_legend(row["docking_score"], row["smiles"], score_format))
                if score_colors:
                    score_val = pd.to_numeric(pd.Series([row["docking_score"]]), errors="coerce").iloc[0]
                    scores.append(float(score_val) if not pd.isna(score_val) else None)
                else:
                    scores.append(None)

            kwargs: dict = {}
            if score_colors:
                kwargs.update(scores=scores, score_min=score_min, score_max=score_max,
                              score_border_width=score_border_width)
            row_grid = draw_grid(mols, legends, mols_per_row=len(bb3_ids), cell_size=cell_size, **kwargs)
            bb2_rows.append(row_grid)

        bb1_section = vstack(bb2_rows, gap=8)
        bb1_sections.append(bb1_section)

    # Arrange BB1 sections into a near-square grid instead of a single tall column.
    cols = math.ceil(math.sqrt(len(bb1_sections)))
    rows: List[Image.Image] = []
    for i in range(0, len(bb1_sections), cols):
        row = hstack(bb1_sections[i : i + cols], gap=30)
        rows.append(row)
    combined = vstack(rows, gap=30)
    final = add_header(combined, "Complete DEL grouped by BB1 / BB2 / BB3", font=title_font, pad=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path)
    print(f"Saved complete DEL PNG: {out_path}")
    if failures:
        print(f"Warning: {failures} product molecule(s) failed to parse and were left blank.", file=sys.stderr)


def render_sampled_products(
    df: pd.DataFrame,
    out_path: Path,
    *,
    grid_shape: Tuple[int, int],
    cell_size: Tuple[int, int],
    score_format: str,
    kekulize: bool,
    seed: int,
    score_colors: bool = False,
    score_min: float = -100.0,
    score_max: float = 0.0,
    score_border_width: int = 0,
) -> None:
    rows, cols = grid_shape
    sample_size = min(rows * cols, len(df))
    sampled = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    mols: List[Optional[Chem.Mol]] = []
    legends: List[str] = []
    scores: List[Optional[float]] = []
    failures = 0

    for _, row in sampled.iterrows():
        mol = mol_from_smiles(row["smiles"], kekulize=kekulize)
        if mol is None:
            failures += 1
        mols.append(mol)
        legends.append(make_product_legend(row["docking_score"], row["smiles"], score_format))
        if score_colors:
            score_val = pd.to_numeric(pd.Series([row["docking_score"]]), errors="coerce").iloc[0]
            scores.append(float(score_val) if not pd.isna(score_val) else None)
        else:
            scores.append(None)

    title_font = load_font(26, bold=True)
    kwargs: dict = {}
    if score_colors:
        kwargs.update(scores=scores, score_min=score_min, score_max=score_max,
                      score_border_width=score_border_width)
    grid = draw_grid(mols, legends, mols_per_row=cols, cell_size=cell_size, **kwargs)
    final = add_header(
        grid,
        f"Random sample of {sample_size} products from {len(df)}-row library",
        font=title_font,
        pad=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path)
    print(f"Saved sampled-products PNG: {out_path}")
    if failures:
        print(f"Warning: {failures} sampled product molecule(s) failed to parse and were left blank.", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"Input CSV does not exist: {args.csv}", file=sys.stderr)
        return 2

    if args.complete_del_threshold < 0:
        print("--complete-del-threshold must be non-negative", file=sys.stderr)
        return 2
    sample_rows, sample_cols = args.sample_grid
    if sample_rows <= 0 or sample_cols <= 0:
        print("--sample-grid ROWS COLS must both be positive integers", file=sys.stderr)
        return 2
    if args.pool_grid is not None:
        pool_rows, pool_cols = args.pool_grid
        if pool_rows <= 0 or pool_cols <= 0:
            print("--pool-grid ROWS COLS must both be positive integers", file=sys.stderr)
            return 2
    if args.margin < 0:
        print("--margin must be non-negative", file=sys.stderr)
        return 2
    if args.frame_width < 0:
        print("--frame-width must be non-negative", file=sys.stderr)
        return 2
    if args.divider_inset < 0:
        print("--divider-inset must be non-negative", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(args.csv)
        validate_input(df, args.csv)
    except Exception as exc:
        print(f"Failed to read/validate input CSV: {exc}", file=sys.stderr)
        return 2

    bb_pools_out, complete_del_out, sample_out = default_output_paths(args)
    render_bb_pools(
        df,
        bb_pools_out,
        cell_size=tuple(args.bb_cell),
        kekulize=args.kekulize,
        sort_numeric=args.sort_numeric,
        pool_layout=args.pool_layout,
        pool_grid=args.pool_grid,
        hide_bb_ids=args.hide_bb_ids,
        margin=args.margin,
        frame_width=args.frame_width,
        frame_color=args.frame_color,
        divider_inset=args.divider_inset,
    )
    score_border_width = args.score_border_width if args.score_colors else 0

    if args.no_complete_del:
        print("Skipping product PNG generation (--no-complete-del).")
    elif len(df) > args.complete_del_threshold:
        print(
            f"Library has {len(df)} rows, above --complete-del-threshold "
            f"{args.complete_del_threshold}; rendering sampled-products PNG instead."
        )
        render_sampled_products(
            df,
            sample_out,
            grid_shape=tuple(args.sample_grid),
            cell_size=tuple(args.cell),
            score_format=args.score_format,
            kekulize=args.kekulize,
            seed=args.sample_seed,
            score_colors=args.score_colors,
            score_min=args.score_min,
            score_max=args.score_max,
            score_border_width=score_border_width,
        )
    else:
        render_complete_del(
            df,
            complete_del_out,
            cell_size=tuple(args.cell),
            score_format=args.score_format,
            kekulize=args.kekulize,
            sort_numeric=args.sort_numeric,
            score_colors=args.score_colors,
            score_min=args.score_min,
            score_max=args.score_max,
            score_border_width=score_border_width,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())