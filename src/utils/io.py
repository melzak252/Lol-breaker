from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV file and print its row count.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(path)
    print(f"Loaded {path} with {len(df):,} rows.")
    return df


def save_markdown(content: str, file_path: str | Path) -> None:
    """Save markdown content, creating parent directories if needed.

    Args:
        content: Markdown text to write.
        file_path: Target markdown path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Saved results to {path}")


def generate_obsidian_header(
    doc_type: str = "research_log",
    tags: Optional[Sequence[str]] = None,
    project: str = "EnsembleLegends",
) -> str:
    """Generate a YAML header for Obsidian documents.

    Args:
        doc_type: Value for the `type` YAML field.
        tags: Optional sequence of Obsidian tags.
        project: Project name written to metadata.

    Returns:
        YAML front matter block as a string.
    """
    tags_str = f"[{', '.join(tags)}]" if tags else "[]"
    header = f"""---
type: {doc_type}
tags: {tags_str}
project: {project}
date: {datetime.now().strftime('%Y-%m-%d')}
---
"""
    return header
