import re
from typing import List, Dict, Any, Optional, Tuple

def parse_typed_value(raw_val: str) -> Any:
    """
    Parses a string into typed values (float, int, currency, quantity, date, or clean text)
    to enable future deterministic computation without LLM.
    Returns structured dict for computable attributes or clean primitive.
    """
    s = raw_val.strip()
    if not s:
        return ""

    # Currency with symbol or ISO code (e.g. $50, USD 50, €25.50, GBP 100, 50 USD)
    curr_match = re.match(r'^([\$\€\£\¥₹]|USD|EUR|GBP|INR)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*([\$\€\£\¥₹]|USD|EUR|GBP|INR)?$', s, re.IGNORECASE)
    if curr_match and (curr_match.group(1) or curr_match.group(3)):
        sym = curr_match.group(1) or curr_match.group(3)
        num_str = curr_match.group(2).replace(",", "")
        try:
            val = float(num_str) if "." in num_str else int(num_str)
            return {"raw": s, "numeric": val, "unit": sym.upper(), "type": "CURRENCY"}
        except ValueError:
            pass

    # Quantity with unit (e.g. 50 GB, 5 TB, 24 days, 10 users, 15 minutes)
    qty_match = re.match(r'^([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s+([A-Za-z]+)$', s)
    if qty_match:
        num_str = qty_match.group(1).replace(",", "")
        try:
            val = float(num_str) if "." in num_str else int(num_str)
            unit = qty_match.group(2)
            return {"raw": s, "numeric": val, "unit": unit, "type": "QUANTITY"}
        except ValueError:
            pass

    # Percentage (e.g. 25%, 5.5%)
    pct_match = re.match(r'^([0-9]+(?:\.[0-9]+)?)\s*\%$', s)
    if pct_match:
        try:
            val = float(pct_match.group(1)) / 100.0
            return {"raw": s, "numeric": val, "unit": "%", "type": "PERCENTAGE"}
        except ValueError:
            pass

    # Plain numeric
    num_clean = s.replace(",", "")
    if re.match(r'^-?[0-9]+(?:\.[0-9]+)?$', num_clean):
        try:
            val = float(num_clean) if "." in num_clean else int(num_clean)
            return {"raw": s, "numeric": val, "type": "NUMERIC"}
        except ValueError:
            pass

    return s

class TableProcessor:
    """
    Domain-agnostic table extractor and row-level decomposer.
    Decomposes markdown/grid tables into row sub-chunks preserving header-value bindings,
    cell coordinates, and typed values for future deterministic computation.
    """

    @staticmethod
    def is_table_text(text: str) -> bool:
        """Determines if a block of text represents a markdown table."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        pipe_lines = sum(1 for l in lines if l.startswith("|") and l.endswith("|") and l.count("|") >= 2)
        return pipe_lines >= 2 and (pipe_lines / max(1, len(lines))) >= 0.50

    @classmethod
    def parse_markdown_table(cls, text: str) -> Optional[Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]]]]:
        """
        Parses markdown pipe table into (headers, rows_str, rows_typed).
        Returns None if not a valid table.
        """
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
        if len(table_lines) < 2:
            return None

        # Extract headers from first row
        header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
        headers = [h for h in header_cells if h]
        if not headers:
            return None

        rows_str: List[Dict[str, str]] = []
        rows_typed: List[Dict[str, Any]] = []

        start_row = 1
        # Check if row 1 is a separator line (e.g. |---|---|)
        if len(table_lines) > 1 and re.match(r'^\|(?:\s*:?-+:?\s*\|)+$', table_lines[1]):
            start_row = 2

        for r_line in table_lines[start_row:]:
            cells = [c.strip() for c in r_line.strip("|").split("|")]
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            cells = cells[:len(headers)]

            row_dict = {}
            typed_dict = {}
            for h, val in zip(headers, cells):
                row_dict[h] = val
                typed_dict[h] = parse_typed_value(val)

            rows_str.append(row_dict)
            rows_typed.append(typed_dict)

        return headers, rows_str, rows_typed

    @classmethod
    def format_row_text(cls, section_title: str, headers: List[str], row_data: Dict[str, str]) -> str:
        """
        Formats a single row into an entity-attribute sentence for high-precision retrieval:
        e.g., [Table: Subscription Plans] Plan: Pro | Price: $50 | Users: 20
        """
        bindings = [f"{h}: {row_data.get(h, '')}" for h in headers if h in row_data]
        prefix = f"[Table: {section_title}] " if section_title else ""
        return prefix + " | ".join(bindings)
