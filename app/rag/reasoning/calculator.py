"""
app/rag/reasoning/calculator.py
Deterministic Arithmetic, Financial, and Aggregation Calculation Engine.
Directly computes:
  - GST / VAT / Tax (e.g. ₹10,000 with 18% GST)
  - Discounts (e.g. 15% discount on ₹200)
  - Totals / Sums (e.g. sum of 100, 250, 350)
  - Averages (e.g. average of 20, 40, 60)
  - Ratios (e.g. ratio of 100 to 200 -> 1:2)
  - Basic arithmetic (+, -, *, /)
All computations are strictly non-generative, deterministic, and evidence-traceable.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple


class DeterministicCalculator:
    """
    Deterministic Arithmetic & Financial Calculation Engine.
    """

    @staticmethod
    def calculate_gst(base_amount: float, gst_rate: float = 18.0) -> Dict[str, float]:
        gst_amount = (base_amount * gst_rate) / 100.0
        total = base_amount + gst_amount
        return {
            "base_amount": round(base_amount, 2),
            "gst_rate": gst_rate,
            "gst_amount": round(gst_amount, 2),
            "total_with_gst": round(total, 2)
        }

    @staticmethod
    def calculate_discount(original_price: float, discount_percent: float) -> Dict[str, float]:
        discount_amount = (original_price * discount_percent) / 100.0
        final_price = original_price - discount_amount
        return {
            "original_price": round(original_price, 2),
            "discount_percent": discount_percent,
            "discount_amount": round(discount_amount, 2),
            "final_price": round(final_price, 2)
        }

    @staticmethod
    def calculate_average(numbers: List[float]) -> Optional[Dict[str, Any]]:
        if not numbers:
            return None
        total = sum(numbers)
        avg = total / len(numbers)
        return {
            "numbers": numbers,
            "count": len(numbers),
            "sum": round(total, 2),
            "average": round(avg, 2)
        }

    @staticmethod
    def calculate_ratio(num1: float, num2: float) -> Optional[Dict[str, Any]]:
        if num2 == 0:
            return None
        # Simplify ratio if whole numbers
        if num1.is_integer() and num2.is_integer():
            g = math.gcd(int(num1), int(num2))
            simp_a, simp_b = int(num1) // g, int(num2) // g
            ratio_str = f"{simp_a}:{simp_b}"
        else:
            ratio_str = f"{round(num1 / num2, 2)}:1"
        return {
            "num1": num1,
            "num2": num2,
            "ratio_str": ratio_str,
            "decimal_ratio": round(num1 / num2, 4)
        }

    @classmethod
    def evaluate_query_math(cls, query: str, context_text: str = "") -> Optional[str]:
        """
        Attempts to compute mathematical answer if query specifies calculation
        using numbers extracted from query and/or retrieved context.
        """
        q_lower = query.lower()

        # Extract numbers from query
        query_nums = [float(n.replace(",", "")) for n in re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', query)]

        # 1. GST / Tax Calculation
        if "gst" in q_lower or "tax" in q_lower:
            gst_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:gst|tax)?', q_lower)
            rate = float(gst_match.group(1)) if gst_match else 18.0

            base_candidates = [n for n in query_nums if abs(n - rate) > 0.001]
            if not base_candidates and context_text:
                ctx_nums = [float(n.replace(",", "")) for n in re.findall(r'(?:₹|\$|usd|inr)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', context_text, re.IGNORECASE)]
                base_candidates = [n for n in ctx_nums if abs(n - rate) > 0.001]

            if base_candidates:
                base = base_candidates[0]
                res = cls.calculate_gst(base, rate)
                return (
                    f"Calculation Result:\n"
                    f"• Base Amount: ₹{res['base_amount']:,.2f}\n"
                    f"• GST ({res['gst_rate']}%): ₹{res['gst_amount']:,.2f}\n"
                    f"• Total Amount (inclusive of GST): ₹{res['total_with_gst']:,.2f}"
                )

        # 2. Percentage of a number (e.g., "what is 20% of 500?")
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(\d+(?:,\d{3})*(?:\.\d+)?)', q_lower)
        if pct_match:
            pct_val = float(pct_match.group(1))
            val = float(pct_match.group(2).replace(",", ""))
            calc = (val * pct_val) / 100.0
            return (
                f"Calculation Result:\n"
                f"• {pct_val}% of {val:,.2f} = {calc:,.2f}"
            )

        # 3. Discount Calculation
        disc_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*discount', q_lower)
        if disc_match:
            d_rate = float(disc_match.group(1))
            disc_base = [n for n in query_nums if abs(n - d_rate) > 0.001]
            if disc_base:
                base = disc_base[0]
                res = cls.calculate_discount(base, d_rate)
                return (
                    f"Calculation Result:\n"
                    f"• Original Price: {res['original_price']:,.2f}\n"
                    f"• Discount ({res['discount_percent']}%): -{res['discount_amount']:,.2f}\n"
                    f"• Final Discounted Price: {res['final_price']:,.2f}"
                )

        # 4. Average / Mean Calculation
        if "average" in q_lower or "mean" in q_lower:
            if len(query_nums) >= 2:
                res = cls.calculate_average(query_nums)
                if res:
                    return (
                        f"Calculation Result:\n"
                        f"• Numbers: {', '.join(str(n) for n in res['numbers'])}\n"
                        f"• Total Sum: {res['sum']:,.2f}\n"
                        f"• Count: {res['count']}\n"
                        f"• Average: {res['average']:,.2f}"
                    )

        # 5. Ratio Calculation (e.g. "ratio of 10 to 50")
        ratio_match = re.search(r'ratio\s+of\s+(\d+(?:\.\d+)?)\s+(?:to|and)\s+(\d+(?:\.\d+)?)', q_lower)
        if ratio_match:
            n1 = float(ratio_match.group(1))
            n2 = float(ratio_match.group(2))
            res = cls.calculate_ratio(n1, n2)
            if res:
                return (
                    f"Calculation Result:\n"
                    f"• Ratio of {n1} to {n2}: {res['ratio_str']} ({res['decimal_ratio']})"
                )

        # 6. Basic Arithmetic expressions (e.g., "150 + 250", "40 * 12", "1000 / 4")
        arith_match = re.search(r'\b(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\b', q_lower)
        if arith_match and not any(k in q_lower for k in ("gst", "tax", "discount")):
            n1 = float(arith_match.group(1))
            op = arith_match.group(2)
            n2 = float(arith_match.group(3))
            ans = 0.0
            if op == "+":
                ans = n1 + n2
            elif op == "-":
                ans = n1 - n2
            elif op == "*":
                ans = n1 * n2
            elif op == "/":
                if n2 != 0:
                    ans = n1 / n2
                else:
                    return "Error: Division by zero."
            return f"Calculation Result:\n• {n1} {op} {n2} = {round(ans, 4)}"

        return None
