import re
from typing import Dict, Any, Optional, List

class DeterministicCalculator:
    """
    Deterministic Arithmetic & Financial Calculation Engine.
    Handles:
      - GST / Tax calculation (e.g. price + 18% GST)
      - Discounts (e.g. 15% discount on $200)
      - Totals and sums across extracted numbers
      - Differences and percentage changes
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

    @classmethod
    def evaluate_query_math(cls, query: str, context_text: str) -> Optional[str]:
        """
        Attempts to compute mathematical answer if query specifies calculation
        using numbers extracted from query and retrieved context.
        """
        q_lower = query.lower()

        # 1. GST Calculation
        gst_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:gst|tax)', q_lower)
        rate = float(gst_match.group(1)) if gst_match else 18.0 if "gst" in q_lower else None

        # Look for base amount in query or context (excluding the rate itself)
        nums = [float(n.replace(",", "")) for n in re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', query)]
        if rate is not None:
            base_candidates = [n for n in nums if abs(n - rate) > 0.001]
        else:
            base_candidates = nums

        if not base_candidates:
            # Check context for prices
            ctx_nums = [float(n.replace(",", "")) for n in re.findall(r'\b(?:₹|\$|usd|inr)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', context_text, re.IGNORECASE)]
            if rate is not None:
                base_candidates = [n for n in ctx_nums if abs(n - rate) > 0.001]
            else:
                base_candidates = ctx_nums

        if rate is not None and base_candidates:
            base = base_candidates[0]
            res = cls.calculate_gst(base, rate)
            return (
                f"Calculation Result:\n"
                f"• Base Amount: {res['base_amount']:,.2f}\n"
                f"• GST ({res['gst_rate']}%): {res['gst_amount']:,.2f}\n"
                f"• Total Amount (inclusive of GST): {res['total_with_gst']:,.2f}"
            )

        # 2. Discount Calculation
        disc_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*discount', q_lower)
        if disc_match:
            d_rate = float(disc_match.group(1))
            disc_base_candidates = [n for n in nums if abs(n - d_rate) > 0.001]
            if disc_base_candidates:
                base = disc_base_candidates[0]
                res = cls.calculate_discount(base, d_rate)
                return (
                    f"Calculation Result:\n"
                    f"• Original Price: {res['original_price']:,.2f}\n"
                    f"• Discount ({res['discount_percent']}%): -{res['discount_amount']:,.2f}\n"
                    f"• Final Discounted Price: {res['final_price']:,.2f}"
                )

        return None
