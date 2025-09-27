import os
import json
import base64
from io import BytesIO
from datetime import datetime
from langchain.tools import tool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


REPORTS_DIR = "/app/reports"


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@tool
def generate_report(analysis_data: str) -> str:
    """
    Generate an HTML report with visualizations from analysis results.
    Input should be a JSON string with keys: gene, disease, literature, databases, enrichment, splicing.
    Returns the path to the generated report.
    """
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        try:
            data = json.loads(analysis_data)
        except json.JSONDecodeError:
            # Treat as plain text summary
            data = {"summary": analysis_data}

        gene = data.get("gene", "Unknown Gene")
        disease = data.get("disease", "Unknown Disease")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORTS_DIR, f"report_{gene}_{disease}_{timestamp}.html")

        # Generate enrichment bar chart if data available
        enrichment_img = ""
        if "enrichment_terms" in data and data["enrichment_terms"]:
            terms = data["enrichment_terms"][:8]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                x=[t.get("pval", 0) for t in terms],
                y=[t.get("term", "")[:50] for t in terms],
                palette="viridis", ax=ax
            )
            ax.set_xlabel("-log10(adj p-value)")
            ax.set_title(f"Pathway Enrichment - {gene}")
            enrichment_img = _fig_to_base64(fig)
            plt.close(fig)

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Genomics Report: {gene} in {disease}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ color: #2c5f8a; }} h2 {{ color: #3a7abf; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; white-space: pre-wrap; }}
  .section {{ margin-bottom: 32px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
</style>
</head>
<body>
<h1>Genomics AI Research Report</h1>
<p><strong>Gene:</strong> {gene} &nbsp;|&nbsp; <strong>Disease:</strong> {disease} &nbsp;|&nbsp; <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<div class="section">
  <h2>Literature Summary</h2>
  <pre>{data.get("literature", "No literature data available.")}</pre>
</div>

<div class="section">
  <h2>Database Annotations</h2>
  <pre>{data.get("databases", "No database data available.")}</pre>
</div>

<div class="section">
  <h2>Splicing Analysis</h2>
  <pre>{data.get("splicing", "No splicing data available.")}</pre>
</div>

<div class="section">
  <h2>Pathway Enrichment</h2>
  <pre>{data.get("enrichment", "No enrichment data available.")}</pre>
  {"<img src='data:image/png;base64," + enrichment_img + "' alt='Enrichment chart'/>" if enrichment_img else ""}
</div>

<div class="section">
  <h2>AI Synthesis</h2>
  <pre>{data.get("summary", "")}</pre>
</div>
</body>
</html>"""

        with open(report_path, "w") as f:
            f.write(html)

        return f"Report generated: {report_path}"
    except Exception as e:
        return f"Report generation error: {str(e)}"
