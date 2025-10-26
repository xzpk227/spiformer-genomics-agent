import os
import json
import base64
from io import BytesIO
from datetime import datetime
from langchain.tools import tool

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


REPORTS_BUCKET = os.environ.get("REPORTS_BUCKET")
PRESIGNED_URL_EXPIRY = 60 * 60 * 24  # 24 hours


def _get_s3_client():
    return boto3.client("s3")


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _upload_to_s3(html: str, key: str) -> str:
    """Upload HTML report to S3 and return a presigned URL."""
    s3 = _get_s3_client()
    s3.put_object(
        Bucket=REPORTS_BUCKET,
        Key=key,
        Body=html.encode("utf-8"),
        ContentType="text/html",
    )
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": REPORTS_BUCKET, "Key": key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )
    return url


@tool
def generate_report(analysis_data: str) -> str:
    """
    Generate an HTML report with visualizations from analysis results.
    Input should be a JSON string with keys: gene, disease, literature, databases, enrichment, splicing.
    If REPORTS_BUCKET is set, uploads to S3 and returns a presigned URL (valid 24h).
    Otherwise saves to /app/reports and returns the local path.
    """
    try:
        try:
            data = json.loads(analysis_data)
        except json.JSONDecodeError:
            data = {"summary": analysis_data}

        gene = data.get("gene", "Unknown Gene")
        disease = data.get("disease", "Unknown Disease")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{gene}_{disease}_{timestamp}.html"

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

        if REPORTS_BUCKET:
            url = _upload_to_s3(html, f"reports/{filename}")
            return f"Report generated. Download link (valid 24h): {url}"
        else:
            reports_dir = "/app/reports"
            os.makedirs(reports_dir, exist_ok=True)
            report_path = os.path.join(reports_dir, filename)
            with open(report_path, "w") as f:
                f.write(html)
            return f"Report generated: {report_path}"

    except (BotoCoreError, ClientError) as e:
        return f"Report generation error (S3): {str(e)}"
    except Exception as e:
        return f"Report generation error: {str(e)}"
