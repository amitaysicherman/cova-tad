"""
paper/build_pdf.py

Compiles paper/main_paper.md and paper/appendix.md into publication-quality PDFs
with full LaTeX math typesetting (via KaTeX) and ICLR academic formatting.
"""

import os
import re
import markdown
from playwright.sync_api import sync_playwright

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <!-- KaTeX for exact LaTeX math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{
                delimiters: [
                    {{left: '$$', right: '$$', display: true}},
                    {{left: '$', right: '$', display: false}},
                    {{left: '\\\\[', right: '\\\\]', display: true}},
                    {{left: '\\\\(', right: '\\\\)', display: false}}
                ],
                throwOnError: false
            }});"></script>
    <style>
        @page {{
            size: letter;
            margin: 20mm 20mm 25mm 20mm;
            @bottom-center {{
                content: counter(page);
                font-size: 9pt;
                font-family: 'Times New Roman', Times, serif;
            }}
        }}
        body {{
            font-family: 'Times New Roman', Times, 'Nimbus Roman No9 L', serif;
            font-size: 10.5pt;
            line-height: 1.5;
            color: #111111;
            margin: 0 auto;
            max-width: 850px;
            background: #ffffff;
        }}
        h1 {{
            font-size: 18pt;
            text-align: center;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 8px;
            line-height: 1.3;
        }}
        .author {{
            text-align: center;
            font-size: 11pt;
            margin-bottom: 25px;
            color: #333333;
        }}
        .venue {{
            text-align: center;
            font-size: 10pt;
            font-style: italic;
            color: #555555;
            margin-top: -15px;
            margin-bottom: 25px;
        }}
        h2 {{
            font-size: 13pt;
            font-weight: bold;
            border-bottom: 1.5px solid #222222;
            padding-bottom: 4px;
            margin-top: 25px;
            margin-bottom: 12px;
        }}
        h3 {{
            font-size: 11.5pt;
            font-weight: bold;
            margin-top: 18px;
            margin-bottom: 8px;
        }}
        h4 {{
            font-size: 10.5pt;
            font-weight: bold;
            font-style: italic;
            margin-top: 14px;
            margin-bottom: 6px;
        }}
        p {{
            margin-bottom: 10px;
            text-align: justify;
        }}
        /* Academic Booktabs Table Style */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 8.5pt;
            line-height: 1.3;
            page-break-inside: avoid;
        }}
        th, td {{
            padding: 5px 6px;
            text-align: right;
        }}
        th:first-child, td:first-child,
        th:nth-child(2), td:nth-child(2) {{
            text-align: left;
        }}
        thead tr:first-child th {{
            border-top: 2px solid #111111;
            border-bottom: 1px solid #111111;
            font-weight: bold;
        }}
        tbody tr:last-child td {{
            border-bottom: 2px solid #111111;
        }}
        tbody tr:hover {{
            background-color: #f8fafc;
        }}
        strong {{
            font-weight: bold;
        }}
        em {{
            font-style: italic;
        }}
        blockquote {{
            border-left: 3px solid #2563eb;
            margin: 15px 0;
            padding: 8px 15px;
            background-color: #f0f7ff;
            font-size: 10pt;
        }}
        code {{
            font-family: 'Courier New', Courier, monospace;
            font-size: 9pt;
            background: #f1f5f9;
            padding: 1px 4px;
            border-radius: 3px;
        }}
        pre {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 9pt;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
        }}
        .katex-display {{
            margin: 12px 0 !important;
        }}
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>
"""


def convert_md_to_pdf(md_path: str, pdf_path: str, title: str):
    print(f"Reading {md_path} ...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-process math equations so markdown processor does not corrupt underscores or stars
    math_blocks = []
    def save_display_math(match):
        math_blocks.append(match.group(0))
        return f"<!--MATH_BLOCK_{len(math_blocks)-1}-->"

    def save_inline_math(match):
        math_blocks.append(match.group(0))
        return f"<!--MATH_INLINE_{len(math_blocks)-1}-->"

    # Save $$ ... $$
    md_text = re.sub(r"\$\$(.*?)\$\$", save_display_math, md_text, flags=re.DOTALL)
    # Save $ ... $
    md_text = re.sub(r"\$(.*?)\$", save_inline_math, md_text)

    # Convert markdown to HTML
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    # Restore math
    for idx, mb in enumerate(math_blocks):
        html_body = html_body.replace(f"&lt;!--MATH_BLOCK_{idx}--&gt;", mb)
        html_body = html_body.replace(f"&lt;!--MATH_INLINE_{idx}--&gt;", mb)
        html_body = html_body.replace(f"<!--MATH_BLOCK_{idx}-->", mb)
        html_body = html_body.replace(f"<!--MATH_INLINE_{idx}-->", mb)

    # Format title & author headers cleanly if present
    full_html = HTML_TEMPLATE.format(title=title, content=html_body)

    tmp_html_path = md_path.replace(".md", ".tmp.html")
    with open(tmp_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Rendering PDF with Chromium via Playwright: {pdf_path} ...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{os.path.abspath(tmp_html_path)}", wait_until="networkidle")
        # Give KaTeX a brief moment to render math typesetting
        page.wait_for_timeout(1000)
        page.pdf(
            path=pdf_path,
            format="Letter",
            print_background=True,
            margin={"top": "20mm", "bottom": "25mm", "left": "20mm", "right": "20mm"}
        )
        browser.close()

    if os.path.exists(tmp_html_path):
        os.remove(tmp_html_path)

    print(f"--> Successfully created {pdf_path} ({os.path.getsize(pdf_path) // 1024} KB)")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Compile main_paper.pdf
    main_md = os.path.join(base_dir, "main_paper.md")
    main_pdf = os.path.join(base_dir, "main_paper.pdf")
    convert_md_to_pdf(main_md, main_pdf, title="FE-TAD: Main Paper")

    # 2. Compile appendix.pdf
    app_md = os.path.join(base_dir, "appendix.md")
    app_pdf = os.path.join(base_dir, "appendix.pdf")
    convert_md_to_pdf(app_md, app_pdf, title="FE-TAD: Supplementary Materials")
