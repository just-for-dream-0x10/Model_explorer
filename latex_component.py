import streamlit.components.v1 as components
import base64


def latex_component(formula, display=True, height=50):
    """
    创建一个能够正确渲染LaTeX公式的组件
    """
    if display:
        formula_html = f"$$ {formula} $$"
    else:
        formula_html = f"$ {formula} $"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0KOVEMeaKrumfonJpasSUgnkYtGIYLpAkH5EVWNeDNJg8jVnbYiVT" crossorigin="anonymous">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzYCPaOoIrBvzqhzd2Fh+R7d4QG4G4G4G4G4" crossorigin="anonymous"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
        <style>
            body {{
                margin: 0;
                padding: 10px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: {height}px;
            }}
            .katex-display {{
                margin: 0;
                font-size: 1.1em;
            }}
        </style>
    </head>
    <body>
        <div id="formula">{formula_html}</div>
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                renderMathInElement(document.body, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '$', right: '$', display: false}},
                        {{left: '\\\\[', right: '\\\\]', display: true}},
                        {{left: '\\\\(', right: '\\\\)', display: false}}
                    ],
                    throwOnError: false
                }});
            }});
        </script>
    </body>
    </html>
    """

    # 使用iframe来避免CSS冲突
    components.iframe(
        src=f"data:text/html;base64,{base64.b64encode(html_code.encode()).decode()}",
        height=height,
        scrolling=False,
    )


def latex_box_component(formula, title="", height=80):
    """
    创建带样式的LaTeX公式框组件
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0KOVEMeaKrumfonJpasSUgnkYtGIYLpAkH5EVWNeDNJg8jVnbYiVT" crossorigin="anonymous">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzYCPaOoIrBvzqhzd2Fh+R7d4QG4G4G4G4G4" crossorigin="anonymous"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: #e3f2fd;
                border: 1px solid #2196F3;
                border-radius: 8px;
                text-align: center;
                font-size: 18px;
            }}
            h4 {{
                margin-top: 0;
                margin-bottom: 15px;
                color: #1976D2;
            }}
            .katex-display {{
                margin: 0;
                font-size: 1.2em;
            }}
        </style>
    </head>
    <body>
        {'<h4>' + title + '</h4>' if title else ''}
        <div>$$ {formula} $$</div>
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                renderMathInElement(document.body, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '$', right: '$', display: false}},
                        {{left: '\\\\[', right: '\\\\]', display: true}},
                        {{left: '\\\\(', right: '\\\\)', display: false}}
                    ],
                    throwOnError: false
                }});
            }});
        </script>
    </body>
    </html>
    """

    components.iframe(
        src=f"data:text/html;base64,{base64.b64encode(html_code.encode()).decode()}",
        height=height,
        scrolling=False,
    )
