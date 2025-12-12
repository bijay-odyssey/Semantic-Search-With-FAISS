from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from search import search, load_index
from build_index import build_index
from add_documents import add_documents

app = FastAPI()


# UI
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>FAISS Semantic Search</title></head>
    <body style="font-family: Arial; max-width: 700px; margin:auto;">

        <h1>FAISS Semantic Search</h1>

        <h2>1. Rebuild Index</h2>
        <p>This completely rebuilds the index with the documents you enter below.</p>
        <form action="/rebuild_index" method="post">
            <textarea name="docs" rows="6" style="width:100%"
                placeholder="Enter documents, one per line"></textarea><br><br>
            <button type="submit">Rebuild Index</button>
        </form>

        <h2>2. Add Documents</h2>
        <form action="/add_docs" method="post">
            <textarea name="docs" rows="6" style="width:100%"
                placeholder="Enter documents to add to index"></textarea><br><br>
            <button type="submit">Add Documents</button>
        </form>

        <h2>3. Search</h2>
        <form action="/search_ui" method="get">
            <input name="q" style="width:80%;" placeholder="Enter your query">
            <button type="submit">Search</button>
        </form>

    </body>
    </html>
    """


# Rebuild Index
@app.post("/rebuild_index", response_class=HTMLResponse)
def rebuild_index_api(docs: str = Form(...)):
    docs_list = [d.strip() for d in docs.split("\n") if d.strip()]

    build_index(docs_list)   
    load_index()              

    return "<h3>Index rebuilt successfully!</h3><a href='/'>Back</a>"


# Add indexing Documents
@app.post("/add_docs", response_class=HTMLResponse)
def add_docs_api(docs: str = Form(...)):
    docs_list = [d.strip() for d in docs.split("\n") if d.strip()]

    add_documents(docs_list)   
    load_index()              

    return "<h3>Documents added!</h3><a href='/'>Back</a>"


# Web Search
@app.get("/search_ui", response_class=HTMLResponse)
def search_ui(q: str):

    results = search(q, k=5)
    if "error" in results[0]:
        return f"<h3>{results[0]['error']}</h3><br><a href='/'>Back</a>"

    html = f"<h2>Search Results for: {q}</h2><ul>"
    for r in results:
        html += f"<li><b>{r['score']:.4f}</b> — {r['text']}</li>"
    html += "</ul><br><a href='/'>Back</a>"

    return html


# JSON Search API
@app.get("/search")
def search_api(q: str, k: int = 5):
    return search(q, k)
