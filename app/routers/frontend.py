"""Serves the project's wardrobe page at /app with a small bridge script added.

The bridge leaves the page's own code untouched. It wraps window.fetch so that chat requests
also carry the user's wardrobe and selected piece (the page itself only sent the selected ID),
and it appends the list of AuraSync models that answered under each chat reply.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import config

router = APIRouter(tags=["Frontend"])

BRIDGE_JS = r"""
<script>
/* AuraSync bridge (added by the API at /app) */
(function () {
  var origFetch = window.fetch.bind(window);
  function wardrobeIds() {
    try { if (typeof wardrobe !== 'undefined' && Array.isArray(wardrobe)) return wardrobe.slice(0, 200).map(String); } catch (e) {}
    try { return JSON.parse(localStorage.getItem('aurasyncWardrobe') || '[]').map(String); } catch (e) { return []; }
  }
  function selectedId() {
    try { return (typeof selected !== 'undefined' && selected) ? String(selected) : null; } catch (e) { return null; }
  }
  window.fetch = async function (input, init) {
    var url = typeof input === 'string' ? input : (input && input.url) || '';
    if (url.indexOf('/chat/message') === -1 || !init || !init.body) return origFetch(input, init);
    try {
      var body = JSON.parse(init.body);
      body.wardrobe = wardrobeIds();
      body.selected_article_id = selectedId();
      init = Object.assign({}, init, { body: JSON.stringify(body) });
    } catch (e) {}
    var res = await origFetch(input, init);
    try {
      var data = await res.clone().json();
      if (res.ok && data && Array.isArray(data.tools_used) && data.tools_used.length) {
        data.reply = (data.reply || '') + '\n\n— used: ' + data.tools_used.join(', ');
        return new Response(JSON.stringify(data), { status: res.status, statusText: res.statusText,
                                                    headers: { 'Content-Type': 'application/json' } });
      }
    } catch (e) {}
    return res;
  };
})();
</script>
"""

NOT_FOUND = """<!doctype html><meta charset="utf-8"><title>AuraSync</title>
<body style="font-family:-apple-system,sans-serif;max-width:560px;margin:60px auto;line-height:1.5">
<h2>Wardrobe page not found</h2>
<p>The API looked for <code>frontend/*.html</code> next to <code>main.py</code> and in
<code>{base}/*/frontend/</code>.</p>
<p>Copy your project's <code>frontend</code> folder next to <code>main.py</code>, or set
<code>AURASYNC_FRONTEND_PATH</code> in <code>.env</code> to the HTML file, then restart.</p>
<p><a href="/docs">Open the API docs</a></p></body>"""


@router.get("/app", response_class=HTMLResponse, include_in_schema=False)
def wardrobe_app():
    path = config.FRONTEND_PATH
    if path is None or not path.exists():
        return HTMLResponse(NOT_FOUND.format(base=config.BASE_DIR), status_code=404)
    html = path.read_text(encoding="utf-8", errors="replace")
    if "</body>" in html:
        head, tail = html.rsplit("</body>", 1)
        html = head + BRIDGE_JS + "</body>" + tail
    else:
        html += BRIDGE_JS
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@router.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/app")
