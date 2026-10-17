"""Smoke test for the AuraSync chatbot. Start the API first:  python3 -m uvicorn main:app --reload
Then, in a second terminal:  python3 scripts/test_chat.py [article_id] [customer_id]
Standard library only.
"""
import json
import sys
import urllib.error
import urllib.request

API = "http://127.0.0.1:8000"
ARTICLE = sys.argv[1] if len(sys.argv) > 1 else "0108775015"
CUSTOMER = sys.argv[2] if len(sys.argv) > 2 else None


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except urllib.error.URLError as e:
        sys.exit(f"API not reachable at {API} ({e.reason}). Is uvicorn running?")


code, st = call("GET", "/chat/status")
print(f"provider={st.get('provider', 'openai')}  model={st['model']}  url={st.get('base_url')}  data_dir={st['data_dir']}")
missing = [f"{agent}.{name}" for agent, flags in st["artifacts"].items() for name, ok in flags.items() if not ok]
print("artifacts not loaded:", missing or "none")
if st["artifact_load_errors"]:
    print("load errors:", json.dumps(st["artifact_load_errors"], indent=2))

turns = [
    ("AI Concierge", f"What is article {ARTICLE}, what should I pair it with, and what could I resell it for?"),
    ("Stylist", "I wore it yesterday but it felt a bit tight around the waist. Any advice?"),
]
if CUSTOMER:
    turns.append(("Economist", "How likely is this customer to stop buying, and what does that mean?"))

sid = None
for agent, msg in turns:
    code, r = call("POST", "/chat/message", {"session_id": sid, "agent": agent, "customer_id": CUSTOMER,
                                             "message": f"Selected article context: {ARTICLE}.\nUser request: {msg}"})
    print(f"\n[{agent}] HTTP {code}")
    if code != 200:
        sys.exit(f"Error: {r.get('detail')}")
    sid = r["session_id"]
    print("tools used:", ", ".join(r["tools_used"]) or "none")
    print(r["reply"])
