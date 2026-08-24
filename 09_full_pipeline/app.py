"""Web app for the full pipeline: change the inputs, watch the answer change.

Run it:

    python app.py                 # V1 and V3, trains the feature judge once
    python app.py --variants V1,V2,V3,V4    # all four (needs tensorflow
                                            # and pennylane installed)
    python app.py --fast          # train on a sample, quicker start
    python app.py --port 8010     # if 8000 is taken

Then open http://localhost:8000 in a browser.

The judges are trained once at startup and kept in memory, so each run of
the pipeline afterwards takes well under a second and the page responds
live.

No web framework is used. The server is Python's own `http.server`, so the
project's rule of numpy, pandas, matplotlib, seaborn and scikit-learn only
is not broken by the demo.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pipeline import Pipeline, load_walls

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"
PIPELINE = None                      # filled in at startup

DEFAULTS = {"n_stories": 6, "min_span": 6.0, "max_span": 12.0,
            "unit_scale": 1.0, "merge_tol": 0.6, "lambda_blend": 0.3,
            "variant": "V1", "score_mode": "algo",
            "column_free_zones": [[15, 9, 21, 15]]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_text(encoding="utf-8"),
                       "text/html; charset=utf-8")
        elif self.path == "/api/defaults":
            out = dict(DEFAULTS)
            out["variants"] = sorted(PIPELINE.judges.keys())
            out["score_modes"] = PIPELINE.score_modes
            self._send(200, json.dumps(out), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/run":
            self._send(404, "not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"ok": False, "message": "bad JSON"}),
                       "application/json")
            return

        try:
            plan_name = request.get("plan", "sample")
            walls = request.get("walls") or load_walls(plan_name)
            merge_tol = request.get("merge_tol", 0.6)
            constraints = {
                "n_stories": int(request.get("n_stories", 6)),
                "min_span": float(request.get("min_span", 6.0)),
                "max_span": float(request.get("max_span", 12.0)),
                "unit_scale": float(request.get("unit_scale", 1.0)),
                "merge_tol": ("auto" if merge_tol == "auto"
                              else float(merge_tol)),
                "column_free_zones": request.get("column_free_zones", []),
            }
            if plan_name.startswith("demo"):
                plan = json.loads((HERE.parent / "data" / "demo_plans" /
                                   f"{plan_name}.json").read_text())
                constraints["fixed_columns"] = plan.get(
                    "annotated_columns", [])
                constraints["n_stories"] = int(request.get(
                    "n_stories", plan.get("n_stories", 6)))
            if constraints["min_span"] >= constraints["max_span"]:
                raise ValueError("minimum span must be smaller than maximum span")
            result = PIPELINE.run(walls, constraints,
                                  float(request.get("lambda_blend", 0.3)),
                                  variant=request.get("variant", "V1"),
                                  score_mode=request.get("score_mode",
                                                         "algo"))
        except Exception as exc:                       # report, do not crash
            result = {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
        self._send(200, json.dumps(result), "application/json")

    def log_message(self, fmt, *args):                 # quieter console
        if "api/run" in (args[0] if args else ""):
            sys.stderr.write(f"  run: {args[0]}\n")


def main():
    global PIPELINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--variants", default="V1,V3",
                    help="comma list of V1,V2,V3,V4 to prepare at startup")
    ap.add_argument("--fast", action="store_true",
                    help="train on a sample of the corpus for a quick start")
    args = ap.parse_args()

    print("Structural arrangement pipeline, web demo")
    print("-" * 46)
    if args.fast:
        print("fast mode: judges trained on a sample, so the ranking can")
        print("differ slightly from the notebooks\n")
    PIPELINE = Pipeline(variants=tuple(args.variants.split(",")),
                        n_rows=3000 if args.fast else None)
    print(f"\njudges ready in {PIPELINE.trained_seconds:.0f}s")
    print(f"open http://localhost:{args.port} in your browser")
    print("press Ctrl+C to stop\n")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
