"""
Exercise demo/app.py's real control flow against a stubbed Streamlit.

Run:  .venv/bin/python tests/test_demo_smoke.py

Streamlit executes the script over a websocket, so an HTTP 200 from the server
proves only that the static shell loads — an exception inside main() never
reaches the HTTP layer. This stub runs main() directly for every traffic-source
branch, which is what actually catches broken widget flow, bad variable
scoping, and shape errors.

Needs a trained checkpoint at checkpoints/world_model_best.pt; skips cleanly
if one is not present.
"""
import contextlib
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CALLS: list[str] = []
CHOICE = 0
UPLOAD = None


class _Stop(Exception):
    """Stands in for st.stop(), which raises to halt the script."""


def _rec(name, ret=None):
    def f(*a, **k):
        CALLS.append(name)
        return ret
    return f


class _Box:
    """Serves as a column, a sidebar, and a context manager."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return _rec(name)


def _install_stub():
    st = types.ModuleType("streamlit")
    for n in ("set_page_config", "title", "caption", "header", "subheader",
              "markdown", "error", "success", "info", "warning", "write",
              "line_chart", "bar_chart", "dataframe", "metric"):
        setattr(st, n, _rec(n))

    st.text_input = lambda label, val="", **k: val
    st.slider = lambda label, lo, hi, default=None, step=None, **k: (
        default if default is not None else lo)
    st.select_slider = lambda label, opts, value=None, **k: (
        value if value is not None else opts[0])
    st.number_input = lambda label, lo, hi, default=None, step=None, **k: (
        default if default is not None else lo)
    st.radio = lambda label, opts, **k: opts[CHOICE]
    st.file_uploader = lambda *a, **k: UPLOAD
    st.columns = lambda n, **k: [_Box() for _ in range(n if isinstance(n, int) else len(n))]
    st.sidebar = _Box()

    def _stop():
        CALLS.append("stop")
        raise _Stop()
    st.stop = _stop

    @contextlib.contextmanager
    def _ctx(*a, **k):
        yield _Box()
    st.spinner = _ctx
    st.expander = _ctx
    st.cache_resource = lambda f=None, **k: (f if f else (lambda g: g))
    st.cache_data = lambda f=None, **k: (f if f else (lambda g: g))
    sys.modules["streamlit"] = st


def _load_app():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "demoapp", os.path.join(ROOT, "demo", "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeUpload:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self._b = open(path, "rb").read()
    def getvalue(self):
        return self._b


def run(label, choice, upload=None):
    global CHOICE, UPLOAD
    CHOICE, UPLOAD = choice, upload
    CALLS.clear()
    try:
        _load_app().main()
        done = True
    except _Stop:
        done = False
    charts = CALLS.count("line_chart") + CALLS.count("bar_chart")
    print(f"  {label:<34} {'completed' if done else 'stopped'}  "
          f"charts={charts} warnings={CALLS.count('warning')} errors={CALLS.count('error')}")
    return done, CALLS.count("warning")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(ROOT, "checkpoints", "world_model_best.pt")):
        print("SKIP: no checkpoint at checkpoints/world_model_best.pt")
        sys.exit(0)

    _install_stub()
    print("demo branches:")

    done, _ = run("bundled real capture", 0)
    assert done, "bundled sample branch must render"

    done, _ = run("upload with no file", 1)
    assert not done, "upload without a file must stop cleanly, not crash"

    _, warns = run("synthetic (out-of-distribution)", 2)
    assert warns >= 1, "synthetic input must raise the drift warning"

    # PCAP upload, end to end through the real loader.
    from tests.test_pcap_ingest import build_pcap
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sample.pcap")
        build_pcap(p)
        # Too few flows to forecast on, so this must fail *gracefully*.
        run("upload PCAP (tiny capture)", 1, _FakeUpload(p))

    print("\nALL DEMO SMOKE TESTS PASSED")
