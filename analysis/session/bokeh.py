"""Small helpers shared by notebook-hosted Bokeh applications."""

from __future__ import annotations


def free_local_port() -> int:
    """Ask the OS for an unused localhost port and return its concrete number."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def ensure_notebook_output() -> bool:
    """Initialize Bokeh's notebook hook when running inside an IPython kernel."""
    import sys

    if "ipykernel" not in sys.modules:
        return False
    import os
    from bokeh.io import output_notebook
    from bokeh.io.state import curstate

    if curstate().notebook_type is None:
        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"
        output_notebook(hide_banner=True)
    return True


def stop_notebook_servers() -> int:
    """Stop and unregister Bokeh servers left by earlier notebook outputs."""
    try:
        from bokeh.io.state import curstate
    except ImportError:
        return 0

    servers = curstate().uuid_to_server
    stopped = 0
    for server_id, server in list(servers.items()):
        try:
            # Notebook Bokeh servers share the kernel's Tornado IOLoop. Waiting
            # synchronously for that same loop to finish its callbacks can
            # deadlock a GUI relaunch or kernel shutdown.
            server.stop(wait=False)
            stopped += 1
        except (AssertionError, RuntimeError):
            # It was already stopped but remained registered in notebook state.
            pass
        finally:
            servers.pop(server_id, None)
    return stopped
