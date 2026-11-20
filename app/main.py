"""Start the MaskVox desktop window."""

from __future__ import annotations

import logging
import sys

from . import config
from .events import Events

LOGGER = logging.getLogger(__name__)


def create_window(events: Events, *, hidden: bool = False):
    import webview

    from .api import Api

    api = Api(events)
    window = webview.create_window(
        "MaskVox", url=config.UI_PATH.as_uri(), js_api=api,
        width=1080, height=700, min_size=(960, 640), hidden=hidden,
        background_color="#0e0f13", text_select=False,
    )
    api._window = window

    window.events.loaded += lambda: events.ready(window)
    window.events.closing += events.close
    return window, api


def main() -> int:
    events = Events()
    api = None
    try:
        config.setup_logging()
        import webview

        _, api = create_window(events)
        webview.start(gui="edgechromium", http_server=False)
        return 0
    except Exception:
        LOGGER.exception("MaskVox 启动失败")
        print("MaskVox 无法启动。请确认 WebView2 可用，并检查 logs/maskvox.log。", file=sys.stderr)
        return 1
    finally:
        try:
            if api is not None:
                api._close()
        except Exception:
            LOGGER.exception("MaskVox 关闭时发生错误")
        finally:
            events.close()


if __name__ == "__main__":
    raise SystemExit(main())
