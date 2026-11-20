"""Start the MaskVox desktop window."""

from __future__ import annotations

import logging
import sys

from . import config
from .events import Events

LOGGER = logging.getLogger(__name__)


def create_window(events: Events, *, hidden: bool = False):
    import webview
    from webview.dom import DOMEventHandler

    from .api import Api

    api = Api(events)
    window = webview.create_window(
        "MaskVox", url=config.UI_PATH.as_uri(), js_api=api,
        width=1080, height=700, min_size=(960, 640), hidden=hidden,
        background_color="#0b0d12", text_select=False,
    )
    api._window = window

    def ready() -> None:
        events.ready(window)
        try:
            library = window.dom.get_element("#voice-library")
            if library is not None:
                library.on("drop", DOMEventHandler(api._on_drop, prevent_default=True))
        except Exception:
            LOGGER.exception("无法启用拖拽导入，请使用文件选择按钮")

    window.events.loaded += ready
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
