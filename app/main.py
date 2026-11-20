"""Start the MaskVox desktop window."""


def main() -> None:
    import webview

    webview.create_window("MaskVox", html="<title>MaskVox</title>", width=1080, height=700, min_size=(960, 640))
    webview.start(gui="edgechromium", http_server=False)


if __name__ == "__main__":
    main()
