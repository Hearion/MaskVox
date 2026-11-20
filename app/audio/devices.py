from __future__ import annotations

import sounddevice as sd


def list_devices() -> dict:
    """Return WASAPI devices using PortAudio's current device IDs."""
    host_apis = sd.query_hostapis()
    wasapi = next(
        (index for index, host in enumerate(host_apis) if "wasapi" in host["name"].casefold()),
        None,
    )
    inputs = []
    outputs = []
    default_input_id = default_output_id = cable_id = None
    if wasapi is not None:
        for index, info in enumerate(sd.query_devices()):
            if info["hostapi"] != wasapi:
                continue
            device = {
                "id": index,
                "name": info["name"],
                "sample_rate": int(info["default_samplerate"]),
            }
            name = info["name"].casefold()
            if info["max_input_channels"] and "cable output" not in name:
                inputs.append(dict(device, channels=info["max_input_channels"]))
            if info["max_output_channels"]:
                outputs.append(dict(device, channels=info["max_output_channels"]))
                if cable_id is None and "cable input" in name:
                    cable_id = index

        default_input_id = _default(inputs, host_apis[wasapi]["default_input_device"])
        default_output_id = _default(outputs, host_apis[wasapi]["default_output_device"])
    return {
        "inputs": inputs,
        "outputs": outputs,
        "default_input_id": default_input_id,
        "default_output_id": default_output_id,
        "cable_output_id": cable_id,
        "has_vb_cable": cable_id is not None,
    }


def _default(devices: list[dict], preferred: int) -> int | None:
    if any(device["id"] == preferred for device in devices):
        return preferred
    return devices[0]["id"] if devices else None


def refresh_devices() -> dict:
    """Refresh the cached enumeration. The caller has already stopped all streams."""
    # PortAudio enumerates devices at Pa_Initialize and has no public refresh API.
    sd._terminate()
    sd._initialize()
    return list_devices()
