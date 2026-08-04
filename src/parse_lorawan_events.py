from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_COLUMNS = [
    "source_file",
    "line_number",
    "event_time",
    "message_time",
    "received_at",
    "gateway_id",
    "gateway_eui",
    "m_type",
    "is_join_request",
    "dev_addr",
    "dev_eui",
    "join_eui",
    "f_cnt",
    "adr",
    "f_port",
    "frequency",
    "bandwidth",
    "spreading_factor",
    "coding_rate",
    "rssi",
    "channel_rssi",
    "snr",
    "channel_index",
    "crc_status",
    "payload_size_bytes",
    "unique_id",
]


def first_dictionary(value: Any) -> dict:
    """Return the first dictionary from a list, or an empty dictionary."""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]

    return {}


def calculate_payload_size(raw_payload: Any) -> int | None:
    """Calculate the decoded LoRaWAN payload size in bytes."""
    if not isinstance(raw_payload, str) or not raw_payload:
        return None

    try:
        return len(base64.b64decode(raw_payload, validate=False))
    except (ValueError, TypeError):
        return None


def parse_uplink_event(
    event: dict,
    source_file: str,
    line_number: int,
) -> dict | None:
    """
    Extract useful fields from a gs.up.receive event.

    Return None for events that are not gateway uplink receptions.
    """
    result = event.get("result")

    if not isinstance(result, dict):
        return None

    if result.get("name") != "gs.up.receive":
        return None

    data = result.get("data") or {}
    message = data.get("message") or {}
    payload = message.get("payload") or {}

    mac_payload = payload.get("mac_payload") or {}
    frame_header = mac_payload.get("f_hdr") or {}
    frame_control = frame_header.get("f_ctrl") or {}

    join_request = payload.get("join_request_payload") or {}

    settings = message.get("settings") or {}
    data_rate = settings.get("data_rate") or {}
    lora_settings = data_rate.get("lora") or {}

    rx_metadata = first_dictionary(message.get("rx_metadata"))

    identifier = first_dictionary(result.get("identifiers"))
    gateway_ids = identifier.get("gateway_ids") or {}

    message_header = payload.get("m_hdr") or {}
    message_type = message_header.get("m_type")

    is_join_request = bool(join_request)

    if not message_type and is_join_request:
        message_type = "JOIN_REQUEST"

    raw_payload = message.get("raw_payload")

    return {
        "source_file": source_file,
        "line_number": line_number,
        "event_time": result.get("time"),
        "message_time": settings.get("time"),
        "received_at": message.get("received_at"),
        "gateway_id": gateway_ids.get("gateway_id"),
        "gateway_eui": gateway_ids.get("eui"),
        "m_type": message_type,
        "is_join_request": is_join_request,
        "dev_addr": frame_header.get("dev_addr"),
        "dev_eui": join_request.get("dev_eui"),
        "join_eui": join_request.get("join_eui"),
        "f_cnt": frame_header.get("f_cnt"),
        "adr": frame_control.get("adr"),
        "f_port": mac_payload.get("f_port"),
        "frequency": settings.get("frequency"),
        "bandwidth": lora_settings.get("bandwidth"),
        "spreading_factor": lora_settings.get("spreading_factor"),
        "coding_rate": lora_settings.get("coding_rate"),
        "rssi": rx_metadata.get("rssi"),
        "channel_rssi": rx_metadata.get("channel_rssi"),
        "snr": rx_metadata.get("snr"),
        "channel_index": rx_metadata.get("channel_index"),
        "crc_status": message.get("crc_status"),
        "payload_size_bytes": calculate_payload_size(raw_payload),
        "unique_id": result.get("unique_id"),
    }


def find_input_files(raw_directory: Path) -> list[Path]:
    """Find supported JSONL-style files recursively."""
    supported_patterns = [
        "*.txt",
        "*.jsonl",
        "*.ndjson",
        "*.json",
    ]

    files: set[Path] = set()

    for pattern in supported_patterns:
        files.update(raw_directory.rglob(pattern))

    return sorted(path for path in files if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract LoRaWAN gateway uplinks from JSONL event files."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing raw LoRaWAN event files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "uplinks.csv",
        help="Output CSV file.",
    )

    parser.add_argument(
        "--error-log",
        type=Path,
        default=PROJECT_ROOT / "logs" / "parse_errors.log",
        help="File used to record malformed JSON lines.",
    )

    args = parser.parse_args()

    raw_directory = args.raw_dir.resolve()
    output_file = args.output.resolve()
    error_log = args.error_log.resolve()

    input_files = find_input_files(raw_directory)

    if not input_files:
        raise SystemExit(
            f"No input files found inside: {raw_directory}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_log.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    malformed_lines = 0
    ignored_events = 0
    uplink_events = 0
    join_requests = 0
    data_uplinks = 0

    with (
        output_file.open("w", encoding="utf-8", newline="") as csv_file,
        error_log.open("w", encoding="utf-8") as log_file,
    ):
        writer = csv.DictWriter(
            csv_file,
            fieldnames=OUTPUT_COLUMNS,
        )
        writer.writeheader()

        for input_file in input_files:
            print(f"Reading: {input_file.name}")

            with input_file.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as raw_file:
                for line_number, line in enumerate(raw_file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    total_lines += 1

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as error:
                        malformed_lines += 1

                        log_file.write(
                            f"{input_file}:{line_number}: "
                            f"{error.msg}\n"
                        )
                        continue

                    parsed_event = parse_uplink_event(
                        event=event,
                        source_file=input_file.name,
                        line_number=line_number,
                    )

                    if parsed_event is None:
                        ignored_events += 1
                        continue

                    writer.writerow(parsed_event)
                    uplink_events += 1

                    if parsed_event["is_join_request"]:
                        join_requests += 1
                    else:
                        data_uplinks += 1

    print("\nParsing completed")
    print("-----------------")
    print(f"Input files:       {len(input_files)}")
    print(f"Total lines:       {total_lines}")
    print(f"Uplink events:     {uplink_events}")
    print(f"Data uplinks:      {data_uplinks}")
    print(f"Join requests:     {join_requests}")
    print(f"Ignored events:    {ignored_events}")
    print(f"Malformed lines:   {malformed_lines}")
    print(f"Output file:       {output_file}")
    print(f"Error log:         {error_log}")


if __name__ == "__main__":
    main()
