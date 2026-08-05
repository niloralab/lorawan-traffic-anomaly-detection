# LoRaWAN Traffic Anomaly Detection

A beginner-friendly machine learning project for analysing LoRaWAN network traffic and detecting anomalous behaviour.

## Project goals

- Explore LoRaWAN network traffic data
- Parse raw gateway event files
- Identify useful traffic features
- Visualise traffic patterns
- Build a simple baseline machine learning model
- Evaluate and interpret the detected anomalies

## Traffic features

The following traffic features were initially considered:

- Packet size
- Inter-arrival time
- Flow duration
- Number of packets
- Packet rate
- Retransmissions

After inspecting the available LoRaWAN gateway logs, the first version of the project focuses on the following features:

- **RSSI (Received Signal Strength Indicator):** Measures the strength of the signal received by the gateway. Unusual changes may indicate interference, movement, or changes in the communication environment.

- **SNR (Signal-to-Noise Ratio):** Describes the quality of the received signal relative to background noise. Sudden changes may indicate degraded or unusual radio conditions.

- **Payload size:** Represents the number of bytes carried by an uplink. Changes in payload size may reveal changes in traffic behaviour even when the payload content is encrypted.

- **Inter-arrival time:** Measures the time between consecutive uplink observations within the same temporally segmented DevAddr group. Unexpected bursts or long silent periods may represent unusual behaviour.

- **Frame counter gap:** Measures the difference between consecutive LoRaWAN frame counters. Zero, negative, or unusually large gaps may require further investigation.

- **Possible retransmission or frame counter reuse:** An unchanged frame counter may indicate a retransmission, but it may also result from counter reuse or an observation-ordering issue.

- **Missing frame counters:** A forward jump in the frame counter indicates that some counter values were not observed. This does not necessarily prove packet loss because the missing messages may be outside the captured data interval.

These features are behavioural indicators. None of them independently proves that an anomaly or attack occurred.

## Project structure

The project is organised as follows:

```text
lorawan-traffic-anomaly-detection/
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── src/
│   ├── parse_lorawan_events.py
│   └── build_features.py
└── README.md
```

- `data/raw/` contains the original TXT files collected from the LoRaWAN gateways.
- `data/processed/` contains the CSV files generated during parsing and feature engineering.
- `results/` contains parsing error logs and will later store anomaly-detection results.
- `src/parse_lorawan_events.py` extracts relevant LoRaWAN uplink information from the raw gateway events.
- `src/build_features.py` creates temporal segments and calculates radio and behavioural features.
- `README.md` documents the project workflow, decisions, results, and limitations.

Raw and processed datasets are excluded from version control.

## Data parsing

Raw LoRaWAN gateway events are processed using `src/parse_lorawan_events.py`.

The parser:

- Reads the raw TXT files line by line
- Parses each line as a JSON event
- Extracts LoRaWAN uplink reception events
- Separates data uplinks from join requests
- Ignores unrelated event types
- Records malformed JSON lines in an error log
- Saves the extracted fields in `data/processed/uplinks.csv`

The parser does not detect anomalies. Its purpose is to transform raw gateway events into a structured dataset for further analysis.

### Parsing results

| Item | Count |
|---|---:|
| Input files | 1,421 |
| Total lines | 47,047 |
| Uplink events | 13,995 |
| Data uplinks | 12,853 |
| Join requests | 1,142 |
| Ignored events | 33,052 |
| Malformed lines | 0 |

The parser successfully processed all input files without encountering malformed JSON lines. The resulting structured dataset was saved to `data/processed/uplinks.csv`.

## DevAddr grouping and temporal segmentation

Ordinary LoRaWAN data uplinks in the available dataset do not contain DevEUI. Therefore, the permanent identity of a physical device cannot be determined from these records.

DevAddr is used only to group potentially related observations. It is not treated as a permanent device identifier because:

- A device may receive a new DevAddr after joining a new session.
- The same DevAddr may be reused in different sessions.
- Different devices may use the same DevAddr at different times.

To reduce the risk of comparing observations from separate capture periods, each DevAddr group is divided into temporal segments.

A new temporal segment begins when:

- An observation is the first occurrence of a DevAddr.
- The time since the previous observation with the same DevAddr is greater than 12 hours.

The 12-hour threshold was selected after examining the time-gap distribution. The median time gap was approximately 1.6 hours, while the 75th percentile increased to approximately 20.5 hours, indicating a separation between shorter behavioural sequences and long capture gaps.

| Item | Count |
|---|---:|
| Data uplinks | 12,853 |
| Unique DevAddr values | 8,840 |
| Temporal segments | 10,071 |
| Within-segment temporal comparisons | 2,782 |

These temporal segments do not represent verified LoRaWAN sessions or physical devices. They are used only to prevent behavioural features from being calculated across long periods of inactivity.

## Feature engineering

Feature engineering is performed using `src/build_features.py`.

The script:

1. Keeps ordinary data uplinks.
2. Removes observations without a valid timestamp or DevAddr.
3. Treats implausibly high RSSI measurements above -20 dBm as missing.
4. Sorts observations by `dev_addr` and `event_time`.
5. Divides each DevAddr group into 12-hour temporal segments.
6. Assigns a generated `session_id` to each segment.
7. Calculates behavioural changes only within the same segment.
8. Saves the generated features to `data/processed/features.csv`.

The generated `session_id` does not represent a verified LoRaWAN
session or a physical device. It is only an internal identifier for a
temporally segmented group of observations.

Eleven RSSI measurements above -20 dBm were considered implausible or
not directly comparable with the remaining measurements. These values
were treated as missing rather than deleting their complete observations.

### Generated features

| Feature | Description |
|---|---|
| `inter_arrival_time` | Time in seconds since the previous observation in the same temporal segment |
| `log_inter_arrival_time` | Log-transformed inter-arrival time used to reduce the influence of large time gaps |
| `rssi_change` | Difference between the current and previous RSSI values |
| `snr_change` | Difference between the current and previous SNR values |
| `payload_size_change` | Difference between the current and previous payload sizes |
| `f_cnt_gap` | Difference between the current and previous frame counters |
| `counter_reset_or_wrap` | Indicates a negative frame-counter difference within a temporal segment |
| `retransmission_or_reuse` | Indicates that the frame counter has not changed |
| `missing_counter_count` | Number of skipped counter values in a forward sequence |
| `log_missing_counter_count` | Log-transformed number of skipped counter values |
| `possible_packet_loss` | Indicates that one or more frame-counter values were not observed |

A missing frame-counter value does not necessarily prove that a packet was lost during radio transmission. The missing message may be outside the captured data interval. Therefore, `possible_packet_loss` is interpreted only as a missing-counter indicator.

### Feature engineering results

The feature engineering script produced 12,853 rows and divided the observations into 10,071 temporally segmented DevAddr groups.

| Feature | Rows where the feature could be calculated |
|---|---:|
| `inter_arrival_time` | 2,782 |
| `log_inter_arrival_time` | 2,782 |
| `rssi_change` | 2,782 |
| `snr_change` | 2,733 |
| `payload_size_change` | 2,782 |
| `f_cnt_gap` | 2,745 |
| `missing_counter_count` | 2,745 |

The first observation in each temporal segment does not have a previous observation in the same segment. Therefore, change-based features cannot be calculated for the first row of each segment:

`12,853 total observations - 10,071 segment-starting observations = 2,782 comparisons`

The number of available SNR and frame-counter features is slightly lower because some original observations do not contain SNR or FCnt values.

After removing rows with missing values in the core modelling features, 2,684 observations remain available for machine learning.

The dataset contains 80 negative frame-counter gaps and 243 zero frame-counter gaps within the temporal segments. These observations are preserved because they may contain useful behavioural information.

A negative frame-counter gap may indicate a counter reset, wraparound, out-of-order observation, or DevAddr reuse. A zero gap may indicate a retransmission or frame-counter reuse. These conditions are treated as indicators rather than confirmed anomalies.

## Current limitations

- The dataset does not contain ground-truth anomaly labels.
- The number of physical devices cannot be determined because DevEUI is absent from ordinary data uplinks.
- DevAddr is not a permanent device identifier.
- The generated temporal segments are not verified LoRaWAN sessions.
- The 12-hour segmentation threshold is an empirical project assumption.
- A frame-counter gap does not necessarily indicate actual packet loss.
- Retransmission cannot be reliably distinguished from frame-counter reuse.
- Most temporal segments contain only one observation and cannot produce change-based features.

## Next steps

The next phase will:

1. Select complete modelling features while retaining identifiers only as metadata.
2. Create `data/processed/ml_features.csv`.
3. Scale numerical features without allowing large values to dominate the model.
4. Train a baseline unsupervised anomaly-detection model because ground-truth labels are unavailable.
5. Inspect and interpret the observations identified as anomalous.

The model will not directly use `source_file`, `dev_addr`, `session_id`, or `gateway_id` as predictive features. These fields will be retained only to interpret the model output.
