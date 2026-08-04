# Network Traffic Anomaly Detection

A beginner-friendly machine learning project for analysing LoRaWAN network traffic and detecting anomalous behaviour.

## Project goals

- Explore LoRaWAN network traffic data
- Parse raw gateway event files
- Identify useful traffic features
- Visualise traffic patterns
- Build a simple baseline machine learning model
- Evaluate and interpret the detected anomalies

## Possible features

The following traffic features were initially considered:

- Packet size
- Inter-arrival time
- Flow duration
- Number of packets
- Packet rate
- Retransmissions

After inspecting the available LoRaWAN gateway logs, the first version
of the project focuses on the following features:

- **RSSI (Received Signal Strength Indicator):** Measures the strength
  of the signal received by the gateway. Unusual changes may indicate
  interference, movement, or changes in the communication environment.

- **SNR (Signal-to-Noise Ratio):** Describes the quality of the received
  signal relative to background noise. Sudden changes may indicate
  degraded or unusual radio conditions.

- **Payload size:** Represents the number of bytes carried by an uplink.
  Changes in payload size may reveal changes in device behaviour even
  when the payload content is encrypted.

- **Inter-arrival time:** Measures the time between consecutive uplink
  observations associated with the same DevAddr. Unexpected bursts or
  long silent periods may represent unusual behaviour.

- **Frame counter gap:** Measures the difference between consecutive
  LoRaWAN frame counters. Zero, negative, or unusually large gaps may
  require further investigation.

- **Possible retransmission or frame counter reuse:** An unchanged frame
  counter may indicate a retransmission, but it may also result from
  counter reuse in a different session.

- **Missing frame counters:** A forward jump in the frame counter
  indicates that some counter values were not observed. This does not
  necessarily prove packet loss because the missing messages may be
  outside the captured data interval.
## Project structure

The project is organised as follows:

```text
network-traffic-anomaly-detection/
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── src/
│   ├── parse_lorawan_events.py
│   └── build_features.py
└── README.md
```

* `data/raw/` contains the original TXT files collected from the LoRaWAN gateways.
* `data/processed/` contains the CSV files generated during parsing and feature engineering.
* `results/` contains parsing error logs and will later store anomaly-detection results.
* `src/parse_lorawan_events.py` extracts relevant LoRaWAN uplink information from the raw gateway events.
* `src/build_features.py` calculates radio and behavioural features from consecutive uplink observations.
* `README.md` documents the project workflow, decisions, results, and limitations.

## Data parsing

Raw LoRaWAN gateway events are processed using
`src/parse_lorawan_events.py`.

The parser:

- Reads the raw TXT files line by line
- Parses each line as a JSON event
- Extracts LoRaWAN uplink reception events
- Separates data uplinks from join requests
- Ignores unrelated event types
- Records malformed JSON lines in an error log
- Saves the extracted fields in `data/processed/uplinks.csv`

The parser does not detect anomalies. Its purpose is to transform raw
gateway events into a structured dataset for further analysis.

### Parsing results

The parser produced the following results:

| Item | Count |
|---|---:|
| Input files | 1,421 |
| Total lines | 47,047 |
| Uplink events | 13,995 |
| Data uplinks | 12,853 |
| Join requests | 1,142 |
| Ignored events | 33,052 |
| Malformed lines | 0 |

The parser successfully processed all input files without encountering
any malformed JSON lines.

The resulting structured dataset was saved to:

`data/processed/uplinks.csv`

## Device and session identification

Ordinary LoRaWAN data uplinks do not normally include the permanent
device identifier, DevEUI. Therefore, this project uses `DevAddr` to
group consecutive uplink observations.

However, DevAddr is treated only as a session-level identifier and not
as a permanent device identity because:

- A device may receive a new DevAddr after starting a new session.
- The same DevAddr may be reused in different sessions.
- Different devices may use the same DevAddr at different times.

The dataset contains:

| Item | Count |
|---|---:|
| Data uplinks | 12,853 |
| Unique DevAddr values | 8,840 |
| DevAddr values observed at least twice | 1,889 |
| DevAddr values observed at least 10 times | 41 |
| DevAddr values observed at least 50 times | 2 |

Most DevAddr values appear only once. Therefore, temporal features can
only be calculated for DevAddr values observed more than once.

## Feature engineering

Feature engineering is performed using:

`src/build_features.py`

The script first removes join requests because they normally do not
contain a DevAddr. It then removes observations without a valid
timestamp or DevAddr.

The remaining observations are sorted by:

- `dev_addr`
- `event_time`

Observations with the same DevAddr are grouped together. Behavioural
features are then calculated by comparing each observation with the
previous observation in the same group.

The generated dataset is saved to:

`data/processed/features.csv`

### Generated features

| Feature | Description |
|---|---|
| `inter_arrival_time` | Time in seconds since the previous observation with the same DevAddr |
| `log_inter_arrival_time` | Log-transformed inter-arrival time used to reduce the effect of very large time gaps |
| `rssi_change` | Difference between the current and previous RSSI values |
| `snr_change` | Difference between the current and previous SNR values |
| `payload_size_change` | Difference between the current and previous payload sizes |
| `f_cnt_gap` | Difference between the current and previous frame counters |
| `counter_reset_or_wrap` | Indicates that the frame-counter difference is negative |
| `retransmission_or_reuse` | Indicates that the frame counter has not changed |
| `missing_counter_count` | Number of counter values skipped in a forward sequence |
| `log_missing_counter_count` | Log-transformed number of missing counter values |
| `possible_packet_loss` | Indicates that one or more frame-counter values were not observed |

A missing frame counter does not necessarily prove that a packet was
lost during radio transmission. The missing message may be outside the
captured time interval. Therefore, `possible_packet_loss` should be
interpreted only as an indicator of missing frame-counter values.

### Feature engineering results

The feature engineering script produced 12,853 rows.

| Feature | Available values |
|---|---:|
| `inter_arrival_time` | 4,013 |
| `log_inter_arrival_time` | 4,013 |
| `rssi_change` | 4,013 |
| `snr_change` | 3,947 |
| `payload_size_change` | 4,013 |
| `f_cnt_gap` | 3,971 |
| `missing_counter_count` | 3,971 |

The dataset contains 8,840 unique DevAddr values. The first observation
of each DevAddr does not have a previous observation and therefore
cannot have change-based features.

This explains the number of available inter-arrival values:

`12,853 total observations - 8,840 first observations = 4,013`

After removing rows with missing values in the core modelling features,
3,905 observations remain available for machine learning.
