# Third-Party Notices

SignTurk source code is licensed under MIT. The following dependencies and
assets keep their own licenses and terms; this file is a notice, not a change
to those terms.

| Component | Use | Upstream / license |
|---|---|---|
| AUTSL | Training and evaluation dataset; videos are not redistributed | Dataset owner and paper terms: Ankara University CVML |
| TensorFlow / Keras | Temporal model training and inference | Apache-2.0 |
| MediaPipe | Live hand-landmark extraction | Apache-2.0 |
| RTMPose / MMPose | 226-class whole-body research pipeline | Apache-2.0 source; checkpoint training-data terms may also apply |
| RTMlib | Lightweight RTMPose runtime | MIT |
| YOLOX | Person detector used by the research extractor | Apache-2.0 source; checkpoint terms may also apply |
| FastAPI / Starlette / Uvicorn | Web application and WebSocket server | MIT / BSD-3-Clause |
| React / ReactDOM | Browser interface | MIT |
| Babel Standalone | In-browser JSX transform in the academic prototype | MIT |
| Three.js | 3D avatar rendering | MIT |
| SQLAlchemy | Persistence layer | MIT |
| gTTS | Optional speech synthesis client | MIT; Google service terms apply |
| Google Fonts (Fraunces, Inter, JetBrains Mono) | Interface typography loaded from Google Fonts | SIL Open Font License 1.1 |

The avatar GLB and recorded landmark animations are project assets. Do not
reuse them outside this project unless you have confirmed their original
source and redistribution permission. Model files published in the GitHub
Release are project checkpoints; they do not grant rights to the AUTSL source
videos or third-party extractor checkpoints.
