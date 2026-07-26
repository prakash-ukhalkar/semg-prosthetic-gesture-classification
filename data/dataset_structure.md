# Dataset Structure & Specification

## NinaPro Database Overview
- **Sampling Frequency:** 2000 Hz
- **Channels:** 12 sEMG channels
- **Gestures:** Rest, finger flexions, wrist movements, functional grasps

## Data Schema & Storage Formats
1. **Raw Signal Arrays:** `[samples x channels]`
2. **Segmented Windows:** `[num_windows x window_length x channels]`
3. **Feature Vectors:** `[num_windows x num_features]`
