# ExoSpecMiner Automated Benchmark Report

**Execution Date**: `2026-05-16 12:20:53`
**Test Cases Evaluated**: `50`
**Hardware Device**: `cpu`

## 1. Regression Accuracy Metrics

| Metric | Value | Target Threshold |
|---|---|---|
| **Mean Absolute Error (MAE)** | `0.0238` | `< 0.05` |
| **Root Mean Squared Error (RMSE)** | `0.0300` | `< 0.08` |
| **R² Score** | `0.9868` | `> 0.90` |

## 2. Uncertainty Quality & Calibration

| Metric | Value | Target Threshold |
|---|---|---|
| **95% Credible Interval Coverage** | `100.0%` | `> 90.0%` |
| **Classifier 2-Sample Test (C2ST)** | `0.9993` | `0.50 ± 0.05` |

## 3. High-Speed Inference Verification

| Metric | Value | Target Threshold | Status |
|---|---|---|---|
| **Posterior Inference Time** | `2.03 s` | `< 5.0 s` | **✅ PASSED** |
