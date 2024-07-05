# Anomaly Detection Based on EMA and Residual Analysis

## Overview

This project is designed to detect anomalies in sensor data by comparing actual readings with the values predicted by an Exponential Moving Average (EMA). Anomalies are identified based on the residuals, which are the differences between the actual sensor readings and the EMA values. If the absolute value of a residual exceeds a specified threshold, which is determined by a multiple of the standard deviation of the residuals, it is flagged as an anomaly.

## Methodology

1. **Calculate EMA (Exponential Moving Average)**:
   - EMA is used to smooth the sensor data and provide a baseline for anomaly detection.
   
2. **Compute Residuals**:
   - Residuals are calculated as the difference between actual sensor readings and their corresponding EMA values: $e_t = x_t - EMA_t$,  where $e_t$ is the residual at time $t$, $x_t$ is the actual sensor reading, and $EMA_t$ is the EMA value at time $t$.

3. **Calculate Standard Deviation of Residuals**:
   - The standard deviation $SD$ of the residuals is computed to quantify the variability in the data.

4. **Anomaly Detection**:
   - An anomaly is detected if the absolute value of a residual exceeds a defined threshold, which is $n$ times the standard deviation of the residuals: $Anomaly \quad if \quad |e_t|>n\cdot SD(e_t)$,
   where $n$ is predetermined multiplier.

## Requirements

- Python 3.x
- Libraries: numpy, pandas

---

This project is part of the [DS.MISIS.HACK 2.0](https://www.kaggle.com/competitions/misis-hack/) competition.
