# Quick Reference: Sensor Alarm Thresholds
## Illustrative Operational Reference Card

> DISCLAIMER: Illustrative values for the IndustrialAI demonstration only.
> Not approved for real equipment operation.

## Normal Operating Ranges

| Parameter              | Low Alarm | Normal Range  | High Alarm |
|------------------------|-----------|---------------|------------|
| Air Temp [K]           | < 292     | 292–308       | > 308      |
| Process Temp [K]       | < 298     | 298–315       | > 315      |
| Rotational Speed [rpm] | < 1100    | 1100–2800     | > 2900     |
| Torque [Nm]            | < 5       | 5–70          | > 70       |
| Tool Wear [min]        | 0         | 0–200         | > 220      |

## Tool Change Schedule

- Inspection at: 150 min cumulative wear
- Detailed check at: 200 min
- Hard replacement limit: 250 min
- Emergency replacement: if torque increases > 15% above baseline at constant RPM

## Power Formula
P [W] = Torque [Nm] × (2π × RPM / 60)
Maximum rated power: 9000 W

Example: 60 Nm × (2π × 2000 rpm / 60) = 12,566 W → EXCEEDS RATED POWER

## Temperature Differential Rule
Process Temp − Air Temp should be 8–10 K under normal conditions.
Differential > 15 K → check cooling system.
Differential < 5 K → check process heating, sensor calibration.

## Failure Mode Quick Guide

| Failure Mode | Primary Trigger                      | First Action                   |
|--------------|--------------------------------------|--------------------------------|
| TWF          | Tool wear > threshold + torque rise  | Stop, replace tool             |
| HDF          | Process temp > 315 K                 | Check coolant, reduce load     |
| PWF          | Power > 9000 W                       | Reduce RPM or Torque           |
| OSF          | Tool wear × Torque > limit           | Emergency stop, inspect        |
| RNF          | None of the above                    | General inspection             |

## Escalation Contacts
- Level 1: Equipment Operator (daily checks)
- Level 2: Maintenance Technician (anomalies, minor repairs)
- Level 3: Maintenance Engineer (failure investigation, schedule changes)
- Level 4: OEM / Specialist (structural failures, control system faults)
