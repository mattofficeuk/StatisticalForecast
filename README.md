# StatisticalForecast
Code and instructions for making analogue/statistical forecasts using CMIP5/6 data

## Very general workflow
1. Run `SUBMIT_ProcessVarsCMIP.sh` with some input (e.g. `cmip6` ) to create the pre-processed data
2. Run `SUBMIT_AnalyseAnalogueSource.sh` (which calls `AnalyseAnalogueSource2_Jasmin.py` ) to do some analysis of time series
3. Also run `SUBMIT_Sbatch_AnalogueCache_Spatial.sh` to do some analysis of _spatial_ data
4. And `SUBMIT_Sbatch_AnalogueCache_Spatial_Skill.sh` to calculate the skill of some of this spatial data
