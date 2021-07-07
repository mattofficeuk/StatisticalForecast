# StatisticalForecast
Code and instructions for making analogue/statistical forecasts using CMIP5/6 data

## Very general workflow
1. Run `SUBMIT_ProcessVarsCMIP.sh` with some input (e.g. `cmip6` ) to create the pre-processed data
2. Run `SUBMIT_Sbatch_AnalogueCache_Spatial.sh` (which calls `AnalogueCache_Spatial.py` ) to do some analysis of _spatial_ data
3. And `SUBMIT_Sbatch_AnalogueCache_Spatial_Skill.sh` (which calls `AnalogueCache_Spatial_Skill.py` ) to calculate the skill of some of this spatial data
4. Run `SUBMIT_AnalyseAnalogueSource.sh` (which calls `AnalyseAnalogueSource2_Jasmin.py` ) to do some analysis of the analogue fields that were used above

## Description of files
- `SUBMIT*` - wrapper scripts (Shell) that take some input (e.g. `cmip6` - see scripts) and submit jobs to the Jasmin queues, calling the python scripts
- `queue_spacer_sbatch.sh` - A script to ensure we don't submit too many jobs at once. I'm not sure if this is necessary
- `AnalogueCache_Spatial.py` - To find the best (based on a given method) analogue source data from the pre-processed data
- `AnalogueCache_Spatial_Skill.py` - To take those source files and estimate the skill of the predictions
- `AnalyseAnalogueSource2_Jasmin.py` - To look at the source files for the skill (maps) and do some analysis of these
