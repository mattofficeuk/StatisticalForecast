# StatisticalForecast
Code and instructions for making analogue/statistical forecasts using CMIP5/6 data

## Things currently being worked on
1. LEO - Should the intermediate files, which are currently Python pickle files, be something more interoperable? Perhaps it would be good to be able to store the VERSION number of the code that created the files in the files themselves
2. MATT - AnalyseAnalogueSource2_Jasmin.py is still in the original format and hardcoded to Matt's paths etc. Make generic.

## Short term future plans
1. A script to `touch` the temporary files on Jasmin so we don't have to keep recreating them (they are auto-deleted after 1 month)
2. ...

## Mid term plans
1. A way of running the system backwards - e.g. in order to estimate which variables/regions would have been most useful for a particular forecasts. To then try and understand *why* that would be in order to design a better forward forecast system. Have to be careful not to cheat though.
2. Make newer and better metrics for actually choosing the analogue predictors
3. Consider more intelligent methods of combining the analogue predictors. Currently we use a linear, equally weighted combination after normalising the variance
4. ...

## Long term targets
1. A flexible system where you can specify a (set of) (CMIP) variables/regions/timescales as predictors and make forecasts of another user-specified variable/region over timescales also specified by the user. In addition, the ability to plug-and-play different methods for choosing the analogues (e.g. RMSE difference; correlation patterns; combinations of these)
2. The ability to use more advanced ML methods to pick the analogues, such as CNNs targetted at specific regions (e.g. North Atlantic SSTs)

## Finished tasks
1. DONE LEO - Abstract out file paths into an initialisation file so it is easier to run as a different user. Could we make it automatic based on your username?
2. DONE LEO - Separate out as much as possible the metric used to choose the analogues. In order to allow us to make progress (see below) by defining newer, better metrics
3. DONE LEO - Include a 'testing' function for the scripts to enable quick development and trouble shooting

## Very general workflow
1. `STEP1_PreProcessing`: Run `SUBMIT_ProcessVarsCMIP.sh` with some input (e.g. `cmip6` ) to create the pre-processed data
2. `STEP2_CreateAnalogues`: Run `SUBMIT_CreateAnalogues.sh` (which calls `CreateAnalogues.py` ) to create the analogues based on choices specified in the `SUBMIT` file
3. `STEP3a_PickAnalogues_AreaAverages`: Run `SUBMIT_PickAnalogues_AreaAverages.sh` (which calls `PickAnalogues_AreaAverages.py` ) to _pick_ the analogues based on choices specified in the `SUBMIT` file. Note: This is for the area-averaged (e.g. SPG area average) indices. The analogues that were chosen are saved as output and you can then visualise these and combine them in different ways (leading to different skill estimates) in the associated Python Notebook scripts (TODO: Matt to add these)
4. `STEP3b_PickAnalogues_CalculateSkill_Maps`: Run `SUBMIT_PickAnalogues_CalculateSkill_Maps.sh` (which calls `PickAnalogues_CreateSkill_Maps.py` ) to do both _pick_ and _calculate the skill_ of a combination of analogues. The skill is also calculated here as it is very memory intensive (and slow) working with spatial data, so to do this in interactive notebooks would be hard.

## Description of files
- `SUBMIT*` - wrapper scripts (Shell) that take some input (e.g. `cmip6` - see scripts) and submit jobs to the Jasmin queues, calling the python scripts
- `queue_spacer_sbatch.sh` - A script to ensure we don't submit too many jobs at once. I'm not sure if this is necessary
- `analogue.py` and `cmip.py` and `mfilter.py` - Somewhere where I have stored custom code relevant for this work
- `selection.py` - Specifies the procedure on which the selection of the analogues is based
- `###_CMIP.py` - To take the raw CMIP data and pre-process it into a common format. No analogue stuff here.
- `CreateAnalogues.py` - To take the pre-processed data and convert it to the analogue format, for later selection
- `PickAnalogues_AreaAverages.py` - To find the best (based on a given method) analogue source data
- `TODO.py` - A *Python notebook* to calculate the skill and visualise the AreaAverage analogue predictions
- `PickAnalogues_CreateSkill_Maps.py` - To both _pick_ the best analogues and then _calculate the skill_ in map form.

![A schematic diagram of the analogue system](images/Schematic.png)
