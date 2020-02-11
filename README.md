# DelibAnalysis version 2.0
This version was developed jointly by Eleonore Fournier-Tombs and Curtis Hendricks. This version is fully controlled by the config file, which means that it requires no modification of the Python files. It is run in the command line and is fully commented using Pydoc.

In order to run:
python3 delib_ana.py -c [name_of_config_file.ini]

Details of configuration in delib_ana.ini


## Contact
If you are using this, we want to hear from you! Please contact us at: eleonore.fournier-tombs@mail.mcgill.ca


## Data
At the moment, each project is providing their own training dataset. This means that researchers are responsible for finding their own speeches or comments and manually coding for the DQI. We are currently collecting training data in order to provide a generalizable model, and will update the community as soon as this is done. In the meantime, test data will be provided shortly.

### Training Data
Training, or "labelled" data, should be formatted as a CSV with the following columns: speaker, comment, participation*, level of justification*, content of justification*, respect*, counterarguments*, constructive politics*. 
* Depending on the DQI used, there can be anywhere from 6 to 12 deliberative quality indicators. However, you only need one in order to run the program. The program should be run separately for each indicator, as a new, separate model will be trained in each case. Note that the speaker and comment columns are always mandatory.

### Unlabelled Data
This is the data that has not been manually coded. There can be anywhere from a few hundred to millions of comments, depending on your dataset, but it should be provided in CSV. Note that we allow batch labelling, so the unlabelled data can be provided in separate files, as long as they are in the same folder (which you should set in the configuration file). The only required columns are: speaker, comment.


## Motivation
DelibAnalysis is a tool that assigns democratic quality scores to any political speech - social media conversations, blogs, parliamentary speeches, etc. It is based on the Discourse Quality Index (DQI). It aspires to take a non-partisan, rooted in political theory, with the assumption that good deliberation is better for society. The tool was launched in June, 2018, and has been in increasing use across academia since.


## Please respect the Copyright
Copyright: Attribution-NonCommercial-ShareAlike CC BY-NC-SA
https://creativecommons.org/licenses/by-nc-sa/4.0/



## References
DelibAnalysis was originally developed by Eleonore Fournier-Tombs as part of her doctoral thesis at the University of Geneva.
Fournier-Tombs, Eleonore, and Giovanna Di Marzo Serugendo. "DelibAnalysis: Understanding the quality of online political discourse with machine learning." Journal of Information Science (2019): 0165551519871828.
Steenbergen, Marco R., et al. "Measuring political deliberation: A discourse quality index." Comparative European Politics 1.1 (2003): 21-48.

