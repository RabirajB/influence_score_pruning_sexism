# influence_score_pruning_sexism
CODE for our IJCNLP-AACL'25 paper titled **"Enhancing Training Data Quality through Influence Scores for Generalizable Classification: A Case Study on Sexism Detection"**

This is the repository for testing dataset_pruning experiments. <br>
First step is the calculation of InfluenceScores, for this run the following commands:<br>
**cd Influence_Scores** <br>
**run_model_full.sh** <br>
**run_model_null.sh** <br>
**run_pvi.sh** <br>
**run_el2n.sh** <br>

After the scores are calculated run the Averaging_Run_Score_Analysis.ipynb and create the final train split containing the PVI and EL2N scores which we will use throughout our pruning strategies. 

For running the submodular pruning experiments one needs to download the submodlib library. First you need to extract the embeddings. To calculate the embeddings go to submodular_pruning/cluster_analysis/get_model_embeddings.sh

Then run submodular_pruning/submodular_pruning.sh

For running Informed Undersampling experiments, go to informed_undersampling folder and run Informed_Undersampling.ipynb

For running Proportional Pruning experiments, go to proportional_pruning folder and run Proportional_Sampling_Prep.ipynb

Then after the datasets are created run the Fine_Tune_Proportional.py and Fine_Tune_Undersampling.py

