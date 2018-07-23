# MSFBinder1
A Model Stacking Framework for Predicting DNA Binding Proteins

DataSet

You can gain PDB1075 and PDB186 datasets from http://server.malab.cn/Local-DPP/Datasets.html

DWT method

The code of DWT method can be gained from https://figshare.com/articles/Improved_detection_of_DNA-binding_proteins_via_compression_technology_on_PSSM_information/5104084

188D method

java -jar 188D.jar test.txt result.txt

When used 188D method, you need to cite the following papers:

Li YH, Xu JY, Tao L, Li XF, Li S, Zeng X, et al. (2016) SVM-Prot 2016: A Web-Server for Machine Learning Prediction of Protein Functional Families from Sequence Irrespective of Similarity. PLoS ONE 11(8): e0155290

Chen Lin, Ying Zou, Ji Qin, Xiangrong Liu, Yi Jiang, Caihuan Ke, Quan Zou. Hierarchical Classification of Protein Folds Using a Novel Ensemble Classifier. PLoS One. 2013, 8(2):e56499

Quan Zou, Zhen Wang, Xinjun Guan, Bin Liu, Yunfeng Wu, Ziyu Lin. An Approach for Identifying Cytokines Based On a Novel Ensemble Classifier. BioMed Research International. 2013, 2013:686090

Ac_struct method

You need to utilize the PSIPRED2 to gain the predicted secondary structure file. 

python ac_struct.py length startpositive endpositive countpositive startnege endnege countnege

The length is the least length of protien sequences. 

The startpositive(startnege) is the number of the first positive(negative) sample. 

The endpositive(endnege) is the number of the last positive(negative) sample. 

The countpositive(countnege) is the amount of positive(negative) samples. 

Then you can gain g_feature_gai1221_structual_186.csv, g_label_gai1221_structual_186.csv, g_feature_gai1221_structual_1075.csv,g_label_gai1221_structual_1075.csv. 

Local_DPP method

You need to gain the PSSM matrix of protein suquences.

MSFBinderSVM, MSFBinderSVMRF and MSFBinderSVMRFNB method

You need to gain parameters of the stacking methods by grid search.






