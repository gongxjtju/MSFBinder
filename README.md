# MSFBinder1
A Model Stacking Framework for Predicting DNA Binding Proteins

## DataSet

The raw sequences of PDB1075 and PDB186 datasets can be obtained from http://server.malab.cn/Local-DPP/Datasets.html

## Feature extraction 

### DWT method


The source code of DWT method can be downloaed from https://figshare.com/articles/Improved_detection_of_DNA-binding_proteins_via_compression_technology_on_PSSM_information/5104084


### 188D method

java -jar 188D.jar test.txt result.txt

When used 188D method, you need to cite the following papers:

Li YH, Xu JY, Tao L, Li XF, Li S, Zeng X, et al. (2016) SVM-Prot 2016: A Web-Server for Machine Learning Prediction of Protein Functional Families from Sequence Irrespective of Similarity. PLoS ONE 11(8): e0155290

Chen Lin, Ying Zou, Ji Qin, Xiangrong Liu, Yi Jiang, Caihuan Ke, Quan Zou. Hierarchical Classification of Protein Folds Using a Novel Ensemble Classifier. PLoS One. 2013, 8(2):e56499

Quan Zou, Zhen Wang, Xinjun Guan, Bin Liu, Yunfeng Wu, Ziyu Lin. An Approach for Identifying Cytokines Based On a Novel Ensemble Classifier. BioMed Research International. 2013, 2013:686090

### Ac_struct method

Using PSIPRED2 to obtain the predicted secondary structure file and  ac_struct.py for extracting features from the structure file.  

python ac_struct.py length startpositive endpositive countpositive startnege endnege countnege

The length is the least length of protien sequences. 

The startpositive(startnege) is the number of the first positive(negative) sample. 

The endpositive(endnege) is the number of the last positive(negative) sample. 

The countpositive(countnege) is the amount of positive(negative) samples. 


### Local_DPP method

Using the PSI-BLAST to obtain the PSSM matrix and localdpp.py for extracting features.

python localdpp.py countpositive countnege lamudamax n

The countpositive and countnege are as same as before.

The suggested values of lamudamax and n are 2 and 2 or 3 and 1.


All the features are preprcessed and stored in the directory: featuredata 











