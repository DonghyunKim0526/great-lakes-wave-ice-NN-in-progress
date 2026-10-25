# WIFF-Model
 Wave-induced Floe Fracture Code

This is code for WIFF1.0 - An accelerated superparameterized wave-induced floe fracture code. 

This was developed by Chris Horvat and Lettie Roach. 

The code consists of several subfolders and code in MATLAB and Python. It references data that is not  included in this repository but is located in a Zenodo repository referenced here: XX AVAILABLE ON PUBLICATION XX. 

The code is structured as follows:

Readme.md

Model/
   --- NN_v1.h5 - Keras-generated network for v1 of WIFF. 
   --- NN_v1.txt - text file corresponding to the Keras network. 
   --- SP_WIFF.m - MATLAB code for running and plotting a single iteration of the SP-WIFF code
   --- Print_and_Test_Network.ipynb - a jupyter notebook that shows how to implement the SP-WIFF Neural Network, loading the Keras network, writing NN_v1.txt, and testing against random input. 

Classifier/ - NOTE THIS IS NOT USED IN WIFF1.0. It is an optional layer to classify input vectors (will fracture/won't fracture) in lieu of hard constraints on sea ice thickness or concentration. 
   --- Classifier_v1.h5 - Keras-generated classifier layer for v1 of WIFF.  
   --- Classifier_v1.txt - text file corresponding to the Keras Classifier
   --- Print_Classifier.ipynb - a jupyter notebook that shows how to implement the SP-WIFF Classifier, loading the Keras network, writing Classifier_v1.txt, and   testing against random input. 
   
Training_Data/ 
   --- collect_training.m - MATLAB code for obtaining training data from a training dataset that contains input spectra, ice thickness, and ice concentration and output histograms. 
   
Training/ 
   Classifier/ - NOTE THIS IS NOT A PART OF WIFF1.0 BUT IS PROVIDED FOR COMPLETENESS. 
      --- 
   Neural_Net/
      ---

Paper_Figures/
   --- drive_plotting.m - code that initializes paths and executes the production of the 3 paper figures. 
   
   Fig-1/
      --- make_fig_1.m - code that produces Fig 1 of the MS. 
   
   Fig-2/
      --- make_fig_2.m - code that produces Fig 2 of the MS. 
   
   Fig-3/
      --- make_fig_3.m - code that produces Fig 3 of the MS. 
   
   Misc/
      --- training_preamble.m - code that loads in data for training data plots (Fig.s 1-2)
      --- gcm_preamble.m - code that loads in data for training data plots (Fig. 3) 
   
   Plot_Tools/
      --- cmocean.m - beautiful colormaps for producing figures. (https://matplotlib.org/cmocean/)
      --- horvat_colors.m - code for producing a matrix clabs for plot colors. 
      
Misc/
   --- NN_params.mat - a file that contains the FSD bin centers, FSD bin edges, and wave spectrum frequencies. 
   --- peakfinder2.mat - a file for finding peaks in, for example, wavy data. Used by SP_WIFF.m

