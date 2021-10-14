# NNAIMQ
This repository gathers the NNAIMQ code along with some instructions and readme files.

NNAIMQ is a Python interfaced model designed to predict QTAIM charges of C, H, O and N atoms with high accuracy within simple gas-phase organic and biological structures. Altogether, NNAIMQ comprises a total of 4 Artificial Neural Networks (ANN) fitted to quantum chemical data.

-The ./code/ folder gathers all the required files to execute the NNAIMQ code.   
-The ./examples/ folder contains an input example for the execution of NNAIMQ.  

Requirements:  
The execution of NNAIMQ requires the following libraries and modules:

-numpy,sys,subprocess,os,tensorflow,pathlib,pandas,seaborn,matplotlib,random.   
-Additionally, the Keras module is required.   
-Python, at least, version 3.7.3.  

It should be noticed that some of the aforementioned modules or libraries are not used explicitly but may be suitable to run some test on the performance of this first version of the NNAIMQ code.

Execution of NNAIMQ:

The execution of NNAIMQ is straightforward:

1) NNAIMQ must be executed in the same directory where all the code files are contained.

2) An input file with any desired name (i.e input) is required. This file contains a list of the geometries whose atomic charge is to be computed. As an example, a simple input file could be:

methane.xyz  
water.xyz  
toluene.xyz  

3) The files containing the structures (i.e methane.xyz) must be in the execution folder. The geometries of the structures to be computed are given in standard XYZ Cartesian coordinates, for instance:

5  
methane  
C -4.36105 1.86636  0.00000  
H -3.29105 1.86636  0.00000  
H -4.71772 1.07300 -0.62311  
H -4.71772 2.80267 -0.37552  
H -4.71772 1.72342  0.99863  

4) NNAIMQ is executed from the command line as:  
   python NNAIMQ_v1.py input

5) If executed correctly, NNAIMQ generates different output files:  
-name.charge: file containing the QTAIM charges.  
-name.acsf: file containing the featurization vector.  

6) After the execution, the .acsf files are not required anymore and can be removed from the current directory.
