# ELE 673 - Pattern Recognition HW

* Train set:
    * There are 2392 eye images which has 500 elements.
    * There are 1919 eye images which has 500 elements.
* Test set:
    * There are 2392 eye images which has 500 elements.
    * There are 1920 eye images which has 500 elements.
    
    
## Results


* 5 Dimensional Data
    * k-NN=5
        * Accuracy: 0.7931354359925789
        * FPR: 0.14590301003344483
    * PCA Classifier based on reconstruction errors
        * Accuracy: 0.7769016697588126
        * FPR: 0.11162207357859531

* 10 Dimensional Data
    * k-NN=5
        * Accuracy: 0.875
        * FPR: 0.06856187290969899
    * k-NN=1
        * Accuracy: 0.8638682745825603
        * FPR: 0.09824414715719064
    * k-NN=10
        * Accuracy: 0.8694341372912802
        * FPR: 0.8694341372912802
    * PCA Classifier based on reconstruction errors
        * Accuracy: 0.7771335807050093
        * FPR: 0.10827759197324414
                

* Number of PCA Components: 15
    * k-NN Classifier 15 dimensional
        * Accuracy: 0.9413265306122449
        * FPR: 0.02717391304347826

    * PCA Classifier based on reconstruction errors
        * Accuracy: 0.7773654916512059
        * FPR: 0.10493311036789298

* Number of PCA Components: 25

    * k-NN Classifier 25 dimensional
        * Accuracy: 0.9666048237476809
        * FPR: 0.01254180602006689

    * PCA Classifier based on reconstruction errors
        * Accuracy: 0.7755102040816326
        * FPR: 0.09991638795986622
        
* Number of PCA Components: 50

	* k-NN Classifier 50 dimensional
		* Accuracy: 0.987012987012987
		* FPR: 0.005016722408026756

	* PCA Classifier based on reconstruction errors
		* Accuracy: 0.7720315398886828
		* FPR: 0.08570234113712374

* kNN on High Dimensional Data
    * k-NN=5
        * Accuracy: 0.9336734693877551
        * FPR: 0.025919732441471572
    * k-NN=1
        * Accuracy: 0.939935064935065
        * FPR: 0.021321070234113712
    * k-NN=10
        * Accuracy: 0.922077922077922
        * FPR: 0.025501672240802676



## Runtimes

### k-NN
* 500 dim:
    * Runtime: 12.799838066101074
* 50 dim:
	* Runtime: 1.2768628597259521
* 25 dim:
	* Runtime: 0.6847412586212158
* 15 dim:
	* Runtime: 0.40080928802490234
* 10 dim:
	* Runtime: 0.28053760528564453
* 5 dim:
	* Runtime: 0.14687252044677734

### PCA Classifier
* 50 dim:
	* Runtime: 0.08042669296264648
* 25 dim:
	* Runtime: 0.0792078971862793
* 15 dim:
	* Runtime: 0.0736684799194336
* 10 dim:
    * Runtime: 0.08567476272583008
* 5 dim:
	* Runtime: 0.07042407989501953

## Comments

* PCA Classifier based on reconstruction errors: 
    * Positive and negative test images will be reconstructed for positive and negative training images.
    * Thus, 4 different error calculation is required.   


    