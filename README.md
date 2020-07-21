# Warfarin-dosing
 
This project implements the LinRel and SupLinRel algorithms from the paper [Using Confidence Bounds for
Exploitation-Exploration Trade-offs][1] by P. Auer, and applies these reinforcement learning algorithms to predict Warfarin dosages given patients' heights, weights, etc. 

The problem description can be found [here](project_description.pdf), which is downloaded from [here][2]. The dosages are classified as low (less than 21mg), medium (between 21mg and 49mg), and high (greater than 49mg).

## Running the Programs

First, download all files in this repository.

The following command will run the file "filename.py". Pandas, NumPy, Sklearn, and Scipy are needed.

```bash
python3 filename.py
```
"q1.py" runs the fixed-dose algorithm and two algorithms proposed by Consortium (2009). The details of these algorithms can be found [here][3].

"q2.py" runs the SupLinRel algorithm on patients' data, and "q2-20.py" runs it 20 times on random permutations of the data and computes the confidence interval of the accuracy and the regret. The output plots are in the [output](/output) folder.

"q3.py" runs the LinReg Bandit algorithm on patients' data, where for each time step j, we fit a linear regression with <a href="https://www.codecogs.com/eqnedit.php?latex=L^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L^2" title="L^2" /></a> loss and <a href="https://www.codecogs.com/eqnedit.php?latex=L^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L^2" title="L^2" /></a> regularization on data from patient 1 to j-1 and predict the dosage for patient j. "q3-20.py" runs it 20 times on random permutations of the data and computes the confidence interval of the accuracy and the regret. The output plots are in the [output](/output) folder.

## Implementation Assumption

LinRel assumes that there exists a vector <a href="https://www.codecogs.com/eqnedit.php?latex=f&space;\in&space;\mathbb{R}^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;\in&space;\mathbb{R}^n" title="f \in \mathbb{R}^n" /></a>, such that for each choice <a href="https://www.codecogs.com/eqnedit.php?latex=i&space;\in&space;\{0,&space;1,&space;...,&space;K-1\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i&space;\in&space;\{0,&space;1,&space;...,&space;K-1\}" title="i \in \{0, 1, ..., K-1\}" /></a> (K choices in total),
contextual information <a href="https://www.codecogs.com/eqnedit.php?latex=z_i&space;\in&space;\mathbb{R}^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_i&space;\in&space;\mathbb{R}^n" title="z_i \in \mathbb{R}^n" /></a> is given, and 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{E}[x_i]=f&space;\cdot&space;z_i." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{E}[x_i]=f&space;\cdot&space;z_i." title="\mathbb{E}[x_i]=f \cdot z_i." /></a>

In this implementation, if the patient's data is <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;\in&space;\mathbb{R}^d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;\in&space;\mathbb{R}^d" title="x \in \mathbb{R}^d" /></a>, then we let

<a href="https://www.codecogs.com/eqnedit.php?latex=z_i&space;=&space;\begin{pmatrix}&space;x,&space;1_{i=1},&space;1_{i=2},&space;\cdots,&space;1_{i=K-1}&space;\end{pmatrix}^T&space;\in&space;\mathbb{R}^{d&plus;K-1}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_i&space;=&space;\begin{pmatrix}&space;x,&space;1_{i=1},&space;1_{i=2},&space;\cdots,&space;1_{i=K-1}&space;\end{pmatrix}^T&space;\in&space;\mathbb{R}^{d&plus;K-1}." title="z_i = \begin{pmatrix} x, 1_{i=1}, 1_{i=2}, \cdots, 1_{i=K-1} \end{pmatrix}^T \in \mathbb{R}^{d+K-1}." /></a>

## Observation

SupLinRel has an accuracy score of [52%](output/SupLinRel_Accuracy_confidence_interval.png). The [SupLinRel accuracy history](output/SupLinRel_Accuracy_history.png) plot shows that SupLinRel takes a long time to explore options, and the accuracy slowly increases after seeing 2000 patients. It predicts about [60%](output/SupLinRel_Running_accuracy.png) of the last 500 patients correctly.

LinReg bandit has an accuray score of [63.5%](output/linreg_Accuracy_confidence_interval.png). The [LinReg accuracy history](output/LinReg_Accuracy_history.png) shows that LinReg learns much faster and makes better predictions early on.

Thus, viewing LinReg bandit as the best linear bandit algorithm, SupLinRel encourages exploration and approaches to the best possible performance with large amount of data. SupLinRel performs well on this dataset since the linear regression in LinReg bandit is fitted with the true values of the dosages, and SupLinRel is given much less information: it only knows whether the chosen dosage range (low, medium, or high) is right or wrong each time.


## Reference

P. Auer. [Using confidence bounds for exploitation-exploration trade-offs][1]. The Journal of Machine Learning Research, 3:397-422, 2002. 

[1]:http://www.jmlr.org/papers/volume3/auer02a/auer02a.pdf 

I. W. P. Consortium. [Estimation of the warfarin dose with clinical and pharmacogenetic data][3]. New England Journal of Medicine, 360(8):753-764, 2009.

[3]:/data/appx.pdf 

[Stanford CS 234 Winter 2020 Default Final Project: Estimation of the Warfarin Dose][2]
 
[2]:http://web.stanford.edu/class/cs234/default_project/default_project.pdf 
