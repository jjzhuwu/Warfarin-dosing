# Warfarin-dosing
 
This project implements the LinRel and SupLinRel algorithms from the paper [Using Confidence Bounds for
Exploitation-Exploration Trade-offs][1] by P. Auer, and applies these reinforcement learning algorithms to predict Warfarin dosing given patients' conditions (heights, weights, etc). 

Problem description can be found [here](project_description.pdf), which is downloaded from [here][2]. The dosages are classified as low (less than 21), mediun (between 21 and 49), and high (greater than 49).

## Running the Programs

First, download all files in this repository.

The following command will run the file "filename.py". Pandas, NumPy, Sklearn, and Scipy are needed.

```bash
python3 filename.py
```
"q1.py" runs the fixed dose algorithm and two algorithms proposed by Consortium (2009). The details of these algorithms can be found [here][3].

"q2.py" runs the SupLinRel algorithm on patients' data, and "q2-20.py" runs it 20 times on random permutations of the data and computes the confidence interval of the accuracy and the regret. The output plots are in the [output](/output) folder.

"q3.py" runs the LinReg Bandit algorithm on patient's data, where for each time step i, we fit a linear regression with <a href="https://www.codecogs.com/eqnedit.php?latex=L^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L^2" title="L^2" /></a> loss and <a href="https://www.codecogs.com/eqnedit.php?latex=L^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L^2" title="L^2" /></a> regularization on data from patient 1 to i-1 and predict the dosage for patient i. "q3-20.py" runs it 20 times on random permutations of the data and computes the confidence interval of the accuracy and the regret. The output plots are in the [output](/output) folder.



## Reference

[1]:http://www.jmlr.org/papers/volume3/auer02a/auer02a.pdf
[2]:http://web.stanford.edu/class/cs234/default_project/default_project.pdf
[3]:/data/appx.pdf
