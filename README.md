# Semi-Supervised Sequence Modeling for Elastic Impedance Inversion
[Motaz Alfarraj](http://www.motaz.me), and [Ghassan AlRegib](http://www.ghassanalregib.info)

Codes and data for a manuscript publoshed in Interpretation Journal, Aug 2019. 

This repository contains the codes for the paper: 

M. Alfarraj, and G. AlRegib, "**Semi-Supervised Sequence Modeling for Elastic Impedance Inversion**," in *Interpretation*, Aug. 2019.[[SEG Digital Library]](https://library.seg.org/doi/abs/10.1190/int-2018-0250.1)


## Abstract
Recent applications of machine learning algorithms in the seismic domain have shown great potential in different areas such as seismic inversion and interpretation. However, such algorithms rarely enforce geophysical constraints — the lack of which might lead to undesirable results. To overcome this issue, we have developed a semisupervised sequence modeling framework based on recurrent neural networks for elastic impedance inversion from multiangle seismic data. Specifically, seismic traces and elastic impedance (EI) traces are modeled as a time series. Then, a neural-network-based inversion model comprising convolutional and recurrent neural layers is used to invert seismic data for EI. The proposed workflow uses well-log data to guide the inversion. In addition, it uses seismic forward modeling to regularize the training and to serve as a geophysical constraint for the inversion. The proposed workflow achieves an average correlation of 98% between the estimated and target EI using 10 well logs for training on a synthetic data set.

## Sample Results 

#### Estimated EI Section
Incident Angle (degrees)|Estimated EI|True EI|Absolute Difference|
|:--:|:--:|:--:|:--:|
0|![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_inv_0.png)| ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_0.png) | ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_diff_0.png)
10|![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_inv_1.png)| ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_1.png) | ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_diff_1.png)
20|![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_inv_2.png)| ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_2.png) | ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_diff_2.png)
30|![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_inv_3.png)| ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_3.png) | ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_diff_3.png)

#### Scatter plots 
|0 degrees|10 degrees|20 degrees|30 degrees|
|:--:|:--:|:--:|:--:|
| ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/Scatter_0.png) | ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/Scatter_1.png) | ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/Scatter_2.png) | ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/Scatter_3.png) 

#### Sample traces 
|x=3300 meters|x=8500 meters|
|:--:|:--:|
| ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_trace_3300m.png) | ![](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/images/EI_trace_8500m.png)

## Data 
The data used in this code are from the elastic model of [Marmousi 2](https://library.seg.org/doi/abs/10.1190/1.1817083)
The synthesis of the seismic data is described in the [paper](https://library.seg.org/doi/abs/10.1190/int-2018-0250.1) 

The data file should be downloaded automatically when the code is run.

Alternatively, you can download the data file manually at this [link](https://www.dropbox.com/s/66u2hbbrvc15lyp/data.npy?raw=1) and place it in the same folder as main.py file 

Both elastic impedance and seismic are saved in the same `data.npy` file.

## Running the code

### Requirements: 
These are the python libraries that are needed to run the code. Newer version should work fine as well. 
```
bruges==0.3.4
matplotlib==3.1.1
numpy==1.17.0
pyparsing==2.4.1.1
python-dateutil==2.8.0
torch==1.1.0
torchvision==0.3.0
tqdm==4.33.0
wget==3.2
```

Execute this command to install the required libraries, 
`pip install -r requirements.txt`

or install them manually from [requirements.txt](https://github.com/olivesgatech/Elastic-Impedance-Inversion-Using-Recurrent-Neural-Networks/blob/master/requirements.txt) file. 


### Training and testing

To train the model using the default parameters (as reported in the paper), and test it on the full Marmousi 2 model, run the following command: 

```bash 
python main.py
```
 However, you can choose those parameters by including the arguments and their values. For example, to change the number of training traces, you can run: 
 
```bash 
python main.py -num_train_wells 10
```

The list arguments can be found in the file `main.py`.  



## Citation: 

If you have found our code and data useful, we kindly ask you to cite our work 
```tex
@article{alfarraj2019semi,
  title={Semi-supervised Sequence Modeling for Elastic Impedance Inversion},
  author={Alfarraj, Motaz and AlRegib, Ghassan},
  journal={Interpretation},
  volume={7},
  number={3},
  pages={1--65},
  year={2019},
  publisher={Society of Exploration Geophysicists and American Association of Petroleum~…}
}

```
