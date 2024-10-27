# Quantitative Evaluation of Dimensionality Reduction Techniques for Out-of-Sample Data

[code](https://github.com/Kpure1000/OOS-DR-Benchmark)

## Datasets

### Synthetic

1. Coordinated Plane

2. Swissroll Cut by Plane

3. Digit 8 

TBC...

### Real world

Scenario 1: Visual Cluster Analysis

7 dataset with class label

|name|type|N_samples|N_dims|classes|
|-|-|-|-|-|
|letter|images|6450|512|?|
|fashionMNIST|images|4200|784|?|
|fmd|images|997|?|?|
|secom|tables|1567|?|?|
|cnae9|text|1080|856|9|
|spambase|text|4601|57|2|
|hatespeech|text|2973|?|2|

Scenario 2: Distribution Diff 

6 dataset with distribution difference

|name|type|N_samples|N_dims|classes|KLdiv|
|-|-|-|-|-|-|
|raid|images|1955|?|?|?|
|sensor|tables|7213|?|?|?|
|dendritic|tables|576|?|?|?|
|merchant|text|3897|?|?|?|

#### Scenario 3: Intrinsic Structure Analysis

4 dataset with obvious manifold structure

|name|type|N_samples|N_dims|
|-|-|-|-|
|isomapFace|images|698|512|
|headpose|images|558|1536|
|coil20|images|1440|16384|
|bison|tables|5000|3|
