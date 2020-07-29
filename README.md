# NASA_Internship_2020
![NASA Internship Logo](pictures/nasa_internship_2020_logo.png)
## Project
### Machine Learning for Precipitation Classification using Synthesized Passive Microwave Data

The Global Precipitation Measurement (GPM) Core Observatory's Microwave Imager (GMI) and Dual-Frequency Precipitation Radar (DPR) together provide ample information on global precipitation characteristics. As an active sensor in particular, DPR provides an accurate precipitation flag assignment, while passive sensors like GMI were traditionally believed not to be able to tell apart. Precipitation flag (precipitating or not; stratiform or convective) is a key parameter for us to make better retrieval of precipitation characteristics as well as understand the physical cloud-precipitation processes.

Using collocated precipitation flag assignment from DPR as the “truth”, this project employs machine learning models to train and test the predictability and accuracy of using passive GMI-only observations together with some ancillary atmosphere information from reanalysis. Precipitation types are classified into the following classes: convective, stratiform, convective-stratiform mixed, no precipitation, and other precipitation. A variety of classification algorithms are tested, including Support Vector Machines, Naive Bayes, Random Forests, Gradient Boosting, and Neural Networks (Multilayer Perceptron Network), and their results are evaluated and compared. As a proof of concept, global statistics and a squall line case study will be presented.

 

* DPR: Dual-Frequency (Ku-Ka band) Precipitation Radar
* GMI: Multi-Channel (10-183 GHz) GPM Microwave Imager
* Combined Radar-Radiometer Retrieval

GPM Core Observatory       |  Scan Details
:-------------------------:|:-------------------------:
![](pictures/GPM_GMI_DPR_data_collection.png)  |  ![](pictures/Satellite%20Data%20Visualization.png)

![](pictures/GMI_Characteristics.png)


## Sources 
* G. Skofronick Jackson, 2nd NOAA User Workshop on the GPM Mission, Nov 29, 2011, College Park, MD (https://www.star.nesdis.noaa.gov/star/documents/meetings/GPM2011/dayOne/Skofronick-JacksonG.pdf)
* MiRS Sensors GPM GMI Overview (https://www.star.nesdis.noaa.gov/mirs/gpmgmi.php)
* NOAA GPM Users Workshop, April 2-3, 2013 (https://www.star.nesdis.noaa.gov/star/documents/meetings/GPM2013/dayOne/Skofronick-Jackson.pdf)
