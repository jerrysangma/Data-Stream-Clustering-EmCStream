# Datasets

## Artificial Datasets
All artificial stream datasets used in this study are created by *DSD\_RandomRBFGeneratorEvents* function of **streamMOA** package of **R**. 50,000 data instances are received from the streams and saved to files. The algorithms are fed by the data instances one by one, simulating a real-time data stream. In this study, we focus on high dimensional data, hence we have created high dimensional, artificial stream datasets. One of the data streams is stationary, which means it is without concept drift. Other data streams evolve with different speeds, which means they include concept drift. Eight streams are created without noise and three streams are created with different levels of noise. We have specified properties of the streams in such a way that it is possible to observe the effects of change in *k*, dimensions, speed of the concept drift and noise level. Details of the data streams are as follows, where $k$ indicates *number of clusters*.


| Streams     |    length   |   k   |   d   |  drift speed*           | noise level   |
|-------------|-------------|-------|-------|-------------------------|---------------|
| Stream 1    |    50.000   |  10   |  50   |  100                    | 0             |
| Stream 2    |    50.000   |  10   |  100  |  100                    | 0             |
| Stream 3    |    50.000   |  10   |  10   |  100                    | 0             |
| Stream 4    |    50.000   |  20   |  50   |  100                    | 0             |
| Stream 5    |    50.000   |  4    |  50   |  100                    | 0             |
| Stream 6    |    50.000   |  10   |  50   |  40   (high speed)      | 0             |
| Stream 7    |    50.000   |  10   |  50   |  400  (low speed)       | 0             |
| Stream 8    |    50.000   |  10   |  50   |  99999 (no drift)       | 0             |
| Stream 9    |    50.000   |  10   |  50   |  100                    | 0.05          |
| Stream 10   |    50.000   |  10   |  50   |  100                    | 0.10          |
| Stream 11   |    50.000   |  10   |  50   |  100                    | 0.20          |

We have also created 5 different artificial data streams with changing number of clusters. Details of these streams are as follows.

| Streams      |    length   |   k    |  d  |  drift speed*  | noise level   |
|--------------|-------------|--------|-----|----------------|---------------|
| Changing k 1 |    50.000   |  10+-2 |  10 |  100           | 0             |
| Changing k 2 |    50.000   |  10+-2 |  20 |  100           | 0             |
| Changing k 3 |    50.000   |  10+-2 |  50 |  100           | 0             |
| Changing k 4 |    50.000   |  20+-4 |  10 |  100           | 0             |
| Changing k 5 |    50.000   |  4+-1  |  10 |  100           | 0             |

\*: Kernels move a predefined distance of 0.01 every *speed* points.

__Note__ : \*_X.txt files include only data, \*_labels.txt files include only labels, \*.csv files include both the data and the labels (last column icludes the labels.)

## Real Datasets
### Meteorological Data
We have composed three real meteorological stream datasets using weather data from https://www.renewables.ninja/ . (Their two datasources are NASA MERRA reanalysis and CM-SAF's SARAH dataset). One of the datasets includes meteorological data of two cities from Turkey, other one includes two cities from Europe and the last dataset includes two cities from US. For each dataset, we chose two cities that have different climate characteristics to create separable datasets. All datasets consist of hourly measurements of five years, from Jan 1<sup>st</sup>, 2015, till Dec 31<sup>st</sup>, 2019. There exist 43,825 measurements for each city. Each dataset consists of two cities, therefore each dataset includes 87,650 instances. Every instance includes six features, which are temperature, precipitation, snowfall, snow mass, air density and cloud cover. Temperature and air density are two features that are always meaningful and distinguishing. However, other features are not always distinguishing. For example, precipitation is not a distinguishing parameter when it is not raining and snowfall or snow mass are not meaningful when it is not snowing. For this reason, we have defined an extra weight to temperature and air density. In order to apply this weight, we have doubled these two features, after normalizing the whole data. 

For creating a dataset, we have merged the instances of two cities according to date and time of the measurement. By this way, the datasets are real stream datasets. Moreover, these are evolving data streams by nature (they include concept drift) because weather data changes both in a day of 24 hours, from daytime to night, and in a year, from season to season. Using such data streams, it is possible to focus either on concept drift that occurs every day, or on concept drift that occurs from season to season.

__Note__ : \*_X.txt files include only data, \*_labels.txt files include only labels, \*.csv files include both the data and the labels (last column icludes the labels.)

### Keystroke Data
We have used three subsets of the larger CMU dataset which is created by typing characteristics of 51 users. The participants type the password ".tie5Roanl" and the *Enter* key 400 times, captured in eight different sessions in different days, 50 times on each session. The original keystroke dataset contains 20,400 instances and each instance consists of 31 features. Keystroke dataset incrementally evolve due to the participants' practice. 

We have created three subsets of the Keystroke dataset, getting all features of some participants. We did not change, eliminate or convert any of the features. We have specified *k (number of clusters)* as two, three and four. Because each participant has 400 records, the subsets that we have used have 800, 1,200 and 1,600 instances respectively. 

__Note__ : \*_X.txt files include only data, \*_labels.txt files include only labels, \*.csv files include both the data and the labels (last column icludes the labels.)

Followind table presents a summary of the characteristics of the real datasets:

| Stream | $k$ | dimensions | instances | evolving |
|--------|-----|------------|-----------|----------|
| Weather-Turkey | 2 | 6  | 87,650 | yes |
| Weather-Europe | 2 | 6  | 87,650 | yes |
| Weather-US     | 2 | 6  | 87,650 | yes |
| Keystroke-2    | 2 | 31 | 800   | yes |
| Keystroke-3    | 3 | 31 | 1,200  | yes |
| Keystroke-4    | 4 | 31 | 1,600  | yes |