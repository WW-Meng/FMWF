# Source code and dataset for our paper "Beyond Single Tabs: A Transformative Few-Shot Approach to Multi-Tab Website Fingerprinting Attacks"

## Dataset Description:
We collected Tor single-tab and multi-tab browsing datasets under real browsing scenarios and our dataset is about 11G in size. You can download the dataset via Google Cloud Drive [link](https://drive.google.com/file/d/1S_fiEatE8oy054iqeNusdqHbXn1Qs1xH/view?usp=drive_link).
Our datasets are in .csv format. you can load them using the following code:

~~~ Python
import pandas as pd
df = pd.read_csv('../tor5tab_dataset.csv', header=None)
~~~
Each row in the dataset represents a traffic instance. For the single-label dataset, the first item in each row is the index, the second is the repetition count, and the subsequent items are the traffic data. In the multi-label dataset, the first 100 items of each row are the multi-hot encoding of the traffic, followed by the index of the corresponding monitored website, and the remaining items are the traffic data.

