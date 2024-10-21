# Source code and dataset for our paper "Beyond Single Tabs: A Transformative Few-Shot Approach to Multi-Tab Website Fingerprinting Attacks"

## Source Code's Description:
The source code mainly consists of three parts:

### Traffic Capture and Processing
We used the codes `2tab-capture.ipynb`, `3tab-capture.ipynb`, `4tab-capture.ipynb`, and `5tab-capture.ipynb` from the FMWF-src folder to capture multi-label traffic data in .pcap format from Google Chrome and Microsoft Edge browsers. The captured traffic data was then processed into .csv format traffic sequences using the codes `2tab-dataprocess.ipynb`, `3tab-dataprocess.ipynb`, `4tab-dataprocess.ipynb`, and `5tab-dataprocess.ipynb`.

Our Tor traffic data was collected with the voluntary assistance of one of the authors of the paper "Robust Multi-tab Website Fingerprinting Attacks in the Wild" (Published in IEEE S&P 2023), and you can access their paper at this [link](https://www.computer.org/csdl/proceedings-article/sp/2023/933600b005/1OXTJ1ZxbXy).

### Pre-training with Augmented Traffic Sequences
The  `PretrainingModel.py` file in the folder `FMWF-src` is used to obtain the pre-training model using manually synthesized multi-tab datasets.

### Few-shot Fine-tuning and Testing
For the Few-shot Fine-tuning and Testing section, we use the 5tab scenario as an example. The `tor-5tab-Accuracy.py` file in the `FMWF-src` folder is used to test the real-world collected 5tab traffic. This code includes both the Few-shot Fine-tuning for Real-world Adaptation and Testing on Dynamic Traffic stages for the 5tab scenario. You can reproduce our experiments using the fine-tuning and testing datasets from the tor dataset. Please note that the paths in the code may not match your environment, so be sure to replace them accordingly.





## Dataset Description:
We collected Tor single-tab and multi-tab browsing datasets under real browsing scenarios and our dataset is about 11G in size. You can download the dataset via Google Cloud Drive [link](https://drive.google.com/file/d/1S_fiEatE8oy054iqeNusdqHbXn1Qs1xH/view?usp=drive_link).
Our datasets are in .csv format. you can load them using the following code:

~~~ Python
import pandas as pd
df = pd.read_csv('../tor5tab_dataset.csv', header=None)
~~~
Each row in the dataset represents a traffic instance. For the single-label dataset, the first item in each row is the index, the second is the repetition count, and the subsequent items are the traffic data. In the multi-label dataset, the first 100 items of each row are the multi-hot encoding of the traffic, followed by the index of the corresponding monitored website, and the remaining items are the traffic data.

