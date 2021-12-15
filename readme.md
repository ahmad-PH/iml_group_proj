## Automatic Library of Congress Classification Code

---

### Running the training code for non-sequential model

**Starting point*  *
The main notebook for running all the models are in the notebook "report/Library of Congression Classification.ipynb".  
Note that the training process required preprocessed embeddings data which lies in "github_data" folder. 


**Preprocessed Embeddings/Featuers**  
Please ensure that the data is downloaded and placed in the "github_data" folder.  
The code for preprocessing these embeddings can be found in files under "runner" folder.

**Caching**  
Note that once each model finishes fitting to the data, the code also stored the result model as a pickle file in the "_cache" folder.



### Training code for sequential model

The training of LSTM on BERT embeddings were all done in Google Collab. 
These notebooks were then saved as jupyter notebook, and stored in this repository. 
To view the result, please view the notebooks in "report/rnn" folder.


![screenshot_rnn_1](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/rnn_notebook_screenshot_1.png?raw=true)

![screenshot_rnn_2](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/rnn_notebook_screenshot_2.png?raw=true)



The rnn codes (LSTM, GRU) can also be found in iml_group_proj/model/bert_[lstm|gpu].py
