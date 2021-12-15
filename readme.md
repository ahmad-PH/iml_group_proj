## Automatic Library of Congress Classification Code

---
### Dependencies
To run our code, you need the following packages:
```
scikit-learn=1.0.1
pytorch=1.10.0
python=3.9.7
numpy=1.21.2
notebook=6.4.6
matplotlib=3.4.3
gensim=4.0.1
tqdm=4.62.3
```

### Checklist

1. Install python packages with [requirements.txt](https://github.com/ahmad-PH/iml_group_proj/blob/main/requirements.txt)

```
$ pip install -r requirements.txt
```

2. Set PYTHONPATH to the root of this folder by running the command below at the root directory of the project.

```
$ export PYTHONPATH=$(PWD)
```

3. Ensure that there are data in "github_data" 

If not, use the runner python scripts in "runner" folder to create features.

*Build all the features* 

Use the command below to build all the features. The whole features preparation steps take around 2.5 hours.

```{shell}
$ python runner/build_all_features.py
```


Due to its large memory consumption, the process might crash along the way.
If that's the case, please try again by running the same command. The script is able to pick up on where it left of.

*Build each feature separately*

*BERT embeddings*

```{shell}
$ python runner/build_bert_embeddings.py --model_size=small  
```

Note that as the whole process requires large amount of memory, the process might crash halfway.

*Word2Vec embeddings*

Please download directly from our Google Drive. [[Link](https://drive.google.com/drive/folders/1B-XNvIdGZazLvDjnH2xWGUBfoe-Jt53B?usp=sharing)]
<!-- ```{shell} -->
<!-- $ python runner/build_bert_embeddings.py --model_size=small -->  
<!-- ``` -->


*tf-idf features*

```{shell}
$ python runner/build_tfidf_features.py
```

If the download still fails, then please download the data directly from our Google Drive [[Link](https://drive.google.com/drive/folders/1B-XNvIdGZazLvDjnH2xWGUBfoe-Jt53B?usp=sharing)] (BERT small and large unavailable).


### Running the training code for non-sequential model

**Starting point**  
The main notebook for running all the models is in this notebook [[Link](https://github.com/ahmad-PH/iml_group_proj/blob/main/report/Library%20of%20Congression%20Classification.ipynb)].  
Note that the training process required preprocessed embeddings data which lies in "github_data" folder. 

**Caching**  
Note that once each model finishes fitting to the data, the code also stored the result model as a pickle file in the "_cache" folder.

![nonsequential_notebook_screenshot_1.png](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/nonsequential_notebook_screenshot_1.png?raw=true)


### Training code for sequential model

The training of LSTM on BERT embeddings were all done in Google Collab. 
These notebooks were then saved as jupyter notebook, and stored in this repository. 
To view the result, please view the notebooks in "report/rnn" folder (e.g., [[Link](https://github.com/ahmad-PH/iml_group_proj/blob/main/report/rnn/LibOfCongress_LSTM.ipynb)].


![screenshot_rnn_1](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/rnn_notebook_screenshot_1.png?raw=true)

![screenshot_rnn_2](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/rnn_notebook_screenshot_2.png?raw=true)


The rnn codes (LSTM, GRU) can also be found in iml_group_proj/model/bert_[lstm|gpu].py
