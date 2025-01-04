#  Extensive Study of Learning-based Models for VD

## Prerequisite
Before you run any code, you need to prepare the following things.
### Clone the latest version of GloVe and install
    
```shell
cd Graph
git clone https://github.com/stanfordnlp/GloVe 
cd Glove && make
```
### Build the Vul4C dataset from scratch, or download our built checkpoints

To build from scratch see this section

Download the Vul4C dataset we have constructed 
- part-01 https://file.io/MwQIZYpqcUOd
- part-02 https://file.io/ckk1tlanzQnX
```shell
mkdir vul4c_dataset && cd vul4c_dataset && unrar vul4c_dataset.part01.rar
```

### Register for an OpenAI account before using ChatGPT
First you need register an OpenAI account for the OpenAI API service
https://platform.openai.com/signup

Then create a new secret key at
https://platform.openai.com/account/api-keys

Please note that you have a `$20` free balance to use.

Then you need to go to `openai.api_key` in `LLM/chatgpt.py` to set the API_KEY

### Install Joern for build Vul4C from scratch[optional]
Before you build Joern, please make sure you have installed sbt and scala, we recommend install them from SDKMAN.
```shell
cd Vul4C
git clone https://github.com/joernio/joern
cd joern && sbt
cp ../Vul4CTest.scala ./joern-cli/frontends/c2cpg/src/test/scala/io/joern/c2cpg/io/Vul4CTest.scala
cp ../run_joern.py ./run_joern.py
```

### Github API Key[optional]
In order to download commit files from GitHub, the official API is used and the API KEY needs to be configured.

Login in GitHub and go to https://github.com/settings/tokens generate a new token

### Build Tree-sitter
We modified the tree-sitter, which extend original grammar rules.

The following command build tree-sitter for c and cpp version
```
cd Vul4C
cd tree-sitter-c && npm run build
cd ..
cd tree-sitter-cpp && npm run build
```

## Directory structure
```
├── Graph    # graph-based models source code
├── LLM      # ChatGPT source code
├── RQ       # different RQ settings
├── Vul4C    # dataset build source code
├── sequence # sequence-based models source code
└── vul4c_dataset # downloaded dataset
```

## Vul4C
Vul4C requires multiple steps to be performed to collect the dataset, 
each of which saves intermediate files in case the dataset needs to be rebuilt from scratch due to the network errors, API exceed limits.

1. Run `crawl_cve_from_nvd.py` to download all CVEs from the NVD database and save them to a JSON file.
2. Run `extract_cve_info.py` to extract vulnerability commit URL in CVE, and automatically download the commit files in local.
3. Run `extract_file_diff.py` to extract function in file, find which function is vulnerable, and save other functions as non-vulnerable.
4. Using scala test-suite and JVM multithreading to generate joern graph for Vul4C, `python joern/run_joern.py -java /path/to/java -scala /path/to/scala -sbt /path/to/sbt -working_dir ./ -timeout 60 `
5. Run `extract_graph.py` to extract joern graph 
6. Run `vul4c_split.py` to get final Vul4C dataset, which has been split by train/valid/test set.

## Experiment
###  D1: Capabilities of Learning-based Models for Vulnerability Detection
This experiment investigates the vulnerability detection performance of different types of models
#### train/inference graph-based models


all graph-based models are saved in `Graph/models` directory.
```
Graph
├── GloVe
├── models
│   ├── __init__.py
│   ├── devign
│   ├── ivdetect
│   ├── reveal
```
You should run `Graph/scripts/process_dataset.py` to get Word2Vec and GloVe embeddings.

To train graph-based models, you can configure the training parameters via config.json(e.g. train batch size).
Then run `python main.py --dataset=vul4c_dataset` 

After training is complete, the model's checkpoint file will be saved in `Graph/storage/result`.

You need to modify `main.py`, fill in the name of the checkpoint directory, and then modify `config.json` `do_test=True`, and run it again to test the model.

All performance metrics will be reported in the log file. e.g. `Graph\storage\results\devign\vul4c_dataset\202307251718_v1\train.log`

#### train/inference sequence-based models
LineVul and SVulD sequence-based models are saved in `sequence` directory.

Before you train the sequence-based models, you need to preprocess the dataset e.g. run `sequence/LineVul/preprocess.py` 
```
sequence
├── LineVul
└── SVulD
```

To train and inference these models use the shell scripts `train.sh` and `test.sh`.

#### inference on large language model(ChatGPT)

We encapsulate the different example selection strategies(in-contex learning and chain-of-thought) for using chatgpt, using `chatgpt_run.py` to set different settings for running.

#### top25_cwe and seven dangerous types of vulnerabilities

After the model inference is completed `test.json` is saved recording the id and inference result, which is convenient for us to calculate the performance on each CWE category. In order to obtain such results please refer to the content in `RQ/RQ2_1`

#### result (for more detailed results see the paper)
|  Models  |   Accuracy  |    Recall   |  Precision  |      F1     |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|
|  Devign  |    0.742    |    0.622    |    0.068    |    0.122    |
|  Reveal  |    0.780    |    0.545    |    0.070    |    0.125    |
| IVDetect |    0.792    |    0.582    |    0.080    |    0.141    |
|  LineVul |    0.962    |    0.593    |    0.117    |    0.195    |
|   SVulD  |    0.820    |    0.637    |    0.100    |    0.172    |
|  ChatGPT | 0.932±0.015 | 0.125±0.020 | 0.057±0.014 | 0.078±0.016 |


###  D2: Interpretation of Learning-based Models for Vulnerability Detection
We use different interpretability techniques for different models, e.g., GNNExplainer for Devin and attention mechanism for LineVul.

We obtained the attention value of each token as much as possible and visualized it using HTML, an example of LineVul visualization is shown below

![arc](imgs/linevul_15.jpg)
<p align="center"> LineVul interpretation result on CVE-2016-15006 </p> 

To get these visualizations, you need to modify the configuration file(e.g. `config.json`) on the trained models so that they run the `do_interpret`.

In order to obtain the degree of attention the model pays to different statement types, we use [tree-sitter](https://github.com/tree-sitter/tree-sitter)

We also saved thel last hidden vector of the final output of the model before making the binary classification and saved it as `test_tSNE_embedding.pkl`.
Then  use [sklearn](https://scikit-learn.org/)'s tSNE model to explore class separation performance.

See more code under `RQ/RQ4`

#### result
![](imgs\statement_types.svg)
![](imgs\tSNE.svg)


### D3: Stability of Learning-based Models for Vulnerability Detection
We use four types of semantic-preserving transformations
- Remove all comments
- Insert comments
- Insert irrelevant code
- Rename all identifiers

Run `python RQ/RQ5/dataset_variant.py` to get four variant dataset, you will see four dataset generate in root project directory.
Follow D1 experiment to train/inference models.

#### result
|          | Devign | Reveal | IVdetect | LineVul | SVulD  | ChatGPT(ICL-same-repo) |
|------------------------|--------|--------|----------|---------|--------|------------------------|
| Raw                    | 0.1219 | 0.1246 | 0.1410   | 0.1952  | 0.1723 | 0.0779                 |
| Remove all comments    | 0.0746 | 0.0988 | 0.0508   | 0.1963  | 0.1744 | 0.0732                 |
| Insert comment         | 0.0752 | 0.0927 | 0.0521   | 0.1864  | 0.1630 | 0.0680                 |
| Insert irrelevant code | 0.0731 | 0.0941 | 0.0547   | 0.1953  | 0.1789 | 0.0885                 |
| Rename all identifier  | 0.0751 | 0.0972 | 0.0247   | 0.1829  | 0.1859 | 0.0663                 |


### D4: Ease of Use of Learning-based Models for Vulnerability Detection.  

We record various features of the model, such as the maximum number of tokens entered and the minimum GPU memory requirement.

#### result
|           | Program Integrity | Compilation | Input Size | Fine-Tuning | Code Availability | Hardware Requirement | Configuration Difficulty | Privacy |
|-----------|-------------------|-------------|------------|-------------|-------------------|----------------------|--------------------------|---------|
| Devign    | ✅                 | ❌           | Medium     | ✅           | ✅                 | >1G                  | Difficult                | Safe    |
| Reveal    | ✅                 | ❌           | Medium     | ✅           | ✅                 | >1G                  | Difficult                | Safe    |
| IVDetect  | ✅                 | ❌           | Medium     | ✅           | ✅                 | >1G                  | Difficult                | Safe    |
| LineVul   | ❌                 | ❌           | Small      | ✅           | ✅                 | >6G                  | Medium                   | Safe    |
| SVulD     | ❌                 | ❌           | Small      | ✅           | ✅                 | >6G                  | Medium                   | Safe    |
| chatgpt   | ❌                 | ❌           | Large      | ❌           | ❌                 | API/Web              | Easy                     | Unsafe  |


### D5: Economy Impact of Learning-based Models for Vulnerability Detection 

We record time information during training/inference via python's built-in time library, and use [torchinfo](https://github.com/TylerYep/torchinfo) to record the size of the model.

#### result
|          | Preprocess Data Time | Training Time | Inference Time | Model Parameter | Cost    |
|----------|----------------------|---------------|----------------|-----------------|---------|
| Devign   | 7,103s               | 2,836s        | 101s           | 0.97M           | 2.8056$ |
| Reveal   | 7,103s               | 5,220s        | 148s           | 1.09M           | 4.4378$ |
| IVDetect | 3,563s               | 13,602s       | 916s           | 1.01M           | 4.8821$ |
| LineVul  | 0s                   | 13,274s       | 322s           | 124.65M         | 6.1712$ |
| SVulD    | 0s                   | 6,048s        | 319s           | 125.93M         | 1.7792$ |
| ChatGPT  |                      |               | 1,263s         | 175,000.00M     | 1.9200$ |