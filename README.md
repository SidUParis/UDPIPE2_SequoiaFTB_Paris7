# Step by Step Tutorial for UDPIPE2 Train & Predict

Author: XU SUN, Université Paris Cité, 
Email: xu.sun@etu.u-paris.fr  




## General Step 1: Setup

**CONDA RECOMMENDED, it needs at least two different env to use Udpipe2**

1. Clone the UDPIPE 2 repository from GitHub:
    ```sh
    git clone https://github.com/ufal/udpipe.git udpipe2
    ```
2. Navigate to the cloned folder:
    ```sh
    cd udpipe2
    ```
3. Switch to the `udpipe-2` branch:
    ```sh
    git checkout udpipe-2
    ```
    You should see the following message:  
    `Branch 'udpipe-2' set up to track remote branch 'udpipe-2' from 'origin'. Switched to a new branch 'udpipe-2'`
4. Remove the broken `wembedding_services` folder:
    ```sh
    rm -rf wembedding_service/
    ```
5. Clone the `wembedding_service` repository from GitHub:
    ```sh
    git clone https://github.com/ufal/wembedding_service.git
    ```
6. Now all the necessary scripts are downloaded, and we can start training!

## General Step 2: Training Setup

**You need at least one trained model to make predictions. This step requires Python 3.8.**

1. **Training requires TensorFlow 1.** If you are using an NVIDIA GPU, use NVIDIA-TensorFlow instead of TensorFlow-GPU 1.15 mentioned in the `requirements.txt`. Check [NVIDIA TensorFlow GitHub](https://github.com/NVIDIA/tensorflow) for more details.

    Here is how to install and use NVIDIA-TensorFlow:

    1.1 Install NVIDIA PyIndex:
        ```
        pip install nvidia-pyindex
        ```
    
    1.2 Install NVIDIA TensorFlow:
        ```
        pip install nvidia-tensorflow
        ```
    
    If you encounter installation issues, recreate a conda environment with Python 3.8:
        

        conda create --name YOUR_ENV_NAME python=3.8

        conda activate YOUR_ENV_NAME
        

    Verify GPU availability:
        ```python
        import tensorflow as tf
        print(tf.test.is_gpu_available())
        ```

2. When install nvidia-tensorflow, the relevant packages for running TensorFlow with your GPU should be well installed.
3. Install necessary ufal packages:
    ```sh
    pip install ufal.chu_liu_edmonds ufal.udpipe
    ```

4. You may encounter compatibility issues with libraries like `protobuf`. Install the required versions as needed:
    ```sh
    pip install protobuf=1.5
    ```

5. At this point, we have everything prepared for training except embeddings.

## General Step 3: Compute Embeddings

**Computing embeddings requires TensorFlow 2.x, which is not compatible with training. Create a new conda environment with Python 3.9.**

1. Create and activate a new environment:
    ```sh
    conda create --name EXAMPLE python=3.9
    conda activate EXAMPLE
    ```
2. Install TensorFlow 2.13.1:
    ```sh
    conda install tensorflow=2.13.1=cuda118py39h8710ada_1 -c conda-forge
    ```
3. Install `transformers` library:
    ```sh
    conda install transformers=4.24.4 -c huggingface
    ```

    Ensure GPU is detected by TensorFlow:
    ```python
    import tensorflow as tf
    print(tf.test.is_gpu_available())
    ```

4. UDPIPE2 can process embeddings for all .conllu files in a folder using the compute_embedding.sh script provided in the original UDPIPE2 GitHub repository. To avoid memory explosion errors, we compute embeddings one by one and remove each after usage. Here’s how to process them individually:
    ```sh
    python3 compute_wembeddings.py --format=conllu YOUR_FILE.conllu YOUR_FILE.conllu.npz
    ```

5. After computing the embeddings, you will have a `.conllu.npz` file in the same directory as your original `.conllu` file.For example:
    ```sh
    ls
    sequoiaftb.mwe_reg.surf.dev.conllu
    sequoiaftb.mwe_reg.surf.dev.conllu.npz
    sequoiaftb.mwe_reg.surf.train.conllu
    sequoiaftb.mwe_reg.surf.train.conllu.npz
    ```
6. The .npz files are the embedding files used during training and prediction. They must have the same prefix as the original .conllu file. For example, if the input is sequoiaftb.mwe_reg.surf.dev.conllu, ensure there is an embedding file named sequoiaftb.mwe_reg.surf.dev.conllu.npz. Otherwise, errors will occur. **Be careful when defining the argument in step 4!**

## Training

**If you already have a trained model, skip to the Prediction section.**

1. Switch back to the training environment with TensorFlow 1.15.

    Ensure the correct pip and Python are used to avoid conflicts.

2. Train the model:
    ```sh
    python ./udpipe2.py ../MODEL_LR_0.001_DP_0.4_EPOCHS_20_SEQUOIAFTB --train ../datasets_sequoia/sequoiaftb.mwe_reg.surf.train.conllu --dev ../datasets_sequoia/sequoiaftb.mwe_reg.surf.dev.conllu --epochs 20:0.001 --dropout 0.4
    ```

    - `MODEL_LR_0.001_DP_0.4_EPOCHS_20_SEQUOIAFTB` is an empty folder indicating training from scratch. If a model exists in this folder, it will be fine-tuned with your datasets.
    - For more arguments, run:
        ```sh
        python ./udpipe2.py -h
        ```

## Prediction

Now that you have a trained model saved in the `MODEL_LR_0.001_DP_0.4_EPOCHS_20_SEQUOIAFTB` folder, you can use it for parsing and tagging.

1. Ensure your new data is in the `.conllu` format. Convert any other formats (e.g., `.txt`) to `.conllu`.
    - Hint: Manually create a null `.conllu` file following the `.conllu` format with no values, only `-` or `null`.

2. Compute embeddings for your dataset (see General Step 3, section 2).

3. With both `.conllu` and `.npz` files in the dataset folder, you can start prediction:
    ```sh
    python ./udpipe2.py MODEL_PATH --predict --predict_input DATASET_PATH/YOUR_FILE.conllu --predict_output DESIRED_PATH/FILENAME
    ```
    *The MODEL_PATH is the folder where you save your model. For example, if you download a model named MODEL_LR_0.001_DP_0.4_EPOCHS_20_SEQUOIAFTB, this folder will contain checkpoints, model binaries, etc. The MODEL_PATH is the directory to this folder.*

    - Replace `MODEL_PATH` with your trained UDPIPE2 model path.
    - Replace `DATASET_PATH/YOUR_FILE.conllu` with your input data path. Ensure the `.conllu` file has a corresponding `.npz` file in the same directory.
    - Customize the `DESIRED_PATH/FILENAME` for the prediction output file.
