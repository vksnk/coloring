<h2> Environment and dependencies </h2>
It's convenient to use one of the virtual environments for Python to keep everything neat. For example, after installing miniconda, first you need to create an environment using:

```
conda create --name cs224w-project python=3.13
```

then activate it:

```
conda activate cs224w-project
```

and install necessary dependencies:

```
pip install -r requirements.txt
```

and you should be good to go (might need to add ` -i https://pypi.org/simple` to the `pip install`).

**Please note that at the moment of writing Python 3.14 is not supported, because some of the dependencies such as PyG haven't added support for it yet.**

<h2> Training </h2>
Example of the command-line to train the model:

```
python train.py --hidden_dim=128 --num_of_gcns=7 --num_classes=8 --conv_type=sage
```

train.py accepts the following parameters:

* input_dim = size of the input dimension.
* hidden_dim = size of the hidden dimension.
* num_of_gcns = number of convolution layers to use.
* conv_type = type of the convolution layer to use. The options are: "sage", "gcn", "gin", "gat".
* num_classes = the size of the output vector (i.e. maximum number of colors to use).
* draw_model = saves a graphical visualization of the model into file.

**Each parameter has a reasonable default option, so providing them through command-line is optional.**

<h2> Evaluation </h2>
Example of the command-line to evaluate the model:

```
python evaluate.py --checkpoint=checkpoints/best_checkpoint_7_16_256.pth --num_of_gcns=7 --hidden_dim=256
```

In addition to the evaluate.py accepts the following parameters:

* checkpoint = path to the checkpoint to evaluate.

**If model was trained with custom parameters it's important to pass the same parameters to the evaluate.py script**

<h2> Visualization. </h2>

You can visualize various information stored in the checkpoint file (such as training and validation losses, number of correct predictions, etc) using the following command:

```
python plots.py -o losses.png experiments/checkpoint_num_gcn_sweep/best_checkpoint_7_16_256.pth
```