# ml4gfinal

```python3 train.py``` will do training for the two gene diff input models.

The available model classes are ConvTransNet (CNN-Transformer), BetterConvNet (6-layer CNN), SimpleConvNet (Simple CNN) and TransformerNoConv (model with Transformer-encoder layers only). Currently the model architectures are configured to use their best performing versions.

To interchange between models, change line 193 in ```train.py``` to use the model of your choice. Be sure to change the ```TRIAL_NAME``` variable on line 19 to use your desired name so that models get saved to the correct folder. 

```python3 train_single.py``` will do training for SingleNet. Be sure to change the ```TRIAL_NAME``` variable on line 18 to use your desired name so that models get saved to the correct folder. 

```python3 graph.py``` will display our results that were obtained for the paper. 
