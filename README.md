# hakare
Model to classify handwritten Japanese characters from the dataset ETL9.
## Data for training

ETL9G contains 607200 images of 3036 unique Japanese characters (hiragana and kanji),
128x127 pixels greyscale.
The data is not included in this public repository due to license constraints.

The preproceesing pipeline center-crops the images to 90x90 pixels and saves the data in a hdf5 file.
Each image is also normalised with respect to the mean and std from the training set.
## Training Models
Assuming you have obtained the data and it is located  in

    data/ETL-9-90x90.hdf5

launch training by calling the model.fit() function, e.g.

    model.fit(None, 40, 2)

to train the winning configuration from study 2 (i.e. the best model from our hyperparameters
optimisation study for 40 epochs. This call returns the trained model with the least loss
on the validation set and also saves it under

    data/models/model.pt


You can change the path where the input data is located in data.py
    PATH_TO_DATA
and the output path in model.py
    OUT_PATH

