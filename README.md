# Context-Aware Urban Pluvial Flood Model

Fast urban pluvial flood models are necessary for a range of applications, such as near real-time flood nowcasting or processing large rainfall ensembles for uncertainty analysis. Data-driven urban flood models can help overcome the long computational time of traditional flood simulation models, yet their lack of generalizability to both terrain and rainfall events still limits their application. We adopted a patch-based framework that overcomes multiple bottlenecks (such as data availability and computational and memory constraints) and extended it to incorporate the spatial contextual information of the small image patch (typically 256 m x 256 m). Thus, our new deep-learning model maintains the high-resolution information of the local patch and incorporates a larger context to increase the visual field of the model with the aim of enhancing the generalizability of urban pluvial flood data-driven models. 

We trained and tested the model for various terrain and rainfall events. The results are presented in the paper 'Enhancing generalizability of data-driven urban flood models by incorporating contextual information'. Our results indicate that the proposed model effectively generates high-resolution urban pluvial flood maps, demonstrating applicability across varied terrains and rainfall events.

## Model Description
The objective of the model is to extract and combine the information from the high-resolution local patch, its context and the rainfall times series to emulate the corresponding flood map. To achieve this, we developed a joint model that couples different types of neural networks and learns dependencies between the local patch and its context. The model consists of the following components: (i) three convolutional encoders that extract latent information from the multi-scale spatial features; (ii) an attention mechanism that measures the correlation between the local patch and its context; (iii) a recurrent neural network (RNN) that analyzes the rainfall time series; and (iv) a decoder that converts the extracted information from both the terrain and rainfall data into the flood depth prediction.

Schematic Diagram of the Modeling Framework
![figure_framework_summary.jpg](https://github.com/tcache1/urban_flood_cnn/blob/7a38cc1c982e02baff7a1d3b810107067daae377/figure_framework_summary.jpg?raw=True)

## Model Reproducibitily
The files necessary to reproduce the results are available [here](https://doi.org/10.5281/zenodo.10688079). 

The files include: 
- some data to train or test the model;
- the models weights;
- the saved patch locations (see [line 307](https://github.com/tcache1/urban_flood_cnn/blob/7a38cc1c982e02baff7a1d3b810107067daae377/model_script.py));
- the singularity container. 
