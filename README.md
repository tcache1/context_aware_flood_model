Fast urban pluvial flood models are necessary for a range of applications, such as near real-time flood nowcasting or processing large 
rainfall ensembles for uncertainty analysis. Data-driven models can help overcome the long computational time of traditional flood simulation 
models, and the state-of-the-art models have shown promising accuracy. Yet the lack of generalizability of urban pluvial flood data-driven 
models to both terrain and rainfall events still limits their application. These models usually adopt a patch-based framework to overcome 
multiple bottlenecks, such as data availability and computational and memory constraints. However, this approach does not incorporate contextual 
information of the small image patch (typically 256 m x 256 m). We propose a new deep-learning model that maintains the high-resolution 
information of the local patch and incorporates a larger context to increase the visual field of the model with the aim of enhancing the 
generalizability of urban pluvial flood data-driven models. We trained and tested the model in the city of Zurich (Switzerland), at a spatial 
resolution of 1 m, for 1-hour rainfall events at 5 min temporal resolution. We demonstrate that our model can faithfully represent flood 
depths for a wide range of rainfall events, with peak rainfall intensities ranging from 42.5 mm~h{-1} to 161.4 mm~h{-1}. Then, we assessed 
the model's terrain generalizability in distinct urban settings, namely Luzern (Switzerland) and Singapore. The model accurately identifies 
locations of water accumulation, which constitutes an improvement compared to other deep-learning models. Using transfer learning, the model 
was successfully retrained in the new cities, requiring only a single rainfall event to adapt the model to new terrains while preserving 
adaptability across diverse rainfall conditions. Our results indicate that by incorporating contextual terrain information into the local 
patches, our proposed model effectively generates high-resolution urban pluvial flood maps, demonstrating applicability across varied terrains 
and rainfall events.


