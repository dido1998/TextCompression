# TextCompression
This is a repo dedicated for demonstrating a possible technique for Text Compression.

## Dataset
I have used the image_coco.txt for training which can be obtained from-
```
https://github.com/geek-ai/Texygen/blob/master/data/image_coco.txt
```
This dataset contains sentence of about 100 characters long.
I have been able to compress each sentence into a vector of size 5.

## Method
I have used the encoder-decoder architecture with a few tweaks which are explained in the comments with the code.

## Results
1 
-*Input sentence*: three dogs by a screen door that has a pet entrance in it leading to the outside of the house
-*compressed vector*: [0.82584465 0.8639319  0.03209151 0.6374205  0.53023213]
-*Reproduced sentence*: three dogs by a screen door that has a pet entrance in it leading to the outside of the house <eos>
  
2 Input sentence: a woman with short brown hair is looking into a circular mirror and holding a camera up to her cheek
 compressed vector: [1.5469722  0.7479113  0.5978319  0.         0.91678953]
 Reproduced sentence: a woman with short brown hair is looking into a circular mirror with a camera in her hand <eos>

3Input sentence: a kitchen with wooden cabinets and a gas range  
 comressed vector: [3.8299806 1.2600164 0.        3.721849  1.0740554]
 Reproduced sentence: a man with a bandana is riding a motorcycle <eos>
