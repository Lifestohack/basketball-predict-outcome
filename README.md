# Convolutional Neural Network for Video Analysis
# Can a network celebrate early?

Players in basketball games are praised when they shoot three pointers. What makes these player lookeven prodigious is when they shoot the ball and don’t even look at the hoop if the ball goes insideand they already start celebrating. Usually the ball is in the air for 1.2 seconds, but the players startscelebrating at around 0.5 second mark. This paper is thus inspired from the premature celebrationof those player. When the ball is still in the air halfway through we try to predict the outcome andcelebrate early.To investigate this phenomenon in depth, different networks architecture are considered.  As weare working with videos, 3d convolutional network seems to be an obvious choice. The frames areflattened and used as input to the FFNN. As suggested by Carreira and Zisserman [2017] two stream3D-Convolution network is also used. We discovered that two stream approach is really effective i.e.one stream uses the images and other stream uses the dense optical flow of those images feed through3d convolutional network and at last they are combined together and classified. LSTM is also a clearchoice as it is one of the most widely used algorithm to solve time series problems.

## Feed forward neural network (FFNN)
FFNN classification algorithm is implemented as a benchmark,every neuron in a layer is connected with all other neuron in the previous layer. 

## 3d convolutional neural network (Conv3d)
Working with videos, Conv3d which applies convolutions in the 3D space where lastdimension is time is also implemented.

## Conv2d long short term memory (CONV2dLSTM)
Each frames moves through layers of 2dconvolutional neural network.  Output of 2d convolutional layer is then feed into long short termmemory (LSTM) network with many to one architecture in time series prediction. So it either predictshit or miss just by looking at all previous time series even with 30 or 55 frames without actuallylooking the ball hitting the hoop. In this way we can extract features from the given frames using 2dconvolution and these features sequence pass through LSTM for prediction.

## Two-stream conv3d (TWOSTREAM): 
This network takes two streams of inputs, one stream takesthe time series images and second stream takes the dense optical flow time series images created usingthe input image of the first stream. Optical-flow quantifies the motion of an object. Dense opticalflow using algorithm as suggested by Farnebäck [2003] is calculated which computes magnitude anddirection of optical flow vector for each pixel of each frame which changes between frames. Andthose vectors are visualized with colour to the direction in which they are moving. Output of thesetwo streams are then combined together and classified.

Two networks are implemented where the coordinates and radius of the ball is used as an input.First a classifier **(PositionFFNN)** is implemented to classify hit or miss using 100 frames. Second,for lower than 100 frames as an input, LSTM **(PositionLSTM)** with many to many architecture isused to predict the next missing sequence by providing previous time series data which can then beclassified using PositionFFNN for making predictions. After calculating loss using mean squarederror, back-propagation happens separately for each view and only during classification they areconcatenated. While back-propagating, the hidden state of the LSTM is detached from the previousstate because if we don’t, we’ll back-propagate all the way to the start even after going throughanother batch and we don’t want that. 

(PDF) Convolutional Neural Network for Video Analysis. Can a network celebrate early?. Available from: https://www.researchgate.net/publication/344442360_Convolutional_Neural_Network_for_Video_Analysis_Can_a_network_celebrate_early.


DOI: 10.13140/RG.2.2.19060.58241
