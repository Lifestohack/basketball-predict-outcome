# Convolutional Neural Network for Video Analysis
# Can a network celebrate early?

Players in basketball games are praised when they shoot three pointers. What makes these player lookeven prodigious is when they shoot the ball and don’t even look at the hoop if the ball goes insideand they already start celebrating. Usually the ball is in the air for 1.2 seconds, but the players startscelebrating at around 0.5 second mark. This paper is thus inspired from the premature celebrationof those player. When the ball is still in the air halfway through we try to predict the outcome andcelebrate early.To investigate this phenomenon in depth, different networks architecture are considered.  As weare working with videos, 3d convolutional network seems to be an obvious choice. The frames areflattened and used as input to the FFNN. As suggested by Carreira and Zisserman [2017] two stream3D-Convolution network is also used. We discovered that two stream approach is really effective i.e.one stream uses the images and other stream uses the dense optical flow of those images feed through3d convolutional network and at last they are combined together and classified. LSTM is also a clearchoice as it is one of the most widely used algorithm to solve time series problems
    
Read more on outcome here:
https://www.researchgate.net/publication/344442360_Convolutional_Neural_Network_for_Video_Analysis_Can_a_network_celebrate_early

DOI: 10.13140/RG.2.2.19060.58241
