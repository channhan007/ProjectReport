# CS342 Deep Learning: Final Project Write-Up

### Introduction

For CS342 Deep Learning's final project, our team built a SuperTux Kart agent using a neural network to play a game of hockey. In order to create this agent, we identified two main components of work: 1) generating a network to learn and predict useful data to inform our agent's next action and 2) using the network's output in conjunction with the state of the agent to make decisions on the next action to take. Having spent some time playing the SuperTux Kart hockey game and learning the game's API, we considered a variety of strategies for both components of work - from how we could create training data to using heatmap scores to more accurately gauge an agent's confidence of detecting the puck. In this write-up, we will discuss our strategies, network architecture, and learnings from building our agent.

### Discovery

This project's open-ended nature lent itself experimentation, and we iterated continuously on data collection, network architecture and performance, and control strategies for our agent.

##### Generating Training Data

Before we could spend time tuning our network or manipulating data to control our agent, we needed to collect training data. Our initial approach was to generate training data by allowing the AI to play the game for a number of steps and to save each frame as well as the corresponding aim point that would allow our agent to steer towards the puck. The aim point was defined as a point on the frame, calculated using the game's internal state data, specifically the kart's view and projection data as well as the absolute location of the puck. While this data is not sufficient to allow our agent to reliably push the puck into our scoring goal, our initial concern was enabling our agent to consistently move towards the puck. 

##### Initial Architecture

With the training data generated from AI play, we were able to design a network that could learn to predict an aim point for our agent given a frame of gameplay. This initial network was a relatively lightweight convolutional neural network that used 4 layers of a convolutional block, containing a 2D batch normalization layer, a 2D convolutional layer, and a rectified linear unit (ReLU). The predicted aim point was generated using a convolutional layer After training this network using approximately 10,000 training samples, L1 loss, and Adam optimization for around 60 epochs, we saw loss decrease consistently, and decided to use this model as a starting point for our agent's controller.

##### Incorrect Puck Predictions

Using a rudimentary controller to calculate steering direction from the predicted aim point, acceleration adjusted for proximity to the puck, and a skid threshold to increase maximum turning angle, our agent was able to drive towards the puck, at least initially. However, once the puck was no longer in the frame, the predicted aim point was extremely unreliable. Not only had our network not learned how to predict an aim point when the puck was not in view, it started to predict aim points based other, non-puck objects. The result was that when our agent lost the puck, it would often drive to the edges of the arena, become stuck, but continue to predict an aim point for a non-existent puck beyond the field of play. 

Hypothesizing that perhaps our network had not learned to identify the puck accurately enough, we trained our network for an additional 60 epochs and saw loss continue to lessen, albeit with diminishing returns. Yet, when we tested this new model, we saw similar results - the agent could track the puck as long as it was in frame but only when it was in frame. To remedy this, we attempted to modify our training data to provide a more consistent training signal by mapping the aim point for all instances in which the puck was not in frame to a specify aim point, e.g. [1, 0]. The intent was to help the network learn to predict a default, recovery aim point when no puck existed; using this default aim point, we expected to perform a routine in the controller to scan for the puck until it was found. However, even after training both the initial network as well as a more complex convolutional network (using residual connections and more aggressive data augmentation), the resulting aim points continued to vary considerably when no puck could be detected.

Having seen the limitations of a network that predicted just an aim point, we realized our network could not be relied upon to return reliable data when no puck was in frame; furthermore, the aim point returned did not lend itself to additional inference. From the perspective of our controller, it was impossible to tell whether the aim point the network returned was valid. 

##### Object Detection Architecture

The problem of not knowing whether the network's prediction could be trusted led us to consider other types of predictions that could be more easily manipulated and interpreted in our controller. Instead of predicting just an aim point, we wanted to know a confidence level in addition to the predicted puck location. Therefore, instead of a network that predicted aim points, we created a network to predict heatmaps. And, by using heatmaps, we could find the location with the highest score, calculate an aim point, and know whether to trust this prediction. In analyzing the score for the top result of heatmaps when the puck was and was not in frame, we saw something that resembled a binomial distribution, and confirmed that we could set a threshold to know when and when not to trust the network's prediction.

##### Final Architecture

Our final network architecture implemented point detection to identify the puck and return a predicated heat map. The network uses two blocks (composed of a 2D convolutional layer, a 2D batch normalization layer, and a rectified linear unit (ReLU)), residual connections, and corresponding up-convolution layers. A convolutional classifier layer allows us to return a predicted heatmap for the puck location. Additionally, we used data augmentation in the form of color jitter and random horizontal flips to ensure our model was less likely to over-fit our training data. Using 30,000 training images and corresponding heat maps from two AI agents playing against one another, we trained for 100 epochs. The resulting model was able to generate a heatmap for the puck in a relatively short amount of time; however, the limited number of channels and the relatively few layers caused the predicted heatmap's accuracy to suffer. In addition to the model's output, we needed to implement strategies in our controller logic to account for the network's shortcomings.

### Controller Strategies Attempted

Besides network architecture, the logic to interpret the network's output and return an action required multiple iterations. We implemented a variety of strategies and found that some ideas, while promising in theory, relied too heavily on perfect exact aim points, which a network is unlikely to provide. While experimenting with different network architectures, we used aim points generated from the game's world state to implement a variety of controller strategies and judge their efficacy against ground truth aim points.

##### World State Assumptions

Early on, we identified an additional challenge beyond detecting the puck and steering towards it - driving the puck to the opponent's goal. In order to identify which goal we should shoot on, we initialized our agents with assumptions of world state depending on whether the agent was part of team one or team two. Knowing the goal coordinates allowed our agent to create an aim point for the goal; however, when our agent was turned around, the coordinates aligned with our own goal. The result was our agent would then attempt to score own goals. To compensate for this, we used the agent's rotation data to generate a yaw value for each frame. Comparing the yaw value when our agent was initialized with the current yaw, we were able to discern which direction the agent was facing.

##### Cut-Angle Calculation

Inspired by 'cut angles' used in billiards, we generated a cut angle to allow our agent to drive towards the puck and knock it in the direction of the opponent's goal. Cut angles work by hitting the puck off center, but in order to do so, we needed to know how much to adjust the aim point and in what direction. Knowing aim points of our kart, the center of the opponent's goal, and the puck, we were able to calculate a goal-kart-puck angle. Additionally, the relative locations of the puck and goal's center indicated whether this adjustment should be made to the left or right. At first, we employed cut angle regardless of how far the kart was from the puck; later, we found this strategy to be more effective only when the kart was sufficiently close to the puck. This approach worked fairly well when our agent was facing the opponent's goal with the puck in between the two, but was much less effective when the kart was turned around or the cut angle was too large. 

##### Recovering

Once our agent lost the puck, it often drove into a wall with no strategy to recover. Using the model's predicted heatmap and extracting the maximum value, we were able to determine whether our agent sees the puck or not. However, the model's output was not sufficiently accurate to be confident in the predicted aim point. To compensate for the model's inaccuracy, we implemented a recovery routine that relied on analyzing the agent's previous states.

Starting at the beginning of each game, our agent saved state data, including the confidence level, predicted aim point, and action taken, for each frame. Using these previous states, our agent considered the last several states to determine if our agent had lost the puck or was stuck against the wall. In either of those cases, an override routine would be initiated where the agent would reverse and turn for a given number of frames. While this approach works some of the time, we found that our model was slow to indicate if the puck was out of frame. If our model were to return heat maps with higher peaks, our agent would be able to enter the recovery routine in a more timely manner.

### Lessons and Future Ideas

In working on our project, we learned lessons from experiments that did not work out and had additional ideas to implement in the future.

##### Quality Model Output vs. Resilient Controller Logic

Throughout the project, we oscillated between optimizing our network to be more accurate and more confident in its predictions and implementing strategies in our controller to have a more robust set of behaviors our agent could use. As our model improved, it opened up new avenues to purse in our controller logic. And, as our controller logic became more nuanced, we gained clarity on how we could further improve the predictions. While this approach worked, it required many iterations to make improvements in our agent's performance. With more time and different tools available, we might have considered using imitation learning to have our model learn behaviors directly. 

##### More Routines

In finding some success with the recovery routine pattern, it seemed reasonable that our agent could use other default routines to improve its behavior. Beyond knowing how to turn around when lost or stuck, our agent could have used a 'search' routine to stay closer to the center of the field and scan for the puck. Additionally, an 'approach' routine would have been useful to help the agent position itself when no cut angle was available given the kart's and puck's locations. The strategy of queueing a set of actions to take in coming frames based on analyzing recent patterns in the model's predictions seemed to be a promising way to enable more sophisticated actions for our agent.   

##### Sharing Detections Between Agents

Because our team was composed of two instances of our agent, we contemplated using each player's knowledge of the puck's location as a way to help either agent if it were to lose the puck. For example, if Player A were to have low confidence in the puck's location but Player B were to have high confidence, Player A could use Player B's location data to navigate towards the puck. By saving each agent's location coordinates as a class variable, we could share data between the instances of our agent. While this idea seemed worthwhile, we prioritized more basic functionality and saved this as a future optimization.

### Conclusion

This project offered an opportunity to attempt a variety of neural network architectures, label data generation approaches, training techniques, and game strategies that built upon previous lessons in this class. While our agent's performance was not what we had hoped for, the process of implementing an idea, assessing the result, and pivoting to a new idea helped us build fluency with deep learning concepts and implementation details. As we continued to iterate, we built momentum and found new ideas that led to deeper understanding of tradeoffs when designing a neural network. 
