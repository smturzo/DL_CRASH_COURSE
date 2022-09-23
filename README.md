# DL\_CRASH\_COURSE
Disclaimer: None of the course material are mine. These materials were found after exhaustively searching the internet and what seemed to be free and relevant to the work we do.
Note: The directories of the individuals are people from the Lindert Lab who showed interest in working through this crash course with me. For the interested individuals I created a directory (per person with their first name). The created directories for the respective individuals to upload their tutorials, examples, notes, assignments, papers and/or problems they want to talk about.

## Week 1:
### Lectures:
- For lecture and reading material, please go through this:
- - https://www.youtube.com/watch?v=QyFrYUCXbgI (lecture).
- - https://www.deeplearningbook.org/contents/mlp.html (this is the related chapter for the lecture).

## Week 2: 
For this focus on logistic model for classification problem with the MNIST dataset. 

#### Pytorch installation guides on OSC
1\. module load miniconda3
2\. module load cuda/10.2.89
3\. conda create --name pytorchenv
4\. conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
5\. conda activate pytorchenv

Note: Alternatively you can also do this on Google Colab. However, at some point you will need to switch over to OSC when you actually use it on your own dataset at some
point in the course.

### Tutorials:
- https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans/lesson/lesson-1-pytorch-basics-and-linear-regression (This covers the basics of pytorch, can be skimmed through super fast.)
- https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans/lesson/lesson-2-working-with-images-and-logistic-regression
- - This tutorial is more fun and covers the following:
- - - Working with images in PyTorch (using the MNIST dataset).
- - - Splitting a dataset into training, validation, and test sets.
- - - Creating PyTorch models with custom logic by extending the nn.Module class.
- - - Interpreting model outputs as probabilities using Softmax and picking predicted labels.
- - - Picking a useful evaluation metric (accuracy) and loss function (cross-entropy) for classification problems.
- - - Setting up a training loop that also evaluates the model using the validation set.
- - - Testing the model manually on randomly picked examples.
- - - Saving and loading model checkpoints to avoid retraining from scratch.

