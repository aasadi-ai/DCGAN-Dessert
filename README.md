# DCGAN-Dessert
### A Deep Convolutional Generative Adversarial Network for Dessert Images

## Project Outline:
The goal of the DCGAN-Dessert project is to generate convincing dessert images which will later be extended with a Conditional Text Generation Model to output recipes.

## File Structure:
* __Models__
  * Generator:Implements the Generator using ConvTranspose2d layers for upsampling
  * Discriminator: Implements a basic Image Classifier CNN
  * modelSetup: Initilizes network weights with those recommended in the [DCGAN Paper](https://arxiv.org/pdf/1511.06434v2.pdf)
* __unitTest__: Checks for shape errors in forward passes of networks
* __trainDCGAN__: Trains the Discriminator and Generator using the hyperparameters and transforms recommended in the DCGAN paper.

## To-do:
* Scrape Images of Desserts
* Change models to make them translation invariant
* Build Conditional Text Generation Model

## Attribution:
The current implementation is closely modelled on the PyTorch implementation of [Aladdin Persson](https://github.com/aladdinpersson) and follows closely the training recommendations of the original DCGAN paper.

