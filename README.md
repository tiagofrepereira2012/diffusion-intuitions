# Generative model intuitions

This repository contains some code snippets and explanations to help understand generative models (applied to images).

In a nutshell what is aimed in generative models is to find way to create some latent representation $z$ over the underlying data distribution $p(x)$, so that we can generate new samples $x$ from this distribution.

There are several ways to build such representations and the most popular ones are the ones based on neural networks. In this repository we will focus on the most popular ones: Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs) and Diffusion Models (DM).